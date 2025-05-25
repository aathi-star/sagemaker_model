"""3D Generation Training Script with MaskGIT3D and TCGA"""
import os
import torch
import argparse

# Set matmul precision for compatible GPUs (e.g., Ampere)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import clip

# Suppress Dynamo errors and fallback to eager execution if compilation fails
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Custom imports
from Network.clip_ulip import ULIP2Wrapper
from Losses.facet_loss import FACETLoss

# Custom imports
from datasets.modelnet_loader import ModelNet10Dataset
from Network.transformer_3d import MaskGIT3D

def parse_args():
    parser = argparse.ArgumentParser(description='3D Generation Training with MaskGIT3D and TCGA')
    
    # Model parameters
    parser.add_argument('--model', default='maskgit3d', choices=['maskgit3d'])
    parser.add_argument('--embed_dim', type=int, default=768, help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP expansion ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for 3D input')
    parser.add_argument('--masking_strategy', type=str, default='random', choices=['random', 'halton'], help='Masking strategy for training (random or halton)')
    parser.add_argument('--tcga_ratio', type=float, default=0.5, help='Ratio of TCGA attention heads')
    
    # ULIP-2 parameters
    parser.add_argument('--use_ulip2', action='store_true', help='Use ULIP-2 for text conditioning')
    
    # FACET loss parameters
    parser.add_argument('--use_facet', action='store_true', help='Use FACET loss for fairness')
    parser.add_argument('--facet_temp', type=float, default=0.07, help='Temperature for FACET contrastive loss')
    parser.add_argument('--lambda_fair', type=float, default=0.5, help='Weight for fairness loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before an optimizer update.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='D:/Halton-MaskGIT/data/ModelNet10', help='Path to ModelNet10 dataset')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points to sample from each 3D model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in ModelNet10')
    parser.add_argument('--target_classes', type=str, default=None, help='Comma-separated list of classes to train on (e.g., chair,table). Default is all classes.')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='outputs/3d_generation', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint save frequency (epochs)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Masked modeling
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio')
    # Note: --tcga_ratio is defined above
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--compile_model', action='store_true', help='Enable torch.compile for the model')

    # Resume and Early Stopping
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from (e.g., outputs/checkpoint_epoch_10.pth)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs to wait for validation loss to improve before stopping. 0 to disable.')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                        help='Minimum change in validation loss to qualify as an improvement for early stopping.')
    
    return parser.parse_args()

def set_seed(seed):
    """Set all random seeds"""
    import random  # Import directly in the function to ensure availability
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(args):
    """Create model with specified parameters"""
    # Create text/3D encoder
    if args.use_ulip2:
        # Load CLIP model
        try:
            clip_model_name = "ViT-B/32"
            clip_model, _ = clip.load(clip_model_name, device=args.device)
            print(f"Successfully loaded CLIP model: {clip_model_name} to device: {args.device}")
        except Exception as e:
            print(f"Error loading CLIP model {clip_model_name}: {e}")
            print("Please ensure the 'clip' package is installed and the model name is correct.")
            raise e

        text_encoder = ULIP2Wrapper(preloaded_clip_model=clip_model)
        text_encoder.eval()  # Freeze ULIP-2/CLIP weights
        text_dim = clip_model.visual.output_dim # Get text_dim from the loaded CLIP model
    else:
        text_encoder = None
        text_dim = 0
    
    # Create 3D generator
    model = MaskGIT3D(
        in_channels=1,  # Single channel for binary voxels
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        tcga_ratio=args.tcga_ratio,
        text_encoder=text_encoder,
        text_dim=text_dim,
        masking_strategy=args.masking_strategy
    )
    
    # Create FACET loss if enabled
    if args.use_facet:
        facet_loss = FACETLoss(
            temperature=args.facet_temp,
            lambda_fair=args.lambda_fair
        )
    else:
        facet_loss = None
    
    return model, facet_loss

def prepare_dataloaders(args):
    """Prepare training and validation dataloaders"""
    target_classes_list = None
    if args.target_classes:
        target_classes_list = [cls.strip() for cls in args.target_classes.split(',') if cls.strip()]
        if not target_classes_list:
            target_classes_list = None # Ensure it's None if string was empty or only commas/whitespace
        print(f"Targeting classes: {target_classes_list if target_classes_list else 'All'}")

    # Load ModelNet10 dataset
    train_dataset = ModelNet10Dataset(
        root_dir=args.dataset,
        split='train',
        num_points=args.num_points,
        target_classes=target_classes_list
    )
    
    val_dataset = ModelNet10Dataset(
        root_dir=args.dataset,
        split='test',
        num_points=args.num_points,
        target_classes=target_classes_list
    )
    
    print(f'Loaded ModelNet10 dataset with {len(train_dataset)} training and {len(val_dataset)} validation samples')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, facet_loss, loader, criterion, optimizer, scaler, epoch, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_fair_loss = 0.0
    
    optimizer.zero_grad() # Zero gradients at the beginning of the epoch

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{args.epochs}")
    for step, batch in pbar:
        # Move data to device
        points = batch['points'].to(args.device)  # ModelNet10 provides points, not voxels
        text = batch.get('text', None)
        category = batch.get('category', None)  # Get category instead of protected_attr
        
        # Forward pass
        with torch.amp.autocast(device_type=args.device, dtype=torch.float16 if args.fp16 else None, enabled=args.fp16):
            if args.use_ulip2 and text is not None:
                # For ULIP-2, text_emb is already handled internally by MaskGIT3D if text is passed
                # The model's forward pass should return both reconstructed points and text_emb if needed for FACET
                outputs, text_emb_for_facet = model(points, text=text, mask_ratio=args.mask_ratio, return_text_emb=True)
            elif args.use_ulip2: # ULIP-2 used, but no text for this batch (should not happen if dataset provides text)
                outputs = model(points, mask_ratio=args.mask_ratio)
                text_emb_for_facet = None # No text, no text_emb for FACET
            else: # Not using ULIP-2
                outputs = model(points, mask_ratio=args.mask_ratio)
                text_emb_for_facet = None

            # Point cloud reconstruction loss - compute chamfer distance
            # Make sure both are same shape (B, N, 3)
            if outputs.shape != points.shape:
                # If we got fewer points back, sample from original to match
                if outputs.shape[1] < points.shape[1]:
                    # Randomly sample points to match output shape
                    idx = torch.randperm(points.shape[1])[:outputs.shape[1]]
                    points_sampled = points[:, idx, :]
                    loss = F.mse_loss(outputs, points_sampled)
                else:
                    # Trim excess output points
                    outputs_trimmed = outputs[:, :points.shape[1], :]
                    loss = F.mse_loss(outputs_trimmed, points)
            else:
                # Same shape, direct comparison
                loss = F.mse_loss(outputs, points)
            
            # FACET loss if enabled
            if facet_loss is not None and text_emb_for_facet is not None and category is not None:
                # Convert category to tensor if it's a string
                if isinstance(category, str):
                    # Use label as proxy for protected attribute
                    label_tensor = batch['label'].to(args.device)
                    fair_loss = facet_loss(text_emb_for_facet, label_tensor)
                else:
                    fair_loss = facet_loss(text_emb_for_facet, batch['label'].to(args.device))
                    
                loss = loss + args.lambda_fair * fair_loss
                total_fair_loss += fair_loss.item()
                total_ce_loss += (loss - args.lambda_fair * fair_loss).item() # Store original recon loss
            elif facet_loss is not None: # FACET enabled but no text_emb or category, log CE loss as main loss
                total_ce_loss += loss.item()
            
            # Normalize loss for accumulation
            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps
        
        # Backward pass
        # optimizer.zero_grad() # Moved to conditional update block
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (conditional on accumulation_steps)
        if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(loader):
            if args.fp16:
                if args.clip_grad is not None:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            
            optimizer.zero_grad() # Zero gradients after optimizer step
        
        # Update metrics
        # Note: loss here is potentially scaled by accumulation_steps.
        # For logging, we want the unscaled loss or the average of unscaled losses.
        # total_loss accumulates the scaled loss. We will divide by len(loader) at the end.
        total_loss += loss.item() * args.accumulation_steps # Accumulate the original-scale loss for epoch avg
        avg_loss = total_loss / (step + 1)
        
        # Update progress bar
        metrics = {'loss': avg_loss}
        if facet_loss is not None:
            metrics.update({
                'ce_loss': total_ce_loss / (step + 1),
                'fair_loss': total_fair_loss / (step + 1)
            })
        pbar.set_postfix(metrics)
    
    return {
        'loss': total_loss / len(loader),
        'ce_loss': total_ce_loss / len(loader) if facet_loss is not None else 0,
        'fair_loss': total_fair_loss / len(loader) if facet_loss is not None else 0
    }

def validate(model, loader, criterion, args):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            points = batch['points'].to(args.device)
            text = batch.get('text', None) # Assuming text might be used in validation too
            category = batch.get('category', None)

            with torch.amp.autocast(device_type=args.device, dtype=torch.float16 if args.fp16 else None, enabled=args.fp16):
                if text is not None:
                    if isinstance(category, str):
                        text = [category]
                    outputs, _ = model(points, text=text, mask_ratio=0.0) # No masking for validation
                else:
                    outputs = model(points, mask_ratio=0.0)  # No masking for validation
                
                # Point cloud reconstruction loss
                if outputs.shape != points.shape:
                    # If we got fewer points back, sample from original to match
                    if outputs.shape[1] < points.shape[1]:
                        # Randomly sample points to match output shape
                        idx = torch.randperm(points.shape[1])[:outputs.shape[1]]
                        points_sampled = points[:, idx, :]
                        loss = F.mse_loss(outputs, points_sampled)
                    else:
                        # Trim excess output points
                        outputs_trimmed = outputs[:, :points.shape[1], :]
                        loss = F.mse_loss(outputs_trimmed, points)
                else:
                    # Same shape, direct comparison
                    loss = F.mse_loss(outputs, points)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def save_checkpoint(state, filename='checkpoint.pth'):
    """Save training checkpoint"""
    torch.save(state, filename)

def main():
    # Import nn directly in the function for explicit access
    import torch.nn as nn
    
    # Parse arguments and setup
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create model and losses
    model, facet_loss = create_model(args)
    model = model.to(args.device)

    # Store original model reference for saving/loading state_dict, especially if compiled
    _original_model = model 

    if args.compile_model:
        # Check PyTorch version, as torch.compile is available from 2.0
        pt_version_major = int(torch.__version__.split('.')[0])
        pt_version_minor = int(torch.__version__.split('.')[1])
        if pt_version_major >= 2:
            print(f"Attempting to compile model with torch.compile() (PyTorch version: {torch.__version__})...")
            try:
                # 'default' mode is a good balance.
                model = torch.compile(model, mode='default') 
                print("Model compiled successfully.")
            except Exception as e:
                print(f"torch.compile() failed: {e}. Continuing without compilation.")
                # model variable remains the original uncompiled model in this case
        else:
            print(f"torch.compile() requires PyTorch 2.0 or newer. Current version: {torch.__version__}. Skipping compilation.")

    if facet_loss is not None:
        facet_loss = facet_loss.to(args.device)
    
    # Print model summary (use _original_model for consistent parameter counting)
    total_params = sum(p.numel() for p in _original_model.parameters())
    trainable_params = sum(p.numel() for p in _original_model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Print ULIP-2 and FACET status
    print(f"Using ULIP-2: {args.use_ulip2}")
    print(f"Using FACET loss: {args.use_facet}")
    if args.use_facet:
        print(f"  - FACET temperature: {args.facet_temp}")
        print(f"  - Lambda fair: {args.lambda_fair}")
    
    # Setup dataloaders
    train_loader, val_loader = prepare_dataloaders(args)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=args.fp16)
    
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0 # For early stopping

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device) 
        
        # Load state into the original model structure
        model_to_load = _original_model
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Warning: Could not load model_state_dict directly: {e}. Attempting to load with strict=False.")
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # If model was successfully compiled, the 'model' variable (compiled wrapper) will use these new weights.
        # If compilation failed or was skipped, 'model' is already '_original_model'.

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: Optimizer state not found in checkpoint.")

        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e: 
                print(f"Warning: Could not load scheduler_state_dict: {e}. Scheduler will start fresh.")
        else:
            print("Warning: Scheduler state not found in checkpoint or is None. Scheduler will start fresh.")

        start_epoch = checkpoint.get('epoch', -1) + 1 
        best_val_loss = checkpoint.get('best_val_loss', float('inf')) 
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

        print(f"Resumed from epoch {start_epoch -1}. Training will start from epoch {start_epoch}.")
        print(f"Resumed with best_val_loss: {best_val_loss:.4f}, epochs_no_improve: {epochs_no_improve}")
    else:
        if args.resume_checkpoint: 
            print(f"Warning: Checkpoint file not found at {args.resume_checkpoint}. Starting training from scratch.")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_metrics = train_epoch(model, facet_loss, train_loader, criterion, optimizer, scaler, epoch, args)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, args)
        
        # Update learning rate
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        # Prepare checkpoint data for this epoch
        current_checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': _original_model.state_dict(), # Always save the original model's state
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_loss,
            'best_val_loss': best_val_loss, 
            'epochs_no_improve': epochs_no_improve, 
            'args': vars(args) # Save args as a dictionary
        }

        # Log metrics
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Log FACET metrics if enabled
        if args.use_facet:
            writer.add_scalar('Loss/train_ce', train_metrics['ce_loss'], epoch)
            writer.add_scalar('Loss/train_fair', train_metrics['fair_loss'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(current_checkpoint_data, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save best model
        if val_loss < best_val_loss - args.early_stopping_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Update checkpoint data for saving the best model
            current_checkpoint_data['best_val_loss'] = best_val_loss
            current_checkpoint_data['epochs_no_improve'] = epochs_no_improve
            save_checkpoint(current_checkpoint_data, os.path.join(args.output_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            # Update epochs_no_improve in current_checkpoint_data for regular checkpoint saving
            # This ensures the counter is saved even if it's not the best model this epoch
            current_checkpoint_data['epochs_no_improve'] = epochs_no_improve
        
        # Print epoch summary
        log_str = f"Epoch {epoch+1}/{args.epochs} | "
        log_str += f"Train Loss: {train_metrics['loss']:.4f} | "
        log_str += f"Val Loss: {val_loss:.4f}"
        
        if args.use_facet:
            log_str += f" | CE: {train_metrics['ce_loss']:.4f}"
            log_str += f" | Fair: {train_metrics['fair_loss']:.4f}"
            
        print(log_str)

        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement on validation loss (patience: {args.early_stopping_patience}).")
            break
    
    # Save final model state, regardless of early stopping
    # Ensure train_metrics and val_loss are defined, e.g. if training loop didn't run (0 epochs)
    final_train_loss = train_metrics['loss'] if 'train_metrics' in locals() and train_metrics else float('nan')
    final_val_loss = val_loss if 'val_loss' in locals() else float('nan')

    final_model_data = {
        'epoch': epoch if 'epoch' in locals() else start_epoch -1, # Last completed epoch or epoch before start if loop didn't run
        'model_state_dict': _original_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': final_train_loss,
        'val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'args': vars(args)
    }
    save_checkpoint(final_model_data, os.path.join(args.output_dir, 'final_model_after_training.pth'))
    
    print("Training complete!")

if __name__ == '__main__':
    main()
