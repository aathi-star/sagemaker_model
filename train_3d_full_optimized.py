"""Optimized 3D Generation Training Script with MaskGIT3D and TCGA
   Full ModelNet10 Training for High-Quality 3D Generation
   Optimized for ml.g4dn.2xlarge instance
"""
import os
import torch
import argparse
import time
import datetime

# Set matmul precision for compatible GPUs
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import clip
from torch.cuda.amp import autocast, GradScaler

# Suppress Dynamo errors and fallback to eager execution if compilation fails
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Custom imports
from Network.clip_ulip import ULIP2Wrapper
from Losses.facet_loss import FACETLoss
from datasets.modelnet_loader import ModelNet10Dataset
from Network.transformer_3d import MaskGIT3D

def parse_args():
    parser = argparse.ArgumentParser(description='3D Generation Training with MaskGIT3D on Full ModelNet10')
    
    # Model parameters - OPTIMIZED for quality generation
    parser.add_argument('--model', default='maskgit3d', choices=['maskgit3d'])
    parser.add_argument('--embed_dim', type=int, default=256, help='Transformer embedding dimension')
    parser.add_argument('--depth', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP expansion ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for 3D input')
    parser.add_argument('--masking_strategy', type=str, default='halton', 
                      choices=['random', 'halton'], help='Masking strategy for training')
    parser.add_argument('--tcga_ratio', type=float, default=0.5, help='Ratio of TCGA attention heads')
    
    # ULIP-2/CLIP text conditioning
    parser.add_argument('--use_ulip2', action='store_true', help='Use ULIP-2 for text conditioning')
    parser.add_argument('--freeze_clip', action='store_true', default=True, 
                      help='Freeze CLIP model weights during training')
    
    # FACET loss parameters
    parser.add_argument('--use_facet', action='store_true', help='Use FACET loss for fairness')
    parser.add_argument('--facet_temp', type=float, default=0.07, help='Temperature for FACET contrastive loss')
    parser.add_argument('--lambda_fair', type=float, default=0.5, help='Weight for fairness loss')
    
    # Training parameters - OPTIMIZED for ml.g4dn.2xlarge
    parser.add_argument('--batch_size', type=int, default=16, 
                      help='Batch size - optimized for ml.g4dn.2xlarge')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='Number of training epochs')
    parser.add_argument('--accumulation_steps', type=int, default=1, 
                      help='Number of steps to accumulate gradients before an optimizer update')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, 
                      help='Weight decay - increased for better generalization')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='D:/Halton-MaskGIT/data/ModelNet10', 
                      help='Path to ModelNet10 dataset')
    parser.add_argument('--num_points', type=int, default=2048, 
                      help='Number of points to sample from each 3D model')
    parser.add_argument('--num_classes', type=int, default=10, 
                      help='Number of classes in ModelNet10')
    parser.add_argument('--class_weights', action='store_true', default=True,
                      help='Use class weighting to balance the dataset')
    
    # Data augmentation
    parser.add_argument('--augmentation', action='store_true', default=True, 
                      help='Enable data augmentation')
    parser.add_argument('--rotation_range', type=float, default=15.0, 
                      help='Random rotation range in degrees')
    parser.add_argument('--scale_range', type=float, default=0.1, 
                      help='Random scaling range (proportion)')
    parser.add_argument('--jitter', type=float, default=0.01, 
                      help='Random jitter for point positions')
    parser.add_argument('--point_dropout', type=float, default=0.05,
                      help='Random point dropout probability')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='outputs/3d_generation', 
                      help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--log_freq', type=int, default=10, help='Logging frequency (iterations)')
    parser.add_argument('--save_freq', type=int, default=5, help='Checkpoint save frequency (epochs)')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency (epochs)')
    parser.add_argument('--num_workers', type=int, default=4, 
                      help='Number of data loading workers')
    
    # Masked modeling
    parser.add_argument('--mask_ratio', type=float, default=0.65, 
                      help='Masking ratio - increased for better results')
    parser.add_argument('--progressive_masking', action='store_true', default=True,
                      help='Progressively increase masking ratio during training')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fp16', action='store_true', default=True, 
                      help='Use mixed precision training - enabled by default')
    parser.add_argument('--compile_model', action='store_true', default=False, 
                      help='Enable torch.compile for the model')
    parser.add_argument('--benchmark', action='store_true', default=True,
                      help='Enable cudnn benchmark mode for faster training')

    # Resume and Early Stopping
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                      help='Path to checkpoint file to resume training from')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                      help='Number of epochs to wait for validation loss to improve before stopping')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                      help='Minimum change in validation loss to qualify as an improvement')
    
    return parser.parse_args()

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    # Use benchmark mode if enabled (faster but less reproducible)
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
    else:
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
            
            # Freeze CLIP model if specified
            if args.freeze_clip:
                for param in clip_model.parameters():
                    param.requires_grad = False
                print("CLIP model weights frozen during training")
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
    
    # Create 3D generator with optimized parameters
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
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params/1e6:.2f}M parameters")
    print(f"Architecture: embed_dim={args.embed_dim}, depth={args.depth}, heads={args.num_heads}")
    
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
    """Prepare training and validation dataloaders with full ModelNet10 dataset"""
    # Load ModelNet10 dataset with augmentation
    train_dataset = ModelNet10Dataset(
        root_dir=args.dataset,
        split='train',
        num_points=args.num_points,
        target_classes=None,  # Use all classes
        augmentation=args.augmentation,
        rotation_range=args.rotation_range,
        scale_range=args.scale_range,
        jitter=args.jitter,
        point_dropout=args.point_dropout
    )
    
    val_dataset = ModelNet10Dataset(
        root_dir=args.dataset,
        split='test',
        num_points=args.num_points,
        target_classes=None,  # Use all classes
        augmentation=False  # No augmentation for validation
    )
    
    print(f'Loaded ModelNet10 dataset with {len(train_dataset)} training and {len(val_dataset)} validation samples')
    
    # Calculate class weights if enabled
    class_weights = None
    if args.class_weights:
        # Count instances of each class
        class_counts = np.zeros(args.num_classes)
        for _, label in train_dataset:
            class_counts[label] += 1
        
        # Compute weights inversely proportional to class frequency
        class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-8))
        class_weights = class_weights / class_weights.sum() * args.num_classes
        print(f"Using class weights: {class_weights}")
    
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
    
    return train_loader, val_loader, class_weights

def train_epoch(model, facet_loss, loader, criterion, optimizer, scaler, epoch, args, class_weights=None):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{args.epochs}")
    
    # Progressive masking: Start with easier task (lower mask ratio) and gradually increase
    if args.progressive_masking:
        progress = min(1.0, epoch / (args.epochs * 0.75))  # Reach full mask_ratio at 75% of training
        current_mask_ratio = args.mask_ratio * (0.5 + 0.5 * progress)  # Start at 50% of target mask_ratio
    else:
        current_mask_ratio = args.mask_ratio
    
    for i, (points, labels) in pbar:
        # Move data to device
        points = points.to(args.device)
        labels = labels.to(args.device)
        
        # Clear accumulated gradients if this is the first step in accumulation cycle
        if i % args.accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Use mixed precision training if enabled
        with autocast(enabled=args.fp16):
            # Forward pass with masking
            outputs = model(points, mask_ratio=current_mask_ratio)
            
            # Calculate loss with class weights if available
            if class_weights is not None and hasattr(criterion, 'weight'):
                criterion.weight = class_weights.to(args.device)
                
            loss = criterion(outputs, labels)
            
            # Add FACET loss if enabled
            if facet_loss is not None:
                fair_loss = facet_loss(outputs, labels)
                loss = loss + fair_loss
                
            # Scale loss for gradient accumulation
            loss = loss / args.accumulation_steps
        
        # Backward pass with gradient scaling for mixed precision
        scaler.scale(loss).backward()
        
        # Update weights if we've accumulated enough gradients
        if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(loader):
            # Clip gradients
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            # Update weights with scaled gradients
            scaler.step(optimizer)
            scaler.update()
        
        # Update statistics
        total_loss += loss.item() * args.accumulation_steps
        
        # Calculate accuracy for monitoring (if applicable)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Log to progress bar every log_freq iterations
        if i % args.log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            acc = 100 * correct / max(total, 1)
            pbar.set_postfix({
                'loss': total_loss / (i + 1), 
                'acc': f"{acc:.1f}%", 
                'mask': f"{current_mask_ratio:.2f}", 
                'lr': f"{lr:.1e}"
            })
    
    # Calculate average loss and accuracy over the epoch
    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total if total > 0 else 0
    
    return avg_loss, avg_acc

def validate(model, loader, criterion, args):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Validation"):
            # Move data to device
            points = points.to(args.device)
            labels = labels.to(args.device)
            
            # Forward pass with masking
            with autocast(enabled=args.fp16):
                outputs = model(points, mask_ratio=args.mask_ratio)
                loss = criterion(outputs, labels)
            
            # Update statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Calculate average loss and accuracy over the validation set
    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total if total > 0 else 0
    
    return avg_loss, avg_acc

def save_checkpoint(state, filename):
    """Save training checkpoint"""
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")

def main():
    # Parse arguments
    global args
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Print hardware information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, using CPU. Training will be very slow.")
    
    # Print training configuration
    print(f"\nTraining configuration:")
    print(f"  Model: MaskGIT3D with TCGA (tcga_ratio={args.tcga_ratio})")
    print(f"  Architecture: embed_dim={args.embed_dim}, depth={args.depth}, heads={args.num_heads}")
    print(f"  Dataset: Full ModelNet10")
    print(f"  Mixed precision (FP16): {args.fp16}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Masking strategy: {args.masking_strategy} (ratio={args.mask_ratio})")
    print(f"  Progressive masking: {args.progressive_masking}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Data augmentation: {args.augmentation}")
    print(f"  Class weighting: {args.class_weights}")
    
    # Create model and loss function
    model, facet_loss = create_model(args)
    model = model.to(args.device)
    
    # Compile model if enabled
    if args.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model successfully compiled with torch.compile()")
        except Exception as e:
            print(f"Warning: Failed to compile model: {e}")
            print("Continuing with standard model")
    
    # Prepare dataloaders
    train_loader, val_loader, class_weights = prepare_dataloaders(args)
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Cosine learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return max(args.min_lr / args.lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=args.fp16)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Check if resuming training
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
    
    # Early stopping variables
    early_stopping_counter = 0
    
    # Time tracking
    start_time = time.time()
    
    print(f"\nStarting training for {args.epochs} epochs on full ModelNet10 dataset")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, facet_loss, train_loader, criterion, optimizer, scaler, epoch, args, class_weights)
        
        # Update learning rate
        scheduler.step()
        
        # Validate model if it's validation frequency
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, args)
            
            # Log to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            # Check if this is the best model so far
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                early_stopping_counter = 0
                
                # Save best model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'args': args,
                }, os.path.join(args.output_dir, 'best_model.pth'))
            else:
                early_stopping_counter += 1
                
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.1f}s, Projected remaining: {epoch_time * (args.epochs-epoch-1)/60:.1f}m")
            
            # Early stopping
            if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            # Log train metrics only
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
            
            # Print train results only
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Time: {epoch_time:.1f}s, Projected remaining: {epoch_time * (args.epochs-epoch-1)/60:.1f}m")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'args': args,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    save_checkpoint({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_loss': train_loss,
        'args': args,
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {datetime.timedelta(seconds=int(total_time))}")
    
    return 0

if __name__ == '__main__':
    main()
