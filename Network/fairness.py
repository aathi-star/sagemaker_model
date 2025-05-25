# Fairness-aware module for Halton-MaskGIT
# This module implements bias mitigation via perceptual loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class FACETLoss(nn.Module):
    """
    Fairness-Aware Conditional gEnerative Transformer (FACET) loss.
    This loss helps mitigate biases in generated images by penalizing biased outputs.
    
    Based on the paper: https://arxiv.org/abs/2301.07567
    """
    def __init__(self, lambda_fairness=0.3):
        super().__init__()
        self.lambda_fairness = lambda_fairness
        
        # Simulated fairness classifiers
        # In a real implementation, these would be pre-trained models
        self.gender_classifier = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 2)  # Binary classification: male/female
        )
        
        self.racial_classifier = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 5)  # 5 racial categories
        )
        
        # Initialize with pre-trained weights (simulated here)
        self._init_weights()
        
        # Freeze classifier weights - they should not be trained
        for param in self.gender_classifier.parameters():
            param.requires_grad = False
        for param in self.racial_classifier.parameters():
            param.requires_grad = False
    
    def _init_weights(self):
        """Initialize the classifiers with simulated pre-trained weights."""
        # In a real implementation, you would load actual pre-trained weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def compute_gender_bias(self, images):
        """
        Compute gender bias in generated images.
        
        Args:
            images: Batch of generated images [B, 3, H, W]
            
        Returns:
            Gender bias score
        """
        logits = self.gender_classifier(images)
        probs = F.softmax(logits, dim=1)
        
        # Calculate bias as deviation from uniform distribution
        uniform = torch.ones_like(probs) / probs.size(1)
        bias = F.kl_div(probs.log(), uniform, reduction='batchmean')
        return bias
    
    def compute_racial_bias(self, images):
        """
        Compute racial bias in generated images.
        
        Args:
            images: Batch of generated images [B, 3, H, W]
            
        Returns:
            Racial bias score
        """
        logits = self.racial_classifier(images)
        probs = F.softmax(logits, dim=1)
        
        # Calculate bias as deviation from uniform distribution
        uniform = torch.ones_like(probs) / probs.size(1)
        bias = F.kl_div(probs.log(), uniform, reduction='batchmean')
        return bias
    
    def forward(self, images, fid_loss):
        """
        Compute the fairness-aware loss.
        
        Args:
            images: Batch of generated images [B, 3, H, W]
            fid_loss: FID loss value
            
        Returns:
            Combined loss with fairness penalty
        """
        gender_bias = self.compute_gender_bias(images)
        racial_bias = self.compute_racial_bias(images)
        
        # Combine biases
        fairness_loss = gender_bias + racial_bias
        
        # Combine with FID loss
        total_loss = fid_loss + self.lambda_fairness * fairness_loss
        
        return total_loss, {
            'fid_loss': fid_loss.item(),
            'gender_bias': gender_bias.item(),
            'racial_bias': racial_bias.item(),
            'fairness_loss': fairness_loss.item(),
            'total_loss': total_loss.item()
        }


class BiasAnalyzer:
    """
    Tool for analyzing and visualizing bias in generated images.
    """
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load pre-trained FairFace model (simulated)
        self.fairface = self._load_fairface()
    
    def _load_fairface(self):
        """Load a simulated FairFace model for bias analysis."""
        # In a real implementation, you would load the actual FairFace model
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 18)  # 9 age groups, 7 race groups, 2 gender groups
        ).to(self.device)
        
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def analyze_batch(self, images):
        """
        Analyze a batch of images for demographic distribution.
        
        Args:
            images: Batch of images [B, 3, H, W]
            
        Returns:
            Dictionary with demographic statistics
        """
        with torch.no_grad():
            features = self.fairface(images.to(self.device))
            
            # Split features into age, race, gender
            age_logits = features[:, :9]
            race_logits = features[:, 9:16]
            gender_logits = features[:, 16:]
            
            # Convert to probabilities
            age_probs = F.softmax(age_logits, dim=1)
            race_probs = F.softmax(race_logits, dim=1)
            gender_probs = F.softmax(gender_logits, dim=1)
            
            # Calculate distributions
            age_dist = age_probs.mean(dim=0).cpu().numpy()
            race_dist = race_probs.mean(dim=0).cpu().numpy()
            gender_dist = gender_probs.mean(dim=0).cpu().numpy()
            
            return {
                'age_distribution': age_dist,
                'race_distribution': race_dist,
                'gender_distribution': gender_dist,
                'gender_bias_score': np.abs(gender_dist[0] - 0.5) * 2,  # 0 is balanced, 1 is max bias
                'race_bias_score': self._calculate_race_bias(race_dist)
            }
    
    def _calculate_race_bias(self, race_dist):
        """Calculate racial bias as deviation from uniform distribution."""
        uniform = np.ones_like(race_dist) / len(race_dist)
        return np.sum(np.abs(race_dist - uniform)) / 2  # Normalized to [0, 1]
