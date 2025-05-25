"""FACET (Fairness-Aware Contrastive Estimation) Loss for reducing gender bias"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FACETLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_fair=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_fair = lambda_fair
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, features, labels, protected_attributes):
        """
        Args:
            features: [batch_size, feature_dim] - Model embeddings
            labels: [batch_size] - Class labels
            protected_attributes: [batch_size] - Binary protected attributes (e.g., gender)
        """
        device = features.device
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute logits for contrastive loss
        logits = torch.matmul(features, features.T) / self.temperature
        
        # Standard contrastive loss
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        labels = torch.arange(batch_size, device=device).masked_fill(mask, -1)
        
        # Positive pairs (same class, different protected attribute)
        class_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
        protected_mask = (protected_attributes.unsqueeze(1) != protected_attributes.unsqueeze(0))
        positive_mask = class_mask & protected_mask
        
        # Fairness-aware contrastive loss
        logits = logits.masked_fill(mask, -9e15)  # Mask self-contrast
        log_prob = F.log_softmax(logits, dim=-1)
        
        # Compute fairness loss (encourage similar representations across protected groups)
        fair_loss = -torch.sum(log_prob * positive_mask.float(), dim=1).mean()
        
        # Standard cross-entropy loss
        ce_loss = self.cross_entropy(logits, labels)
        
        # Combined loss
        total_loss = ce_loss + self.lambda_fair * fair_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'fair_loss': fair_loss
        }
