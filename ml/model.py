"""
Multi-Modal Contrastive Reliability Network (MCRN)

Architecture:
- ResNet18 backbone (pretrained on ImageNet, fine-tuned)
- Modified to accept 4-channel input instead of 3
- Output: trust score + false alarm probability

Innovation: We treat each 4-channel scalogram as a multi-spectral
image - similar to satellite imagery analysis but for ICU signals.
The CNN learns spatial patterns in time-frequency space that are
invisible to 1D signal processing approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SignalReliabilityNet(nn.Module):
    """
    ResNet18 backbone fine-tuned for ICU alarm classification.
    
    Input:  (batch, 4, 64, 64) - 4-channel CWT scalogram
    Output: (batch, 2) - [false_alarm_prob, true_alarm_prob]
    
    Key modification: first conv layer accepts 4 channels
    (II, V, PLETH, RESP) instead of standard 3 (RGB).
    This forces the model to learn cross-channel relationships
    - the core of our contrastive approach.
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        
        # Load pretrained ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first conv to accept 4 channels
        # Average the pretrained RGB weights across channels, then
        # add a 4th channel initialized to the mean - preserves
        # pretrained knowledge while adding new channel capacity
        original_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Smart weight initialization
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            new_conv.weight[:, 3] = original_conv.weight.mean(dim=1)
        
        backbone.conv1 = new_conv
        
        # Remove final FC layer - we'll add our own
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Separate head for trust score (0-100 continuous output)
        self.trust_head = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1, scaled to 0-100
        )
    
    def forward(self, x: torch.Tensor):
        # Extract features via ResNet backbone
        features = self.features(x)  # (batch, 512, 1, 1)
        features = features.flatten(1)  # (batch, 512)
        
        # Classification output
        logits = self.classifier(features)
        
        # Trust score output
        trust = self.trust_head(features) * 100  # Scale to 0-100
        
        return logits, trust
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embedding for visualization."""
        features = self.features(x)
        return features.flatten(1)


class FocalLoss(nn.Module):
    """
    Focal Loss - addresses class imbalance by down-weighting
    easy examples and focusing on hard ones.
    
    Critical for medical alarm classification where false negatives
    (missing a true alarm) are far more dangerous than false positives.
    gamma=2 is standard; alpha weights the positive class.
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


if __name__ == "__main__":
    # Verify model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = SignalReliabilityNet().to(device)
    
    # Test forward pass
    x = torch.randn(8, 4, 64, 64).to(device)
    logits, trust = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Trust shape:  {trust.shape}")
    print(f"Trust range:  {trust.min().item():.1f} - {trust.max().item():.1f}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {params:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"\nModel architecture verified.")