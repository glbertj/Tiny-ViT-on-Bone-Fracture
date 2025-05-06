import torch
import torch.nn as nn
import timm
from config import MODEL_NAME, DEVICE


class FractureClassifier(nn.Module):
    """
    Vision Transformer based classification model with custom head
    """
    def __init__(self, dropout_rate=0.2):
        super(FractureClassifier, self).__init__()
        
        # Load pre-trained model
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0  # Remove classifier head
        )
        
        # Get the feature dimension
        feature_dim = self.backbone.num_features
        
        # Create a custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def get_model(pretrained=True, dropout_rate=0.2):
    """
    Initialize the model
    
    Args:
        pretrained: Whether to load pre-trained weights
        dropout_rate: Dropout rate for classifier
        
    Returns:
        Initialized model
    """
    model = FractureClassifier(
        dropout_rate=dropout_rate
    )
    
    return model.to(DEVICE)


def get_loss_fn(pos_weight=None):
    """
    Get loss function with optional class weighting
    
    Args:
        pos_weight: Optional weight for positive class
        
    Returns:
        Loss function
    """
    if pos_weight is not None:
        # Create tensor with the positive class weight
        pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        return nn.BCEWithLogitsLoss()