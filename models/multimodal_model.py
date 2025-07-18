
"""
Multimodal fusion model for mental health detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

from .text_model import TextModel
from .audio_model import AudioModel

class EarlyFusionModel(nn.Module):
    """Early fusion model that combines features at the feature level."""

    def __init__(self,
                 text_model: nn.Module,
                 audio_model: nn.Module,
                 num_classes: int = 3,
                 dropout_rate: float = 0.2):
        """
        Initialize early fusion model.

        Args:
            text_model: Pre-trained text model
            audio_model: Pre-trained audio model
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(EarlyFusionModel, self).__init__()

        self.text_model = text_model
        self.audio_model = audio_model
        self.num_classes = num_classes

        # Remove final classification layers from individual models
        if hasattr(text_model, 'classifier'):
            self.text_feature_size = text_model.classifier[0].in_features
        else:
            self.text_feature_size = text_model.hidden_size

        if hasattr(audio_model, 'classifier'):
            self.audio_feature_size = audio_model.classifier[0].in_features
        else:
            self.audio_feature_size = audio_model.lstm.output_size

        # Combined feature size
        combined_feature_size = self.text_feature_size + self.audio_feature_size

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_feature_size, combined_feature_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_feature_size // 2, combined_feature_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_feature_size // 4, num_classes)
        )

        logging.info(f"Initialized EarlyFusionModel with text_size={self.text_feature_size}, audio_size={self.audio_feature_size}")

    def forward(self, text_input: Dict[str, torch.Tensor], audio_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through early fusion model.

        Args:
            text_input: Dictionary with 'input_ids' and 'attention_mask'
            audio_input: Audio tensor

        Returns:
            Classification logits
        """
        # Extract features from individual models
        text_features = self.text_model.get_features(
            text_input['input_ids'], 
            text_input['attention_mask']
        )
        audio_features = self.audio_model.get_features(audio_input)

        # Concatenate features
        combined_features = torch.cat([text_features, audio_features], dim=1)

        # Fusion classification
        logits = self.fusion_layers(combined_features)

        return logits

class LateFusionModel(nn.Module):
    """Late fusion model that combines predictions at the decision level."""

    def __init__(self,
                 text_model: nn.Module,
                 audio_model: nn.Module,
                 num_classes: int = 3,
                 fusion_method: str = 'weighted_average',
                 weights: Optional[List[float]] = None):
        """
        Initialize late fusion model.

        Args:
            text_model: Pre-trained text model
            audio_model: Pre-trained audio model
            num_classes: Number of output classes
            fusion_method: Method for combining predictions ('average', 'weighted_average', 'learned')
            weights: Weights for weighted average (if applicable)
        """
        super(LateFusionModel, self).__init__()

        self.text_model = text_model
        self.audio_model = audio_model
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        if fusion_method == 'weighted_average':
            if weights is None:
                self.weights = nn.Parameter(torch.ones(2) / 2)
            else:
                self.weights = nn.Parameter(torch.tensor(weights))
        elif fusion_method == 'learned':
            self.fusion_layer = nn.Linear(2 * num_classes, num_classes)

        logging.info(f"Initialized LateFusionModel with fusion_method={fusion_method}")

    def forward(self, text_input: Dict[str, torch.Tensor], audio_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through late fusion model.

        Args:
            text_input: Dictionary with 'input_ids' and 'attention_mask'
            audio_input: Audio tensor

        Returns:
            Classification logits
        """
        # Get predictions from individual models
        text_logits = self.text_model(text_input['input_ids'], text_input['attention_mask'])
        audio_logits = self.audio_model(audio_input)

        # Fusion
        if self.fusion_method == 'average':
            logits = (text_logits + audio_logits) / 2
        elif self.fusion_method == 'weighted_average':
            weights = F.softmax(self.weights, dim=0)
            logits = weights[0] * text_logits + weights[1] * audio_logits
        elif self.fusion_method == 'learned':
            combined_logits = torch.cat([text_logits, audio_logits], dim=1)
            logits = self.fusion_layer(combined_logits)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return logits

class AttentionFusionModel(nn.Module):
    """Attention-based fusion model."""

    def __init__(self,
                 text_model: nn.Module,
                 audio_model: nn.Module,
                 num_classes: int = 3,
                 hidden_size: int = 256,
                 dropout_rate: float = 0.2):
        """
        Initialize attention fusion model.

        Args:
            text_model: Pre-trained text model
            audio_model: Pre-trained audio model
            num_classes: Number of output classes
            hidden_size: Hidden size for attention mechanism
            dropout_rate: Dropout rate
        """
        super(AttentionFusionModel, self).__init__()

        self.text_model = text_model
        self.audio_model = audio_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Get feature sizes
        if hasattr(text_model, 'classifier'):
            self.text_feature_size = text_model.classifier[0].in_features
        else:
            self.text_feature_size = text_model.hidden_size

        if hasattr(audio_model, 'classifier'):
            self.audio_feature_size = audio_model.classifier[0].in_features
        else:
            self.audio_feature_size = audio_model.lstm.output_size

        # Projection layers
        self.text_projection = nn.Linear(self.text_feature_size, hidden_size)
        self.audio_projection = nn.Linear(self.audio_feature_size, hidden_size)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

        logging.info(f"Initialized AttentionFusionModel with hidden_size={hidden_size}")

    def forward(self, text_input: Dict[str, torch.Tensor], audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention fusion model.

        Args:
            text_input: Dictionary with 'input_ids' and 'attention_mask'
            audio_input: Audio tensor

        Returns:
            Tuple of (logits, attention_weights)
        """
        # Extract features
        text_features = self.text_model.get_features(
            text_input['input_ids'], 
            text_input['attention_mask']
        )
        audio_features = self.audio_model.get_features(audio_input)

        # Project features to common space
        text_projected = self.text_projection(text_features)
        audio_projected = self.audio_projection(audio_features)

        # Stack features for attention
        features = torch.stack([text_projected, audio_projected], dim=1)  # (batch_size, 2, hidden_size)

        # Calculate attention weights
        attention_weights = self.attention(features)  # (batch_size, 2, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, 2)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention
        weighted_features = torch.bmm(attention_weights.unsqueeze(1), features)
        weighted_features = weighted_features.squeeze(1)  # (batch_size, hidden_size)

        # Classification
        logits = self.classifier(weighted_features)

        return logits, attention_weights

class CrossModalAttentionModel(nn.Module):
    """Cross-modal attention model for feature interaction."""

    def __init__(self,
                 text_model: nn.Module,
                 audio_model: nn.Module,
                 num_classes: int = 3,
                 hidden_size: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.2):
        """
        Initialize cross-modal attention model.

        Args:
            text_model: Pre-trained text model
            audio_model: Pre-trained audio model
            num_classes: Number of output classes
            hidden_size: Hidden size for attention
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(CrossModalAttentionModel, self).__init__()

        self.text_model = text_model
        self.audio_model = audio_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Get feature sizes
        if hasattr(text_model, 'classifier'):
            self.text_feature_size = text_model.classifier[0].in_features
        else:
            self.text_feature_size = text_model.hidden_size

        if hasattr(audio_model, 'classifier'):
            self.audio_feature_size = audio_model.classifier[0].in_features
        else:
            self.audio_feature_size = audio_model.lstm.output_size

        # Projection layers
        self.text_projection = nn.Linear(self.text_feature_size, hidden_size)
        self.audio_projection = nn.Linear(self.audio_feature_size, hidden_size)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

        logging.info(f"Initialized CrossModalAttentionModel with hidden_size={hidden_size}, num_heads={num_heads}")

    def forward(self, text_input: Dict[str, torch.Tensor], audio_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cross-modal attention model.

        Args:
            text_input: Dictionary with 'input_ids' and 'attention_mask'
            audio_input: Audio tensor

        Returns:
            Classification logits
        """
        # Extract features
        text_features = self.text_model.get_features(
            text_input['input_ids'], 
            text_input['attention_mask']
        )
        audio_features = self.audio_model.get_features(audio_input)

        # Project features
        text_projected = self.text_projection(text_features).unsqueeze(1)  # (batch_size, 1, hidden_size)
        audio_projected = self.audio_projection(audio_features).unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Cross-modal attention: text attends to audio
        text_attended, _ = self.cross_attention(text_projected, audio_projected, audio_projected)
        text_attended = self.layer_norm(text_attended + text_projected)

        # Cross-modal attention: audio attends to text
        audio_attended, _ = self.cross_attention(audio_projected, text_projected, text_projected)
        audio_attended = self.layer_norm(audio_attended + audio_projected)

        # Combine attended features
        combined_features = torch.cat([
            text_attended.squeeze(1), 
            audio_attended.squeeze(1)
        ], dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits

def create_multimodal_model(text_model: nn.Module, 
                          audio_model: nn.Module,
                          fusion_config: Dict) -> nn.Module:
    """
    Create multimodal model based on configuration.

    Args:
        text_model: Pre-trained text model
        audio_model: Pre-trained audio model
        fusion_config: Fusion configuration

    Returns:
        Multimodal model
    """
    fusion_type = fusion_config.get('fusion_type', 'late')

    if fusion_type == 'early':
        return EarlyFusionModel(
            text_model=text_model,
            audio_model=audio_model,
            num_classes=fusion_config.get('num_classes', 3),
            dropout_rate=fusion_config.get('dropout_rate', 0.2)
        )
    elif fusion_type == 'late':
        return LateFusionModel(
            text_model=text_model,
            audio_model=audio_model,
            num_classes=fusion_config.get('num_classes', 3),
            fusion_method=fusion_config.get('fusion_method', 'weighted_average'),
            weights=fusion_config.get('weights', None)
        )
    elif fusion_type == 'attention':
        return AttentionFusionModel(
            text_model=text_model,
            audio_model=audio_model,
            num_classes=fusion_config.get('num_classes', 3),
            hidden_size=fusion_config.get('hidden_size', 256),
            dropout_rate=fusion_config.get('dropout_rate', 0.2)
        )
    elif fusion_type == 'cross_attention':
        return CrossModalAttentionModel(
            text_model=text_model,
            audio_model=audio_model,
            num_classes=fusion_config.get('num_classes', 3),
            hidden_size=fusion_config.get('hidden_size', 256),
            num_heads=fusion_config.get('num_heads', 8),
            dropout_rate=fusion_config.get('dropout_rate', 0.2)
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

# Example usage
if __name__ == "__main__":
    # Create individual models
    text_model = TextModel()
    audio_model = AudioModel()

    # Create multimodal model
    fusion_config = {
        'fusion_type': 'late',
        'num_classes': 3,
        'fusion_method': 'weighted_average'
    }

    multimodal_model = create_multimodal_model(text_model, audio_model, fusion_config)

    # Test forward pass
    batch_size = 2
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    audio_input = torch.randn(batch_size, 1, 128, 1000)

    logits = multimodal_model(text_input, audio_input)
    print(f"Multimodal output shape: {logits.shape}")
