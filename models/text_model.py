
"""
Text model for mental health detection using BERT/DistilBERT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging

class TextModel(nn.Module):
    """Text model for mental health detection."""

    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_classes: int = 3,
                 dropout_rate: float = 0.1,
                 freeze_bert: bool = False):
        """
        Initialize text model.

        Args:
            model_name: Name of the transformer model
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
        """
        super(TextModel, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Get hidden size
        self.hidden_size = self.bert.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # Additional layers for better performance
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.intermediate = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.classifier_final = nn.Linear(self.hidden_size // 2, num_classes)

        logging.info(f"Initialized TextModel with {model_name}, hidden_size={self.hidden_size}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Logits for classification
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use pooled output (CLS token)
        pooled_output = outputs.pooler_output

        # Apply layer normalization
        pooled_output = self.layer_norm(pooled_output)

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Intermediate layer
        intermediate_output = F.relu(self.intermediate(pooled_output))
        intermediate_output = self.dropout(intermediate_output)

        # Final classification
        logits = self.classifier_final(intermediate_output)

        return logits

    def get_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get feature representations.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Feature representations
        """
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.layer_norm(pooled_output)

        return pooled_output

class TextModelWithFeatures(nn.Module):
    """Text model that combines BERT features with handcrafted features."""

    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 num_classes: int = 3,
                 num_features: int = 10,
                 dropout_rate: float = 0.1):
        """
        Initialize text model with features.

        Args:
            model_name: Name of the transformer model
            num_classes: Number of output classes
            num_features: Number of additional features
            dropout_rate: Dropout rate
        """
        super(TextModelWithFeatures, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32)
        )

        # Combined processing
        combined_size = self.hidden_size + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_size // 2, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with additional features.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            features: Additional features

        Returns:
            Logits for classification
        """
        # Get BERT features
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_features = bert_outputs.pooler_output

        # Process additional features
        processed_features = self.feature_processor(features)

        # Combine features
        combined_features = torch.cat([bert_features, processed_features], dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits

class AttentionTextModel(nn.Module):
    """Text model with attention mechanism for better interpretability."""

    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 num_classes: int = 3,
                 dropout_rate: float = 0.1):
        """
        Initialize attention-based text model.

        Args:
            model_name: Name of the transformer model
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(AttentionTextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weights.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (logits, attention_weights)
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Calculate attention weights
        attention_weights = self.attention(sequence_output)  # (batch_size, seq_len, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)

        # Apply attention mask
        attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention to get weighted representation
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), sequence_output)
        weighted_output = weighted_output.squeeze(1)  # (batch_size, hidden_size)

        # Classification
        logits = self.classifier(weighted_output)

        return logits, attention_weights

def create_text_model(model_config: Dict) -> nn.Module:
    """
    Create text model based on configuration.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Text model
    """
    model_type = model_config.get('model_type', 'basic')

    if model_type == 'basic':
        return TextModel(
            model_name=model_config.get('model_name', 'distilbert-base-uncased'),
            num_classes=model_config.get('num_classes', 3),
            dropout_rate=model_config.get('dropout_rate', 0.1)
        )
    elif model_type == 'with_features':
        return TextModelWithFeatures(
            model_name=model_config.get('model_name', 'distilbert-base-uncased'),
            num_classes=model_config.get('num_classes', 3),
            num_features=model_config.get('num_features', 10),
            dropout_rate=model_config.get('dropout_rate', 0.1)
        )
    elif model_type == 'attention':
        return AttentionTextModel(
            model_name=model_config.get('model_name', 'distilbert-base-uncased'),
            num_classes=model_config.get('num_classes', 3),
            dropout_rate=model_config.get('dropout_rate', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example usage
if __name__ == "__main__":
    # Test the model
    model = TextModel()

    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Forward pass
    logits = model(input_ids, attention_mask)
    print(f"Output shape: {logits.shape}")
    print(f"Logits: {logits}")
