
"""
Audio model for mental health detection using CNN and LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

class AudioCNN(nn.Module):
    """CNN for audio feature extraction."""

    def __init__(self, 
                 input_channels: int = 1,
                 cnn_filters: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 dropout_rate: float = 0.3):
        """
        Initialize CNN for audio processing.

        Args:
            input_channels: Number of input channels
            cnn_filters: List of filter sizes for each layer
            kernel_size: Kernel size for convolution
            dropout_rate: Dropout rate
        """
        super(AudioCNN, self).__init__()

        self.input_channels = input_channels
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # CNN layers
        layers = []
        in_channels = input_channels

        for i, out_channels in enumerate(cnn_filters):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout_rate)
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Calculate output size (this depends on input size and pooling)
        self.output_channels = cnn_filters[-1]

        logging.info(f"Initialized AudioCNN with filters={cnn_filters}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            CNN features
        """
        return self.cnn(x)

class AudioLSTM(nn.Module):
    """LSTM for audio sequence modeling."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize LSTM for audio processing.

        Args:
            input_size: Size of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(AudioLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_size = hidden_size * (2 if bidirectional else 1)

        logging.info(f"Initialized AudioLSTM with hidden_size={hidden_size}, layers={num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            LSTM output
        """
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        return hidden

class AudioModel(nn.Module):
    """Complete audio model for mental health detection."""

    def __init__(self,
                 input_shape: Tuple[int, int] = (128, 1000),  # (n_mels, n_frames)
                 num_classes: int = 3,
                 cnn_filters: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 lstm_hidden_size: int = 128,
                 lstm_layers: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize complete audio model.

        Args:
            input_shape: Shape of input spectrograms
            num_classes: Number of output classes
            cnn_filters: List of CNN filter sizes
            kernel_size: CNN kernel size
            lstm_hidden_size: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(AudioModel, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        # CNN for feature extraction
        self.cnn = AudioCNN(
            input_channels=1,
            cnn_filters=cnn_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )

        # Calculate CNN output size
        # This is a simplified calculation - in practice, you'd need to compute this exactly
        cnn_output_height = input_shape[0] // (2 ** len(cnn_filters))
        cnn_output_width = input_shape[1] // (2 ** len(cnn_filters))
        cnn_output_channels = cnn_filters[-1]

        # Reshape for LSTM
        lstm_input_size = cnn_output_height * cnn_output_channels

        # LSTM for sequence modeling
        self.lstm = AudioLSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout_rate=dropout_rate,
            bidirectional=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.lstm.output_size, self.lstm.output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.lstm.output_size // 2, num_classes)
        )

        logging.info(f"Initialized AudioModel with input_shape={input_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete audio model.

        Args:
            x: Input tensor (batch_size, 1, height, width)

        Returns:
            Classification logits
        """
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch_size, channels, height, width)

        # Reshape for LSTM
        # Convert (batch_size, channels, height, width) to (batch_size, seq_len, features)
        batch_size, channels, height, width = cnn_features.size()

        # Transpose to make width the sequence dimension
        cnn_features = cnn_features.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        cnn_features = cnn_features.contiguous().view(batch_size, width, channels * height)

        # LSTM processing
        lstm_features = self.lstm(cnn_features)

        # Classification
        logits = self.classifier(lstm_features)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature representations.

        Args:
            x: Input tensor

        Returns:
            Feature representations
        """
        with torch.no_grad():
            batch_size = x.size(0)

            # CNN features
            cnn_features = self.cnn(x)

            # Reshape for LSTM
            batch_size, channels, height, width = cnn_features.size()
            cnn_features = cnn_features.permute(0, 3, 1, 2)
            cnn_features = cnn_features.contiguous().view(batch_size, width, channels * height)

            # LSTM features
            lstm_features = self.lstm(cnn_features)

        return lstm_features

class SimpleAudioModel(nn.Module):
    """Simplified audio model for MFCC features."""

    def __init__(self,
                 input_size: int = 13,  # MFCC features
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout_rate: float = 0.3):
        """
        Initialize simple audio model for MFCC features.

        Args:
            input_size: Size of input features (e.g., 13 for MFCC)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(SimpleAudioModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        lstm_output_size = hidden_size * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MFCC features.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            Classification logits
        """
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        logits = self.classifier(hidden)

        return logits

def create_audio_model(model_config: Dict) -> nn.Module:
    """
    Create audio model based on configuration.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Audio model
    """
    model_type = model_config.get('model_type', 'cnn_lstm')

    if model_type == 'cnn_lstm':
        return AudioModel(
            input_shape=model_config.get('input_shape', (128, 1000)),
            num_classes=model_config.get('num_classes', 3),
            cnn_filters=model_config.get('cnn_filters', [64, 128, 256]),
            kernel_size=model_config.get('kernel_size', 3),
            lstm_hidden_size=model_config.get('lstm_hidden_size', 128),
            lstm_layers=model_config.get('lstm_layers', 2),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
    elif model_type == 'simple':
        return SimpleAudioModel(
            input_size=model_config.get('input_size', 13),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            num_classes=model_config.get('num_classes', 3),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example usage
if __name__ == "__main__":
    # Test the model
    model = AudioModel()

    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 128, 1000)

    # Forward pass
    logits = model(input_tensor)
    print(f"Output shape: {logits.shape}")
    print(f"Logits: {logits}")
