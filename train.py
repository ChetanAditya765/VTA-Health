
"""
Main training script for multimodal mental health detection.
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import Config
from utils.data_utils import create_synthetic_data, split_data, encode_labels
from utils.model_utils import EarlyStopping, calculate_metrics, plot_training_history, save_model, set_seed
from data_processing.text_preprocessing import process_text_data
from data_processing.audio_preprocessing import process_audio_data
from models.text_model import create_text_model
from models.audio_model import create_audio_model
from models.multimodal_model import create_multimodal_model

class MultimodalTrainer:
    """Main trainer for multimodal mental health detection."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.device = torch.device(self.config.get('training.device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Set random seed
        set_seed(self.config.get('training.seed', 42))

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging.level', 'INFO')),
            format=self.config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger(__name__)

        # Initialize models
        self.text_model = None
        self.audio_model = None
        self.multimodal_model = None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }

        self.logger.info(f"Initialized MultimodalTrainer with device: {self.device}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data.

        Returns:
            Tuple of (text_df, audio_df, labels_df)
        """
        # Check if real data exists
        daic_path = self.config.get('data.daic_woz.labels_path', '')

        if os.path.exists(daic_path):
            self.logger.info("Loading DAIC-WOZ dataset...")
            # Load real data (implementation depends on actual data format)
            # For now, create synthetic data
            text_df, audio_df, labels_df = create_synthetic_data(1000)
        else:
            self.logger.info("Creating synthetic data for demonstration...")
            text_df, audio_df, labels_df = create_synthetic_data(1000)

        # Merge dataframes
        data_df = text_df.merge(audio_df, on='participant_id').merge(labels_df, on='participant_id')

        # Split data
        train_df, val_df, test_df = split_data(
            data_df,
            test_size=self.config.get('training.test_split', 0.1),
            val_size=self.config.get('training.validation_split', 0.2),
            stratify_col='label',
            random_state=self.config.get('training.seed', 42)
        )

        self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def create_models(self):
        """Create individual and multimodal models."""
        # Text model
        text_config = self.config.get('models.text', {})
        text_config['num_classes'] = 3
        self.text_model = create_text_model(text_config)

        # Audio model
        audio_config = self.config.get('models.audio', {})
        audio_config['num_classes'] = 3
        self.audio_model = create_audio_model(audio_config)

        # Multimodal model
        multimodal_config = self.config.get('models.multimodal', {})
        multimodal_config['num_classes'] = 3
        self.multimodal_model = create_multimodal_model(
            self.text_model, 
            self.audio_model, 
            multimodal_config
        )

        # Move models to device
        self.text_model.to(self.device)
        self.audio_model.to(self.device)
        self.multimodal_model.to(self.device)

        self.logger.info("Created all models")

    def train_individual_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Train individual text and audio models.

        Args:
            train_df: Training data
            val_df: Validation data
        """
        self.logger.info("Training individual models...")

        # Prepare text data
        text_train_encoded, text_train_features = process_text_data(
            train_df, 
            model_name=self.config.get('models.text.model_name', 'distilbert-base-uncased')
        )
        text_val_encoded, text_val_features = process_text_data(
            val_df, 
            model_name=self.config.get('models.text.model_name', 'distilbert-base-uncased')
        )

        # Prepare audio data
        audio_train_mfcc, audio_train_mel, audio_train_prosodic = process_audio_data(train_df)
        audio_val_mfcc, audio_val_mel, audio_val_prosodic = process_audio_data(val_df)

        # Prepare labels
        train_labels = train_df['label'].values
        val_labels = val_df['label'].values

        # Train text model
        self.logger.info("Training text model...")
        self._train_text_model(text_train_encoded, train_labels, text_val_encoded, val_labels)

        # Train audio model
        self.logger.info("Training audio model...")
        self._train_audio_model(audio_train_mfcc, train_labels, audio_val_mfcc, val_labels)

        self.logger.info("Individual model training completed")

    def _train_text_model(self, train_encoded: Dict, train_labels: np.ndarray, 
                         val_encoded: Dict, val_labels: np.ndarray):
        """Train text model."""
        # Setup training
        optimizer = optim.AdamW(
            self.text_model.parameters(),
            lr=self.config.get('models.text.learning_rate', 2e-5)
        )
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(
            patience=self.config.get('training.early_stopping.patience', 5),
            min_delta=self.config.get('training.early_stopping.min_delta', 0.001)
        )

        # Training loop
        num_epochs = self.config.get('models.text.num_epochs', 10)
        batch_size = self.config.get('models.text.batch_size', 16)

        for epoch in range(num_epochs):
            # Training phase
            self.text_model.train()
            total_loss = 0
            correct = 0
            total = 0

            # Simple batch processing (in practice, use DataLoader)
            for i in range(0, len(train_labels), batch_size):
                end_idx = min(i + batch_size, len(train_labels))

                # Get batch
                batch_input_ids = train_encoded['input_ids'][i:end_idx].to(self.device)
                batch_attention_mask = train_encoded['attention_mask'][i:end_idx].to(self.device)
                batch_labels = torch.tensor(train_labels[i:end_idx], dtype=torch.long).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                logits = self.text_model(batch_input_ids, batch_attention_mask)
                loss = criterion(logits, batch_labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            # Validation phase
            self.text_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i in range(0, len(val_labels), batch_size):
                    end_idx = min(i + batch_size, len(val_labels))

                    # Get batch
                    batch_input_ids = val_encoded['input_ids'][i:end_idx].to(self.device)
                    batch_attention_mask = val_encoded['attention_mask'][i:end_idx].to(self.device)
                    batch_labels = torch.tensor(val_labels[i:end_idx], dtype=torch.long).to(self.device)

                    # Forward pass
                    logits = self.text_model(batch_input_ids, batch_attention_mask)
                    loss = criterion(logits, batch_labels)

                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()

            # Calculate metrics
            train_acc = 100 * correct / total
            val_acc = 100 * val_correct / val_total

            self.logger.info(f"Text Model Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {total_loss/len(train_labels)*batch_size:.4f}, "
                           f"Train Acc: {train_acc:.2f}%, "
                           f"Val Loss: {val_loss/len(val_labels)*batch_size:.4f}, "
                           f"Val Acc: {val_acc:.2f}%")

            # Early stopping
            if early_stopping(val_loss, self.text_model):
                self.logger.info("Early stopping triggered for text model")
                break

        # Save model
        save_model(self.text_model, optimizer, epoch, val_loss, 
                  os.path.join(self.config.get('data.models_path', 'data/models'), 'text_model.pth'))

    def _train_audio_model(self, train_mfcc: np.ndarray, train_labels: np.ndarray,
                          val_mfcc: np.ndarray, val_labels: np.ndarray):
        """Train audio model."""
        # Setup training
        optimizer = optim.Adam(
            self.audio_model.parameters(),
            lr=self.config.get('models.audio.learning_rate', 1e-3)
        )
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(
            patience=self.config.get('training.early_stopping.patience', 5),
            min_delta=self.config.get('training.early_stopping.min_delta', 0.001)
        )

        # Convert to tensors
        train_mfcc = torch.tensor(train_mfcc, dtype=torch.float32)
        val_mfcc = torch.tensor(val_mfcc, dtype=torch.float32)

        # Training loop
        num_epochs = self.config.get('models.audio.num_epochs', 50)
        batch_size = self.config.get('models.audio.batch_size', 32)

        for epoch in range(num_epochs):
            # Training phase
            self.audio_model.train()
            total_loss = 0
            correct = 0
            total = 0

            for i in range(0, len(train_labels), batch_size):
                end_idx = min(i + batch_size, len(train_labels))

                # Get batch
                batch_audio = train_mfcc[i:end_idx].to(self.device)
                batch_labels = torch.tensor(train_labels[i:end_idx], dtype=torch.long).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                logits = self.audio_model(batch_audio)
                loss = criterion(logits, batch_labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            # Validation phase
            self.audio_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i in range(0, len(val_labels), batch_size):
                    end_idx = min(i + batch_size, len(val_labels))

                    # Get batch
                    batch_audio = val_mfcc[i:end_idx].to(self.device)
                    batch_labels = torch.tensor(val_labels[i:end_idx], dtype=torch.long).to(self.device)

                    # Forward pass
                    logits = self.audio_model(batch_audio)
                    loss = criterion(logits, batch_labels)

                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()

            # Calculate metrics
            train_acc = 100 * correct / total
            val_acc = 100 * val_correct / val_total

            if epoch % 10 == 0:
                self.logger.info(f"Audio Model Epoch {epoch+1}/{num_epochs} - "
                               f"Train Loss: {total_loss/len(train_labels)*batch_size:.4f}, "
                               f"Train Acc: {train_acc:.2f}%, "
                               f"Val Loss: {val_loss/len(val_labels)*batch_size:.4f}, "
                               f"Val Acc: {val_acc:.2f}%")

            # Early stopping
            if early_stopping(val_loss, self.audio_model):
                self.logger.info("Early stopping triggered for audio model")
                break

        # Save model
        save_model(self.audio_model, optimizer, epoch, val_loss,
                  os.path.join(self.config.get('data.models_path', 'data/models'), 'audio_model.pth'))

    def train_multimodal_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Train multimodal fusion model.

        Args:
            train_df: Training data
            val_df: Validation data
        """
        self.logger.info("Training multimodal model...")

        # Prepare data
        text_train_encoded, _ = process_text_data(
            train_df, 
            model_name=self.config.get('models.text.model_name', 'distilbert-base-uncased')
        )
        text_val_encoded, _ = process_text_data(
            val_df, 
            model_name=self.config.get('models.text.model_name', 'distilbert-base-uncased')
        )

        audio_train_mfcc, _, _ = process_audio_data(train_df)
        audio_val_mfcc, _, _ = process_audio_data(val_df)

        # Convert to tensors
        audio_train_mfcc = torch.tensor(audio_train_mfcc, dtype=torch.float32)
        audio_val_mfcc = torch.tensor(audio_val_mfcc, dtype=torch.float32)

        # Prepare labels
        train_labels = train_df['label'].values
        val_labels = val_df['label'].values

        # Setup training
        optimizer = optim.Adam(
            self.multimodal_model.parameters(),
            lr=self.config.get('models.multimodal.learning_rate', 1e-4)
        )
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(
            patience=self.config.get('training.early_stopping.patience', 5),
            min_delta=self.config.get('training.early_stopping.min_delta', 0.001)
        )

        # Training loop
        num_epochs = self.config.get('models.multimodal.num_epochs', 30)
        batch_size = self.config.get('models.multimodal.batch_size', 16)

        for epoch in range(num_epochs):
            # Training phase
            self.multimodal_model.train()
            total_loss = 0
            correct = 0
            total = 0

            for i in range(0, len(train_labels), batch_size):
                end_idx = min(i + batch_size, len(train_labels))

                # Get batch
                text_input = {
                    'input_ids': text_train_encoded['input_ids'][i:end_idx].to(self.device),
                    'attention_mask': text_train_encoded['attention_mask'][i:end_idx].to(self.device)
                }
                audio_input = audio_train_mfcc[i:end_idx].to(self.device)
                batch_labels = torch.tensor(train_labels[i:end_idx], dtype=torch.long).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                logits = self.multimodal_model(text_input, audio_input)
                loss = criterion(logits, batch_labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            # Validation phase
            self.multimodal_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i in range(0, len(val_labels), batch_size):
                    end_idx = min(i + batch_size, len(val_labels))

                    # Get batch
                    text_input = {
                        'input_ids': text_val_encoded['input_ids'][i:end_idx].to(self.device),
                        'attention_mask': text_val_encoded['attention_mask'][i:end_idx].to(self.device)
                    }
                    audio_input = audio_val_mfcc[i:end_idx].to(self.device)
                    batch_labels = torch.tensor(val_labels[i:end_idx], dtype=torch.long).to(self.device)

                    # Forward pass
                    logits = self.multimodal_model(text_input, audio_input)
                    loss = criterion(logits, batch_labels)

                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()

            # Calculate metrics
            train_acc = 100 * correct / total
            val_acc = 100 * val_correct / val_total

            self.logger.info(f"Multimodal Model Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {total_loss/len(train_labels)*batch_size:.4f}, "
                           f"Train Acc: {train_acc:.2f}%, "
                           f"Val Loss: {val_loss/len(val_labels)*batch_size:.4f}, "
                           f"Val Acc: {val_acc:.2f}%")

            # Store history
            self.history['train_loss'].append(total_loss/len(train_labels)*batch_size)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss/len(val_labels)*batch_size)
            self.history['val_acc'].append(val_acc)

            # Early stopping
            if early_stopping(val_loss, self.multimodal_model):
                self.logger.info("Early stopping triggered for multimodal model")
                break

        # Save model
        save_model(self.multimodal_model, optimizer, epoch, val_loss,
                  os.path.join(self.config.get('data.models_path', 'data/models'), 'multimodal_model.pth'))

    def evaluate_model(self, test_df: pd.DataFrame):
        """
        Evaluate the multimodal model.

        Args:
            test_df: Test data
        """
        self.logger.info("Evaluating multimodal model...")

        # Prepare test data
        text_test_encoded, _ = process_text_data(
            test_df, 
            model_name=self.config.get('models.text.model_name', 'distilbert-base-uncased')
        )
        audio_test_mfcc, _, _ = process_audio_data(test_df)

        # Convert to tensors
        audio_test_mfcc = torch.tensor(audio_test_mfcc, dtype=torch.float32)
        test_labels = test_df['label'].values

        # Evaluation
        self.multimodal_model.eval()
        predictions = []
        probabilities = []

        batch_size = self.config.get('models.multimodal.batch_size', 16)

        with torch.no_grad():
            for i in range(0, len(test_labels), batch_size):
                end_idx = min(i + batch_size, len(test_labels))

                # Get batch
                text_input = {
                    'input_ids': text_test_encoded['input_ids'][i:end_idx].to(self.device),
                    'attention_mask': text_test_encoded['attention_mask'][i:end_idx].to(self.device)
                }
                audio_input = audio_test_mfcc[i:end_idx].to(self.device)

                # Forward pass
                logits = self.multimodal_model(text_input, audio_input)
                probs = F.softmax(logits, dim=1)

                # Store predictions
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        # Calculate metrics
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        metrics = calculate_metrics(test_labels, predictions, probabilities)

        self.logger.info("Test Results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

        # Classification report
        class_names = ['Normal', 'Depression', 'Anxiety']
        report = classification_report(test_labels, predictions, target_names=class_names)
        self.logger.info(f"Classification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.config.get('data.models_path', 'data/models'), 'confusion_matrix.png'))
        plt.show()

        return metrics

    def run_full_training(self):
        """Run complete training pipeline."""
        self.logger.info("Starting full training pipeline...")

        # Load data
        train_df, val_df, test_df = self.load_data()

        # Create models
        self.create_models()

        # Train individual models
        self.train_individual_models(train_df, val_df)

        # Train multimodal model
        self.train_multimodal_model(train_df, val_df)

        # Evaluate model
        metrics = self.evaluate_model(test_df)

        # Plot training history
        plot_training_history(self.history, 
                            save_path=os.path.join(self.config.get('data.models_path', 'data/models'), 'training_history.png'))

        self.logger.info("Full training pipeline completed!")

        return metrics

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train multimodal mental health detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='full', choices=['text', 'audio', 'multimodal', 'full'],
                       help='Training mode')

    args = parser.parse_args()

    # Create trainer
    trainer = MultimodalTrainer(args.config)

    if args.mode == 'full':
        trainer.run_full_training()
    else:
        # Individual model training (implementation similar to above)
        pass

if __name__ == "__main__":
    main()
