import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from audio_preprocessing import AudioPreprocessor, process_audio_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing for mental health detection"""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = 512

    def preprocess_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of texts"""
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoded

class TextModel(nn.Module):
    """Text model for mental health detection"""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_classes: int = 3):
        super(TextModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class AudioModel(nn.Module):
    """Audio model for mental health detection"""

    def __init__(self, mfcc_dim: int = 13, mel_dim: int = 128, 
                 prosodic_dim: int = 16, num_classes: int = 3):
        super(AudioModel, self).__init__()

        # MFCC processing
        self.mfcc_conv = nn.Sequential(
            nn.Conv1d(mfcc_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Mel-spectrogram processing
        self.mel_conv = nn.Sequential(
            nn.Conv1d(mel_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Prosodic features processing
        self.prosodic_fc = nn.Sequential(
            nn.Linear(prosodic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Fusion and classification
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, mfcc, mel_spec, prosodic):
        # Process MFCC
        mfcc_out = self.mfcc_conv(mfcc.transpose(1, 2)).squeeze(-1)

        # Process Mel-spectrogram
        mel_out = self.mel_conv(mel_spec.transpose(1, 2)).squeeze(-1)

        # Process prosodic features
        prosodic_out = self.prosodic_fc(prosodic)

        # Fuse features
        fused = torch.cat([mfcc_out, mel_out, prosodic_out], dim=1)
        logits = self.fusion_fc(fused)

        return logits

class MultimodalModel(nn.Module):
    """Multimodal model combining text and audio"""

    def __init__(self, text_model: TextModel, audio_model: AudioModel, 
                 num_classes: int = 3, fusion_type: str = "late"):
        super(MultimodalModel, self).__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.fusion_type = fusion_type

        if fusion_type == "late":
            self.fusion_fc = nn.Linear(num_classes * 2, num_classes)
        elif fusion_type == "early":
            # For early fusion, we'd need to modify the architecture
            self.fusion_fc = nn.Linear(768 + 256, num_classes)  # BERT hidden + audio features

    def forward(self, text_input, audio_input):
        if self.fusion_type == "late":
            text_logits = self.text_model(**text_input)
            audio_logits = self.audio_model(*audio_input)

            # Late fusion
            combined_logits = torch.cat([text_logits, audio_logits], dim=1)
            final_logits = self.fusion_fc(combined_logits)

            return final_logits
        else:
            # Early fusion would require different implementation
            pass

class MultimodalDataset(Dataset):
    """Dataset for multimodal mental health detection"""

    def __init__(self, df: pd.DataFrame, text_preprocessor: TextPreprocessor, 
                 audio_preprocessor: AudioPreprocessor):
        self.df = df
        self.text_preprocessor = text_preprocessor
        self.audio_preprocessor = audio_preprocessor

        # Preprocess all data
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()

        # Process audio data
        self.mfcc_data, self.mel_data, self.prosodic_data = process_audio_data(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Get audio features
        mfcc = torch.FloatTensor(self.mfcc_data[idx])
        mel_spec = torch.FloatTensor(self.mel_data[idx])
        prosodic = torch.FloatTensor(self.prosodic_data[idx])

        return {
            'text': text,
            'mfcc': mfcc,
            'mel_spec': mel_spec,
            'prosodic': prosodic,
            'label': torch.LongTensor([label])
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    texts = [item['text'] for item in batch]
    mfcc = torch.stack([item['mfcc'] for item in batch])
    mel_spec = torch.stack([item['mel_spec'] for item in batch])
    prosodic = torch.stack([item['prosodic'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])

    return {
        'texts': texts,
        'mfcc': mfcc,
        'mel_spec': mel_spec,
        'prosodic': prosodic,
        'labels': labels
    }

class MentalHealthTrainer:
    """Trainer for multimodal mental health detection"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize preprocessors
        self.text_preprocessor = TextPreprocessor(config.get('text_model', 'distilbert-base-uncased'))
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=config.get('sample_rate', 16000),
            frame_duration=config.get('frame_duration', 30)
        )

        # Initialize models
        self.text_model = TextModel(config.get('text_model', 'distilbert-base-uncased'))
        self.audio_model = AudioModel()
        self.multimodal_model = MultimodalModel(
            self.text_model, 
            self.audio_model, 
            fusion_type=config.get('fusion_type', 'late')
        )

        # Move models to device
        self.text_model.to(self.device)
        self.audio_model.to(self.device)
        self.multimodal_model.to(self.device)

        # Initialize optimizers
        self.text_optimizer = optim.Adam(self.text_model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.audio_optimizer = optim.Adam(self.audio_model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.multimodal_optimizer = optim.Adam(self.multimodal_model.parameters(), lr=config.get('learning_rate', 1e-4))

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"MentalHealthTrainer initialized with device: {self.device}")

    def create_synthetic_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Create synthetic data for training"""
        data = []

        for i in range(num_samples):
            # Generate synthetic text
            if i % 3 == 0:  # Normal
                text = "I'm feeling good today and everything is going well."
                label = 0
            elif i % 3 == 1:  # Depression
                text = "I feel sad and empty inside. Nothing seems to matter anymore."
                label = 1
            else:  # Anxiety
                text = "I'm constantly worried about everything and can't relax."
                label = 2

            # Generate synthetic audio path
            audio_path = f"data/synthetic_audio/synthetic_audio_{i}.wav"

            data.append({
                'text': text,
                'audio_filepath': audio_path,
                'label': label
            })

        return pd.DataFrame(data)

    def train_text_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10):
        """Train the text model"""
        logger.info("Training text model...")

        for epoch in range(epochs):
            self.text_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                texts = batch['texts']
                labels = batch['labels'].to(self.device)

                # Tokenize texts
                encoded = self.text_preprocessor.preprocess_batch(texts)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                # Forward pass
                self.text_optimizer.zero_grad()
                outputs = self.text_model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.text_optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation
            val_acc = self.evaluate_model(self.text_model, val_loader, model_type='text')

            logger.info(f"Text Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {100*train_correct/train_total:.2f}%, "
                       f"Val Acc: {val_acc:.2f}%")

    def train_audio_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10):
        """Train the audio model"""
        logger.info("Training audio model...")

        for epoch in range(epochs):
            self.audio_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                mfcc = batch['mfcc'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                prosodic = batch['prosodic'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                self.audio_optimizer.zero_grad()
                outputs = self.audio_model(mfcc, mel_spec, prosodic)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.audio_optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation
            val_acc = self.evaluate_model(self.audio_model, val_loader, model_type='audio')

            logger.info(f"Audio Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {100*train_correct/train_total:.2f}%, "
                       f"Val Acc: {val_acc:.2f}%")

    def train_multimodal_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10):
        """Train the multimodal model"""
        logger.info("Training multimodal model...")

        for epoch in range(epochs):
            self.multimodal_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                texts = batch['texts']
                mfcc = batch['mfcc'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                prosodic = batch['prosodic'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Prepare text input
                encoded = self.text_preprocessor.preprocess_batch(texts)
                text_input = {
                    'input_ids': encoded['input_ids'].to(self.device),
                    'attention_mask': encoded['attention_mask'].to(self.device)
                }

                # Prepare audio input
                audio_input = (mfcc, mel_spec, prosodic)

                # Forward pass
                self.multimodal_optimizer.zero_grad()
                outputs = self.multimodal_model(text_input, audio_input)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.multimodal_optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation
            val_acc = self.evaluate_model(self.multimodal_model, val_loader, model_type='multimodal')

            logger.info(f"Multimodal Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {100*train_correct/train_total:.2f}%, "
                       f"Val Acc: {val_acc:.2f}%")

    def evaluate_model(self, model, data_loader: DataLoader, model_type: str = 'multimodal') -> float:
        """Evaluate model performance"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                labels = batch['labels'].to(self.device)

                if model_type == 'text':
                    texts = batch['texts']
                    encoded = self.text_preprocessor.preprocess_batch(texts)
                    input_ids = encoded['input_ids'].to(self.device)
                    attention_mask = encoded['attention_mask'].to(self.device)
                    outputs = model(input_ids, attention_mask)

                elif model_type == 'audio':
                    mfcc = batch['mfcc'].to(self.device)
                    mel_spec = batch['mel_spec'].to(self.device)
                    prosodic = batch['prosodic'].to(self.device)
                    outputs = model(mfcc, mel_spec, prosodic)

                else:  # multimodal
                    texts = batch['texts']
                    mfcc = batch['mfcc'].to(self.device)
                    mel_spec = batch['mel_spec'].to(self.device)
                    prosodic = batch['prosodic'].to(self.device)

                    encoded = self.text_preprocessor.preprocess_batch(texts)
                    text_input = {
                        'input_ids': encoded['input_ids'].to(self.device),
                        'attention_mask': encoded['attention_mask'].to(self.device)
                    }
                    audio_input = (mfcc, mel_spec, prosodic)

                    outputs = model(text_input, audio_input)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def save_models(self, model_dir: str = 'models'):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)

        # Save text model
        torch.save({
            'model_state_dict': self.text_model.state_dict(),
            'optimizer_state_dict': self.text_optimizer.state_dict(),
        }, os.path.join(model_dir, 'text_model.pth'))

        # Save audio model
        torch.save({
            'model_state_dict': self.audio_model.state_dict(),
            'optimizer_state_dict': self.audio_optimizer.state_dict(),
        }, os.path.join(model_dir, 'audio_model.pth'))

        # Save multimodal model
        torch.save({
            'model_state_dict': self.multimodal_model.state_dict(),
            'optimizer_state_dict': self.multimodal_optimizer.state_dict(),
        }, os.path.join(model_dir, 'multimodal_model.pth'))

        logger.info(f"Models saved to {model_dir}")

    def run_full_training(self):
        """Run complete training pipeline"""
        logger.info("Starting full training pipeline...")

        # Create synthetic data
        logger.info("Creating synthetic data...")
        df = self.create_synthetic_data(num_samples=1000)

        # Save synthetic data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/synthetic_data.csv', index=False)

        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

        # Create datasets
        train_dataset = MultimodalDataset(train_df, self.text_preprocessor, self.audio_preprocessor)
        val_dataset = MultimodalDataset(val_df, self.text_preprocessor, self.audio_preprocessor)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            collate_fn=collate_fn
        )

        # Train individual models
        epochs = self.config.get('epochs', 10)

        self.train_text_model(train_loader, val_loader, epochs)
        self.train_audio_model(train_loader, val_loader, epochs)
        self.train_multimodal_model(train_loader, val_loader, epochs)

        # Save models
        self.save_models()

        logger.info("Training completed successfully!")

def main():
    """Main training function"""
    config = {
        'text_model': 'distilbert-base-uncased',
        'sample_rate': 16000,
        'frame_duration': 30,
        'fusion_type': 'late',
        'learning_rate': 1e-4,
        'batch_size': 16,
        'epochs': 10
    }

    trainer = MentalHealthTrainer(config)
    trainer.run_full_training()

if __name__ == "__main__":
    main()
