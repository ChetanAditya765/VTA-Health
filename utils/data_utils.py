
"""
Data utilities for the multimodal mental health detection system.
"""

import pandas as pd
import numpy as np
import os
import torch
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

def load_daic_woz_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load DAIC-WOZ dataset.

    Args:
        data_path: Path to the DAIC-WOZ dataset

    Returns:
        Tuple of (text_df, audio_df, labels_df)
    """
    try:
        # Load text data
        text_path = os.path.join(data_path, "text")
        text_files = [f for f in os.listdir(text_path) if f.endswith('.txt')]

        text_data = []
        for file in text_files:
            participant_id = file.replace('.txt', '')
            with open(os.path.join(text_path, file), 'r', encoding='utf-8') as f:
                text_content = f.read()
            text_data.append({
                'participant_id': participant_id,
                'text': text_content
            })

        text_df = pd.DataFrame(text_data)

        # Load audio metadata
        audio_path = os.path.join(data_path, "audio")
        audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

        audio_data = []
        for file in audio_files:
            participant_id = file.replace('.wav', '')
            audio_data.append({
                'participant_id': participant_id,
                'audio_path': os.path.join(audio_path, file)
            })

        audio_df = pd.DataFrame(audio_data)

        # Load labels
        labels_path = os.path.join(data_path, "labels.csv")
        labels_df = pd.read_csv(labels_path)

        return text_df, audio_df, labels_df

    except Exception as e:
        logging.error(f"Error loading DAIC-WOZ data: {e}")
        raise

def create_synthetic_data(num_samples: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic data for testing purposes.

    Args:
        num_samples: Number of synthetic samples to create

    Returns:
        Tuple of (text_df, audio_df, labels_df)
    """
    np.random.seed(42)

    # Create synthetic text data
    depression_texts = [
        "I feel very sad and hopeless today",
        "Nothing seems to matter anymore",
        "I can't get out of bed",
        "Everything feels overwhelming",
        "I don't see the point in anything",
        "I feel empty inside",
        "I can't concentrate on anything",
        "I feel like a burden to everyone",
        "I have no energy for anything",
        "I feel worthless"
    ]

    normal_texts = [
        "I'm feeling good today",
        "Looking forward to the weekend",
        "Had a great day at work",
        "Excited about my new project",
        "Feeling grateful for my friends",
        "I'm optimistic about the future",
        "Today was productive",
        "I feel energized and motivated",
        "Life is good",
        "I'm happy with my progress"
    ]

    anxiety_texts = [
        "I'm worried about everything",
        "I can't stop thinking about what could go wrong",
        "My heart is racing",
        "I feel nervous all the time",
        "I'm scared something bad will happen",
        "I can't relax",
        "I'm constantly on edge",
        "I feel like I'm losing control",
        "I'm afraid of making mistakes",
        "I can't shake this feeling of dread"
    ]

    # Generate synthetic data
    text_data = []
    audio_data = []
    labels_data = []

    for i in range(num_samples):
        participant_id = f"P{i:04d}"

        # Random label assignment
        label = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])  # Normal, Depression, Anxiety

        if label == 0:
            text = np.random.choice(normal_texts)
            label_name = "Normal"
        elif label == 1:
            text = np.random.choice(depression_texts)
            label_name = "Depression"
        else:
            text = np.random.choice(anxiety_texts)
            label_name = "Anxiety"

        # Add some noise to text
        text = text + f" {np.random.choice(['', 'you know', 'like', 'um', 'yeah'])}"

        text_data.append({
            'participant_id': participant_id,
            'text': text
        })

        audio_data.append({
            'participant_id': participant_id,
            'audio_path': f"synthetic_audio_{participant_id}.wav"
        })

        labels_data.append({
            'participant_id': participant_id,
            'label': label,
            'label_name': label_name,
            'phq8_score': np.random.randint(0, 25) if label == 1 else np.random.randint(0, 10)
        })

    return pd.DataFrame(text_data), pd.DataFrame(audio_data), pd.DataFrame(labels_data)

def split_data(df: pd.DataFrame, 
               test_size: float = 0.2, 
               val_size: float = 0.1, 
               stratify_col: str = None,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: DataFrame to split
        test_size: Size of test set
        val_size: Size of validation set
        stratify_col: Column to use for stratification
        random_state: Random state for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    stratify = df[stratify_col].values if stratify_col else None

    # First split: train + val, test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=stratify, 
        random_state=random_state
    )

    # Second split: train, val
    if val_size > 0:
        stratify_train_val = train_val_df[stratify_col].values if stratify_col else None
        val_size_adjusted = val_size / (1 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=stratify_train_val,
            random_state=random_state
        )
    else:
        train_df = train_val_df
        val_df = pd.DataFrame()

    return train_df, val_df, test_df

def encode_labels(labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integers.

    Args:
        labels: List of string labels

    Returns:
        Tuple of (encoded_labels, label_encoder)
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder

def create_data_loaders(features: np.ndarray, 
                       labels: np.ndarray, 
                       batch_size: int = 32,
                       shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader from features and labels.

    Args:
        features: Input features
        labels: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        DataLoader
    """
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(features),
        torch.LongTensor(labels)
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
