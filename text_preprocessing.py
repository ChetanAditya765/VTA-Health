
"""
Text preprocessing for the multimodal mental health detection system.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing pipeline for mental health detection."""

    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 max_length: int = 512,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True):
        """
        Initialize text preprocessor.

        Args:
            model_name: Name of the transformer model
            max_length: Maximum sequence length
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
        """
        self.model_name = model_name
        self.max_length = max_length
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

        logging.info(f"Initialized TextPreprocessor with model: {model_name}")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and formatting.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)

        # Remove extra punctuation
        text = re.sub(r'[.,!?;:]{2,}', '.', text)

        return text.strip()

    def tokenize_and_preprocess(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text.

        Args:
            text: Text to tokenize

        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Lemmatize if specified
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def extract_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract various text features.

        Args:
            texts: List of texts to extract features from

        Returns:
            Dictionary of feature arrays
        """
        features = {}

        # Basic statistics
        features['text_length'] = np.array([len(text) for text in texts])
        features['word_count'] = np.array([len(text.split()) for text in texts])
        features['sentence_count'] = np.array([len(text.split('.')) for text in texts])

        # Punctuation features
        features['exclamation_count'] = np.array([text.count('!') for text in texts])
        features['question_count'] = np.array([text.count('?') for text in texts])
        features['comma_count'] = np.array([text.count(',') for text in texts])

        # Emotional indicators
        depression_words = ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 'tired', 'exhausted']
        anxiety_words = ['anxious', 'worried', 'nervous', 'panic', 'afraid', 'scared', 'stress']
        positive_words = ['happy', 'joy', 'excited', 'good', 'great', 'wonderful', 'amazing']

        features['depression_word_count'] = np.array([
            sum(1 for word in depression_words if word in text.lower()) for text in texts
        ])
        features['anxiety_word_count'] = np.array([
            sum(1 for word in anxiety_words if word in text.lower()) for text in texts
        ])
        features['positive_word_count'] = np.array([
            sum(1 for word in positive_words if word in text.lower()) for text in texts
        ])

        return features

    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode texts using transformer tokenizer.

        Args:
            texts: List of texts to encode

        Returns:
            Dictionary of encoded tensors
        """
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Tokenize
        encoded = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def preprocess_batch(self, texts: List[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of texts to preprocess

        Returns:
            Tuple of (encoded_texts, features)
        """
        # Encode texts
        encoded = self.encode_texts(texts)

        # Extract features
        features = self.extract_features(texts)

        return encoded, features

def create_tfidf_features(texts: List[str], max_features: int = 5000) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF features from texts.

    Args:
        texts: List of texts
        max_features: Maximum number of features

    Returns:
        Tuple of (features, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    features = vectorizer.fit_transform(texts).toarray()
    return features, vectorizer

def process_text_data(df: pd.DataFrame, 
                     text_column: str = 'text',
                     model_name: str = "distilbert-base-uncased",
                     max_length: int = 512) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Process text data from DataFrame.

    Args:
        df: DataFrame with text data
        text_column: Name of the text column
        model_name: Name of the transformer model
        max_length: Maximum sequence length

    Returns:
        Tuple of (encoded_texts, features)
    """
    preprocessor = TextPreprocessor(model_name=model_name, max_length=max_length)

    texts = df[text_column].fillna("").astype(str).tolist()
    encoded, features = preprocessor.preprocess_batch(texts)

    return encoded, features

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    sample_texts = [
        "I feel very sad and hopeless today. Nothing seems to matter anymore.",
        "I'm feeling great! Today was an amazing day and I'm excited about tomorrow.",
        "I'm worried about everything. My heart is racing and I can't relax."
    ]

    preprocessor = TextPreprocessor()
    encoded, features = preprocessor.preprocess_batch(sample_texts)

    print("Encoded shape:", encoded['input_ids'].shape)
    print("Features keys:", list(features.keys()))
    print("Sample features:", {k: v[:3] for k, v in features.items()})
