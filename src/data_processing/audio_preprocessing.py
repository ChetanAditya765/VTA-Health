
"""
Audio preprocessing for the multimodal mental health detection system.
"""

import librosa
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
import logging
import os
import soundfile as sf
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """Audio preprocessing pipeline for mental health detection."""

    def __init__(self, 
                 sample_rate: int = 16000,
                 max_length: int = 30,
                 hop_length: int = 512,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 n_mfcc: int = 13):
        """
        Initialize audio preprocessor.

        Args:
            sample_rate: Target sample rate
            max_length: Maximum audio length in seconds
            hop_length: Hop length for STFT
            n_fft: FFT window size
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        # Calculate derived parameters
        self.max_samples = sample_rate * max_length
        self.n_frames = int(self.max_samples // hop_length) + 1

        logging.info(f"Initialized AudioPreprocessor with sample_rate={sample_rate}, max_length={max_length}s")

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio signal as numpy array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Normalize audio
            audio = librosa.util.normalize(audio)

            return audio

        except Exception as e:
            logging.error(f"Error loading audio {audio_path}: {e}")
            # Return silence if loading fails
            return np.zeros(self.max_samples)

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio signal.

        Args:
            audio: Raw audio signal

        Returns:
            Preprocessed audio signal
        """
        # Pad or truncate to fixed length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), 'constant')

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Apply pre-emphasis filter
        audio = signal.lfilter([1, -0.97], [1], audio)

        return audio

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features.

        Args:
            audio: Audio signal

        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )

        # Pad or truncate to fixed length
        if mfcc.shape[1] > self.n_frames:
            mfcc = mfcc[:, :self.n_frames]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.n_frames - mfcc.shape[1])), 'constant')

        return mfcc

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram features.

        Args:
            audio: Audio signal

        Returns:
            Mel spectrogram features
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )

        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or truncate to fixed length
        if mel_spec.shape[1] > self.n_frames:
            mel_spec = mel_spec[:, :self.n_frames]
        else:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, self.n_frames - mel_spec.shape[1])), 'constant')

        return mel_spec

    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features from audio.

        Args:
            audio: Audio signal

        Returns:
            Dictionary of prosodic features
        """
        features = {}

        # Fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )

        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            features['pitch_mean'] = np.mean(f0_clean)
            features['pitch_std'] = np.std(f0_clean)
            features['pitch_min'] = np.min(f0_clean)
            features['pitch_max'] = np.max(f0_clean)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0
            features['pitch_range'] = 0.0

        # Energy and intensity
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_max'] = np.max(rms)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        # Tempo
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features['tempo'] = tempo

        # Speaking rate (approximate)
        features['speaking_rate'] = len(beats) / (len(audio) / self.sample_rate)

        return features

    def create_synthetic_audio(self, duration: int = 5, label: int = 0) -> np.ndarray:
        """
        Create synthetic audio for testing.

        Args:
            duration: Duration in seconds
            label: Label for synthetic audio (0=normal, 1=depression, 2=anxiety)

        Returns:
            Synthetic audio signal
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        if label == 0:  # Normal
            # Clear speech with normal pitch
            freq = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
        elif label == 1:  # Depression
            # Lower pitch, slower speech
            freq = 120 + 20 * np.sin(2 * np.pi * 0.3 * t)
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
        else:  # Anxiety
            # Higher pitch, faster speech
            freq = 180 + 80 * np.sin(2 * np.pi * 0.8 * t)
            audio = 0.6 * np.sin(2 * np.pi * freq * t)

        # Add some noise
        noise = 0.05 * np.random.randn(len(audio))
        audio += noise

        return audio

    def process_audio_file(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Process a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (mfcc_features, mel_spectrogram, prosodic_features)
        """
        if not os.path.exists(audio_path):
            logging.warning(f"Audio file not found: {audio_path}, creating synthetic audio")
            audio = self.create_synthetic_audio()
        else:
            audio = self.load_audio(audio_path)

        # Preprocess audio
        audio = self.preprocess_audio(audio)

        # Extract features
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        prosodic = self.extract_prosodic_features(audio)

        return mfcc, mel_spec, prosodic

    def process_audio_batch(self, audio_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a batch of audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            Tuple of (mfcc_batch, mel_spec_batch, prosodic_batch)
        """
        mfcc_batch = []
        mel_spec_batch = []
        prosodic_batch = []

        for audio_path in audio_paths:
            mfcc, mel_spec, prosodic = self.process_audio_file(audio_path)
            mfcc_batch.append(mfcc)
            mel_spec_batch.append(mel_spec)
            prosodic_batch.append(list(prosodic.values()))

        return (
            np.array(mfcc_batch),
            np.array(mel_spec_batch),
            np.array(prosodic_batch)
        )

def process_audio_data(df: pd.DataFrame, 
                      audio_column: str = 'audio_path',
                      sample_rate: int = 16000,
                      max_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process audio data from DataFrame.

    Args:
        df: DataFrame with audio data
        audio_column: Name of the audio path column
        sample_rate: Target sample rate
        max_length: Maximum audio length in seconds

    Returns:
        Tuple of (mfcc_features, mel_spectrograms, prosodic_features)
    """
    preprocessor = AudioPreprocessor(sample_rate=sample_rate, max_length=max_length)

    audio_paths = df[audio_column].fillna("").astype(str).tolist()
    mfcc_batch, mel_spec_batch, prosodic_batch = preprocessor.process_audio_batch(audio_paths)

    return mfcc_batch, mel_spec_batch, prosodic_batch

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = AudioPreprocessor()

    # Create synthetic audio samples
    synthetic_audio = preprocessor.create_synthetic_audio(duration=5, label=1)

    # Extract features
    mfcc = preprocessor.extract_mfcc(synthetic_audio)
    mel_spec = preprocessor.extract_mel_spectrogram(synthetic_audio)
    prosodic = preprocessor.extract_prosodic_features(synthetic_audio)

    print("MFCC shape:", mfcc.shape)
    print("Mel spectrogram shape:", mel_spec.shape)
    print("Prosodic features:", list(prosodic.keys()))
    print("Sample prosodic values:", {k: round(v, 3) for k, v in list(prosodic.items())[:5]})
