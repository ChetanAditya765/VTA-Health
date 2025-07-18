import numpy as np
import librosa
import soundfile as sf
import webrtcvad
import os
import warnings
from typing import List, Tuple, Optional
from scipy.signal import butter, sosfiltfilt
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Comprehensive audio preprocessing for multimodal mental health detection.
    Handles MFCC extraction, mel spectrograms, and prosodic features.
    Includes caching of synthetic audio files so they're generated only once.
    """
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13, 
                 n_fft: int = 2048, hop_length: int = 512,
                 synthetic_dir: str = 'data/synthetic_audio'):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3
        
        # Expected feature dimensions
        self.expected_mfcc_frames = 100
        self.expected_mel_frames = 128
        self.expected_prosodic_dim = 16
        
        # Directory to store synthetic audio files
        self.synthetic_dir = synthetic_dir
        os.makedirs(self.synthetic_dir, exist_ok=True)
    
    # --------------------------- CORE METHODS --------------------------- #

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load and preprocess audio file. If the file is synthetic and missing, create and cache it."""
        try:
            if not os.path.exists(file_path) and self._is_synthetic_path(file_path):
                label = self._infer_label_from_path(file_path)
                duration = 3.0
                self._create_and_save_synthetic_audio(file_path, duration, label)
            
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio) + 1e-6)
       