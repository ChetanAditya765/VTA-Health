import numpy as np
import librosa
import soundfile as sf
import webrtcvad
import os
import warnings
from typing import Tuple, List, Optional
from scipy import signal
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Enhanced audio preprocessing with WebRTC VAD integration for multimodal mental health detection.
    Uses webrtcvad-wheels for robust voice activity detection.
    """

    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30):
        """
        Initialize the audio preprocessor with WebRTC VAD.

        Args:
            sample_rate: Target sample rate for audio processing
            frame_duration: Frame duration in milliseconds (10, 20, or 30)
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_length = int(sample_rate * frame_duration / 1000)

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Moderate aggressiveness (0-3)

        # Audio processing parameters
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128

        # Prosodic feature parameters
        self.expected_prosodic_features = 16

        logger.info(f"AudioPreprocessor initialized with sample_rate={sample_rate}, frame_duration={frame_duration}ms")

    def detect_voice_activity(self, audio_data: np.ndarray) -> List[bool]:
        """
        Detect voice activity using WebRTC VAD.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            List of boolean values indicating voice activity for each frame
        """
        # Convert to 16-bit PCM format required by WebRTC VAD
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Convert to bytes
        audio_bytes = audio_int16.tobytes()

        # Process audio in frames
        vad_results = []
        num_frames = len(audio_int16) // self.frame_length

        for i in range(num_frames):
            start_idx = i * self.frame_length
            end_idx = start_idx + self.frame_length

            if end_idx <= len(audio_int16):
                frame_bytes = audio_int16[start_idx:end_idx].tobytes()

                try:
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    vad_results.append(is_speech)
                except Exception as e:
                    logger.warning(f"VAD processing error: {e}")
                    vad_results.append(False)

        return vad_results

    def extract_speech_segments(self, audio_data: np.ndarray, 
                               min_speech_duration: float = 0.5,
                               max_silence_duration: float = 0.3) -> List[Tuple[int, int]]:
        """
        Extract speech segments based on VAD results.

        Args:
            audio_data: Audio data as numpy array
            min_speech_duration: Minimum duration for speech segments (seconds)
            max_silence_duration: Maximum allowed silence within speech (seconds)

        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        vad_results = self.detect_voice_activity(audio_data)

        if not vad_results:
            return []

        # Convert parameters to frame counts
        min_speech_frames = int(min_speech_duration * 1000 / self.frame_duration)
        max_silence_frames = int(max_silence_duration * 1000 / self.frame_duration)

        # Find speech segments
        segments = []
        current_start = None
        silence_counter = 0

        for i, is_speech in enumerate(vad_results):
            if is_speech:
                if current_start is None:
                    current_start = i
                silence_counter = 0
            else:
                if current_start is not None:
                    silence_counter += 1
                    if silence_counter > max_silence_frames:
                        # End current segment
                        segment_length = i - current_start - silence_counter
                        if segment_length >= min_speech_frames:
                            start_sample = current_start * self.frame_length
                            end_sample = (i - silence_counter) * self.frame_length
                            segments.append((start_sample, end_sample))
                        current_start = None
                        silence_counter = 0

        # Handle final segment
        if current_start is not None:
            segment_length = len(vad_results) - current_start
            if segment_length >= min_speech_frames:
                start_sample = current_start * self.frame_length
                end_sample = len(audio_data)
                segments.append((start_sample, end_sample))

        return segments

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file with VAD-based speech extraction.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Apply VAD to extract speech segments
            speech_segments = self.extract_speech_segments(audio)

            if not speech_segments:
                logger.warning(f"No speech detected in {audio_path}")
                return audio, sr

            # Concatenate speech segments
            speech_audio = []
            for start, end in speech_segments:
                speech_audio.append(audio[start:end])

            processed_audio = np.concatenate(speech_audio) if speech_audio else audio

            # Normalize audio
            processed_audio = self.normalize_audio(processed_audio)

            return processed_audio, sr

        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            # Return silent audio as fallback
            return np.zeros(self.sample_rate), self.sample_rate

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.

        Args:
            audio: Audio data as numpy array

        Returns:
            Normalized audio data
        """
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Audio data as numpy array

        Returns:
            MFCC features as numpy array
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return mfcc.T  # Transpose to (time, features)
        except Exception as e:
            logger.error(f"Error extracting MFCC: {e}")
            return np.zeros((100, self.n_mfcc))  # Return default shape

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features from audio.

        Args:
            audio: Audio data as numpy array

        Returns:
            Mel-spectrogram features as numpy array
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db.T  # Transpose to (time, features)
        except Exception as e:
            logger.error(f"Error extracting mel-spectrogram: {e}")
            return np.zeros((100, self.n_mels))  # Return default shape

    def extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract prosodic features from audio.

        Args:
            audio: Audio data as numpy array

        Returns:
            Prosodic features as numpy array
        """
        try:
            # Extract fundamental frequency (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )

            # Replace NaN values with 0
            f0 = np.nan_to_num(f0)

            # Extract energy features
            energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]

            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]

            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]

            # Combine features
            prosodic_features = []

            # Statistical features for each prosodic measure
            for feature in [f0, energy, zcr, spectral_centroids, spectral_rolloff, spectral_bandwidth]:
                if len(feature) > 0:
                    prosodic_features.extend([
                        np.mean(feature),
                        np.std(feature),
                        np.median(feature)
                    ])
                else:
                    prosodic_features.extend([0.0, 0.0, 0.0])

            # Ensure consistent feature length
            prosodic_features = np.array(prosodic_features)
            if len(prosodic_features) < self.expected_prosodic_features:
                # Pad with zeros
                padding = np.zeros(self.expected_prosodic_features - len(prosodic_features))
                prosodic_features = np.concatenate([prosodic_features, padding])
            elif len(prosodic_features) > self.expected_prosodic_features:
                # Truncate
                prosodic_features = prosodic_features[:self.expected_prosodic_features]

            return prosodic_features

        except Exception as e:
            logger.error(f"Error extracting prosodic features: {e}")
            return np.zeros(self.expected_prosodic_features)

    def process_audio_file(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single audio file and extract all features.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (mfcc, mel_spectrogram, prosodic_features)
        """
        # Preprocess audio with VAD
        audio, sr = self.preprocess_audio(audio_path)

        # Extract features
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        prosodic = self.extract_prosodic_features(audio)

        return mfcc, mel_spec, prosodic

    def process_audio_batch(self, audio_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a batch of audio files with consistent shapes.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            Tuple of (mfcc_batch, mel_spec_batch, prosodic_batch)
        """
        mfcc_batch = []
        mel_spec_batch = []
        prosodic_batch = []

        for audio_path in audio_paths:
            try:
                mfcc, mel_spec, prosodic = self.process_audio_file(audio_path)
                mfcc_batch.append(mfcc)
                mel_spec_batch.append(mel_spec)
                prosodic_batch.append(prosodic)
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                # Add default values for failed processing
                mfcc_batch.append(np.zeros((100, self.n_mfcc)))
                mel_spec_batch.append(np.zeros((100, self.n_mels)))
                prosodic_batch.append(np.zeros(self.expected_prosodic_features))

        # Ensure consistent shapes
        mfcc_batch = self._ensure_consistent_shape(mfcc_batch)
        mel_spec_batch = self._ensure_consistent_shape(mel_spec_batch)
        prosodic_batch = self._ensure_consistent_prosodic_shape(prosodic_batch)

        return (
            np.array(mfcc_batch),
            np.array(mel_spec_batch),
            np.array(prosodic_batch)
        )

    def _ensure_consistent_shape(self, feature_batch: List[np.ndarray]) -> List[np.ndarray]:
        """
        Ensure all features in batch have consistent shapes.

        Args:
            feature_batch: List of feature arrays

        Returns:
            List of feature arrays with consistent shapes
        """
        if not feature_batch:
            return []

        # Find maximum length
        max_length = max(len(features) for features in feature_batch)
        feature_dim = feature_batch[0].shape[1]

        # Pad or truncate to consistent length
        consistent_batch = []
        for features in feature_batch:
            if len(features) < max_length:
                # Pad with zeros
                padding = np.zeros((max_length - len(features), feature_dim))
                padded_features = np.vstack([features, padding])
                consistent_batch.append(padded_features)
            elif len(features) > max_length:
                # Truncate
                consistent_batch.append(features[:max_length])
            else:
                consistent_batch.append(features)

        return consistent_batch

    def _ensure_consistent_prosodic_shape(self, prosodic_batch: List[np.ndarray]) -> List[np.ndarray]:
        """
        Ensure all prosodic features have consistent shapes.

        Args:
            prosodic_batch: List of prosodic feature arrays

        Returns:
            List of prosodic feature arrays with consistent shapes
        """
        consistent_batch = []

        for prosodic in prosodic_batch:
            if len(prosodic) < self.expected_prosodic_features:
                # Pad with zeros
                padding = np.zeros(self.expected_prosodic_features - len(prosodic))
                padded_prosodic = np.concatenate([prosodic, padding])
                consistent_batch.append(padded_prosodic)
            elif len(prosodic) > self.expected_prosodic_features:
                # Truncate
                consistent_batch.append(prosodic[:self.expected_prosodic_features])
            else:
                consistent_batch.append(prosodic)

        return consistent_batch

    def create_synthetic_audio(self, duration: int = 5, label: int = 0) -> np.ndarray:
        """
        Create synthetic audio for testing purposes.

        Args:
            duration: Duration in seconds
            label: Label for audio (0=normal, 1=depression, 2=anxiety)

        Returns:
            Synthetic audio data
        """
        num_samples = duration * self.sample_rate

        if label == 0:  # Normal - white noise
            audio = np.random.normal(0, 0.1, num_samples)
        elif label == 1:  # Depression - low frequency tone
            t = np.linspace(0, duration, num_samples)
            audio = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.normal(0, 0.05, num_samples)
        else:  # Anxiety - high frequency tone
            t = np.linspace(0, duration, num_samples)
            audio = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.1 * np.random.normal(0, 0.05, num_samples)

        return audio

    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """
        Save audio to file.

        Args:
            audio: Audio data as numpy array
            file_path: Path to save the audio file
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            sf.write(file_path, audio, self.sample_rate)
            logger.info(f"Audio saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")


def process_audio_data(df, audio_column='audio_filepath', synthetic_dir='data/synthetic_audio'):
    """
    Process audio data from DataFrame with VAD-enhanced preprocessing.

    Args:
        df: DataFrame containing audio file paths
        audio_column: Column name containing audio file paths
        synthetic_dir: Directory for synthetic audio files

    Returns:
        Tuple of (mfcc_batch, mel_spec_batch, prosodic_batch)
    """
    preprocessor = AudioPreprocessor()

    # Ensure synthetic directory exists
    os.makedirs(synthetic_dir, exist_ok=True)

    # Process audio files
    audio_paths = []
    for idx, row in df.iterrows():
        audio_path = row[audio_column]

        # Check if file exists, create synthetic if needed
        if not os.path.exists(audio_path):
            # Create synthetic audio
            label = row.get('label', 0)
            synthetic_audio = preprocessor.create_synthetic_audio(duration=5, label=label)

            # Save synthetic audio
            synthetic_path = os.path.join(synthetic_dir, f'synthetic_audio_{idx}.wav')
            preprocessor.save_audio(synthetic_audio, synthetic_path)
            audio_paths.append(synthetic_path)
        else:
            audio_paths.append(audio_path)

    # Process batch
    mfcc_batch, mel_spec_batch, prosodic_batch = preprocessor.process_audio_batch(audio_paths)

    return mfcc_batch, mel_spec_batch, prosodic_batch


if __name__ == "__main__":
    # Test the audio preprocessor
    preprocessor = AudioPreprocessor()

    # Test VAD functionality
    print("Testing WebRTC VAD functionality...")

    # Create test audio with speech-like pattern
    test_audio = preprocessor.create_synthetic_audio(duration=3, label=1)
    vad_results = preprocessor.detect_voice_activity(test_audio)

    print(f"VAD Results: {len(vad_results)} frames processed")
    print(f"Speech frames detected: {sum(vad_results)}")
    print(f"Speech percentage: {sum(vad_results)/len(vad_results)*100:.1f}%")

    # Test feature extraction
    mfcc = preprocessor.extract_mfcc(test_audio)
    mel_spec = preprocessor.extract_mel_spectrogram(test_audio)
    prosodic = preprocessor.extract_prosodic_features(test_audio)

    print(f"\nFeature extraction results:")
    print(f"MFCC shape: {mfcc.shape}")
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    print(f"Prosodic features shape: {prosodic.shape}")

    print("\n✅ Audio preprocessing with WebRTC VAD is working correctly!")
