#!/usr/bin/env python3
"""
Test script to verify webrtcvad-wheels installation and functionality
for the multimodal mental health detection project.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_webrtcvad_installation():
    """Test webrtcvad-wheels installation"""
    print("🔍 Testing webrtcvad-wheels installation...")

    try:
        import webrtcvad
        print("✅ webrtcvad-wheels imported successfully")

        # Test VAD creation
        vad = webrtcvad.Vad()
        vad.set_mode(2)
        print("✅ VAD instance created and configured")

        # Test with synthetic audio
        sample_rate = 16000
        frame_duration = 30  # ms
        frame_length = int(sample_rate * frame_duration / 1000)

        # Create synthetic audio frame
        audio_frame = np.random.randint(-32768, 32767, frame_length, dtype=np.int16)
        audio_bytes = audio_frame.tobytes()

        # Test VAD processing
        is_speech = vad.is_speech(audio_bytes, sample_rate)
        print(f"✅ VAD processing test: is_speech = {is_speech}")

        return True

    except ImportError as e:
        print(f"❌ Failed to import webrtcvad: {e}")
        return False
    except Exception as e:
        print(f"❌ VAD test failed: {e}")
        return False

def test_audio_preprocessing():
    """Test audio preprocessing functionality"""
    print("\n🔍 Testing audio preprocessing...")

    try:
        from audio_preprocessing import AudioPreprocessor

        # Initialize preprocessor
        preprocessor = AudioPreprocessor(sample_rate=16000, frame_duration=30)
        print("✅ AudioPreprocessor initialized")

        # Test synthetic audio creation
        synthetic_audio = preprocessor.create_synthetic_audio(duration=2, label=1)
        print(f"✅ Synthetic audio created: shape {synthetic_audio.shape}")

        # Test VAD functionality
        vad_results = preprocessor.detect_voice_activity(synthetic_audio)
        print(f"✅ VAD detection: {len(vad_results)} frames, {sum(vad_results)} speech frames")

        # Test feature extraction
        mfcc = preprocessor.extract_mfcc(synthetic_audio)
        mel_spec = preprocessor.extract_mel_spectrogram(synthetic_audio)
        prosodic = preprocessor.extract_prosodic_features(synthetic_audio)

        print(f"✅ Feature extraction:")
        print(f"   - MFCC: {mfcc.shape}")
        print(f"   - Mel-spectrogram: {mel_spec.shape}")
        print(f"   - Prosodic features: {prosodic.shape}")

        return True

    except ImportError as e:
        print(f"❌ Failed to import audio_preprocessing: {e}")
        return False
    except Exception as e:
        print(f"❌ Audio preprocessing test failed: {e}")
        return False

def test_dependencies():
    """Test other required dependencies"""
    print("\n🔍 Testing other dependencies...")

    dependencies = [
        'torch',
        'transformers',
        'librosa',
        'numpy',
        'pandas',
        'scipy',
        'sklearn'
    ]

    missing_deps = []

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} imported successfully")
        except ImportError:
            print(f"❌ {dep} not found")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True

def main():
    """Main test function"""
    print("="*60)
    print("🧠 Multimodal Mental Health Detection - System Test")
    print("="*60)

    tests_passed = 0
    total_tests = 3

    # Test 1: webrtcvad-wheels installation
    if test_webrtcvad_installation():
        tests_passed += 1

    # Test 2: Audio preprocessing
    if test_audio_preprocessing():
        tests_passed += 1

    # Test 3: Dependencies
    if test_dependencies():
        tests_passed += 1

    # Summary
    print("\n" + "="*60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready for use.")
        print("\n🚀 Next steps:")
        print("   1. Run 'python train.py' to start training")
        print("   2. Check 'data/' directory for synthetic data")
        print("   3. Monitor training progress in console")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Common solutions:")
        print("   1. pip install -r requirements.txt")
        print("   2. pip install webrtcvad-wheels==2.0.14")
        print("   3. Check Python version compatibility")

    print("="*60)

if __name__ == "__main__":
    main()
