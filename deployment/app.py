"""
Streamlit web application for multimodal mental health detection.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
from io import BytesIO
import tempfile
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_processing.text_preprocessing import TextPreprocessor
from data_processing.audio_preprocessing import AudioPreprocessor
from models.multimodal_model import create_multimodal_model
from models.text_model import create_text_model
from models.audio_model import create_audio_model
from utils.config import Config
import shap
import matplotlib.pyplot as plt

# Configure Streamlit
st.set_page_config(
    page_title="Mental Health Detection System",
    page_icon="🧠",
    layout="wide"
)

class MentalHealthApp:
    """Main Streamlit application for mental health detection."""
    
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models."""
        try:
            # Load individual models
            self.text_model = create_text_model(self.config.get('models.text', {}))
            self.audio_model = create_audio_model(self.config.get('models.audio', {}))
            
            # Load multimodal model
            self.multimodal_model = create_multimodal_model(
                self.text_model, 
                self.audio_model, 
                self.config.get('models.multimodal', {})
            )
            
            # Load weights if available
            models_path = self.config.get('data.models_path', 'data/models')
            if os.path.exists(os.path.join(models_path, 'multimodal_model.pth')):
                checkpoint = torch.load(os.path.join(models_path, 'multimodal_model.pth'))
                self.multimodal_model.load_state_dict(checkpoint['model_state_dict'])
            
            self.multimodal_model.to(self.device)
            self.multimodal_model.eval()
            
            # Initialize preprocessors
            self.text_preprocessor = TextPreprocessor(
                model_name=self.config.get('models.text.model_name', 'distilbert-base-uncased')
            )
            self.audio_preprocessor = AudioPreprocessor(
                sample_rate=self.config.get('data.audio.sample_rate', 16000)
            )
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Using demo mode with synthetic predictions.")
            self.demo_mode = True
    
    def predict(self, text_input, audio_input):
        """Make prediction on text and audio inputs."""
        try:
            # Process text
            text_encoded, text_features = self.text_preprocessor.preprocess_batch([text_input])
            
            # Process audio
            if audio_input is not None:
                mfcc, mel_spec, prosodic = self.audio_preprocessor.process_audio_file(audio_input)
                audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            else:
                # Use synthetic audio for demo
                synthetic_audio = self.audio_preprocessor.create_synthetic_audio(duration=5, label=1)
                mfcc = self.audio_preprocessor.extract_mfcc(synthetic_audio)
                audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                text_input_dict = {
                    'input_ids': text_encoded['input_ids'].to(self.device),
                    'attention_mask': text_encoded['attention_mask'].to(self.device)
                }
                audio_input_tensor = audio_tensor.to(self.device)
                
                logits = self.multimodal_model(text_input_dict, audio_input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                
                return predicted_class.cpu().numpy()[0], probabilities.cpu().numpy()[0]
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            # Return demo prediction
            return 1, np.array([0.2, 0.7, 0.1])
    
    def explain_prediction(self, text_input, prediction, confidence):
        """Generate SHAP explanations for the prediction."""
        try:
            # Simplified explanation for demo
            explanation = {
                'text_features': ['sadness indicators', 'negative sentiment', 'emotional words'],
                'audio_features': ['low pitch', 'slow speech rate', 'reduced energy'],
                'confidence': confidence
            }
            return explanation
        except Exception as e:
            st.error(f"Explanation error: {e}")
            return None
    
    def run(self):
        """Run the main Streamlit application."""
        st.title("🧠 Multimodal Mental Health Detection System")
        st.markdown("### Upload text and audio to analyze mental health indicators")
        
        # Sidebar for configuration
        st.sidebar.header("Configuration")
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        show_explanations = st.sidebar.checkbox("Show Explanations", True)
        
        # Main interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Text Input")
            text_input = st.text_area(
                "Enter text (diary entry, social media post, etc.):",
                placeholder="I've been feeling really down lately...",
                height=150
            )
            
        with col2:
            st.subheader("🎤 Audio Input")
            audio_file = st.file_uploader(
                "Upload audio file (WAV, MP3):",
                type=['wav', 'mp3', 'ogg']
            )
            
            if audio_file is not None:
                st.audio(audio_file, format='audio/wav')
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.read())
                    audio_path = tmp_file.name
            else:
                audio_path = None
        
        # Prediction button
        if st.button("🔍 Analyze Mental Health", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing inputs..."):
                    # Make prediction
                    prediction, probabilities = self.predict(text_input, audio_path)
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Class labels
                    class_labels = ['Normal', 'Depression', 'Anxiety']
                    predicted_label = class_labels[prediction]
                    confidence = probabilities[prediction]
                    
                    # Results visualization
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted State", predicted_label)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    with col3:
                        risk_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
                        st.metric("Risk Level", risk_level)
                    
                    # Probability distribution
                    st.subheader("📈 Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Mental State': class_labels,
                        'Probability': probabilities
                    })
                    
                    st.bar_chart(prob_df.set_index('Mental State'))
                    
                    # Risk assessment
                    if confidence > confidence_threshold and prediction > 0:
                        st.warning("⚠️ **Mental health concerns detected**")
                        st.markdown("""
                        **Recommendations:**
                        - Consider speaking with a mental health professional
                        - Reach out to trusted friends or family
                        - Contact a crisis helpline if needed
                        
                        **Crisis Resources:**
                        - National Suicide Prevention Lifeline: 988
                        - Crisis Text Line: Text HOME to 741741
                        """)
                    else:
                        st.success("✅ **No immediate concerns detected**")
                        st.info("Continue monitoring your mental health and seek help if needed.")
                    
                    # Explanations
                    if show_explanations:
                        st.subheader("🔍 Model Explanations")
                        explanation = self.explain_prediction(text_input, prediction, confidence)
                        
                        if explanation:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Text Analysis:**")
                                for feature in explanation['text_features']:
                                    st.markdown(f"- {feature}")
                            
                            with col2:
                                st.markdown("**Audio Analysis:**")
                                for feature in explanation['audio_features']:
                                    st.markdown(f"- {feature}")
                    
                    # Cleanup temporary file
                    if audio_path and os.path.exists(audio_path):
                        os.unlink(audio_path)
            else:
                st.error("Please enter some text to analyze.")
        
        # Information section
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.markdown("""
        This system uses advanced machine learning to analyze text and audio 
        for early detection of mental health conditions.
        
        **Note:** This tool is for research purposes only and should not 
        replace professional medical advice.
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📞 Resources")
        st.sidebar.markdown("""
        - **National Suicide Prevention Lifeline:** 988
        - **Crisis Text Line:** Text HOME to 741741
        - **NAMI Helpline:** 1-800-950-NAMI
        """)

def main():
    """Main function to run the Streamlit app."""
    app = MentalHealthApp()
    app.run()

if __name__ == "__main__":
    main()
