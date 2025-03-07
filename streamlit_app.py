import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import os
import tempfile
from tensorflow.keras.models import load_model
import time
import datetime

# Current information
CURRENT_DATE = "2025-03-07 11:19:11"
USER_LOGIN = "Mangun10"

class SleepAnalyzer:
    def __init__(self, model_path='models/sleep_audio_model.h5'):
        """Initialize sleep analyzer with a pre-trained model"""
        self.model = load_model(model_path)
        self.sr = 8000  # Sampling rate
        self.n_mfcc = 13  # Number of MFCCs
        
    def extract_features(self, audio_data):
        """Extract MFCC features from audio segment"""
        try:
            # Ensure consistent length
            segment_samples = self.sr  # 1 second
            if len(audio_data) < segment_samples:
                audio_data = np.pad(audio_data, (0, segment_samples - len(audio_data)), 'constant')
            else:
                audio_data = audio_data[:segment_samples]
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=self.n_mfcc)
            
            return mfccs
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None
    
    def analyze_audio(self, audio_path, segment_duration=30):
        """Analyze a sleep audio recording"""
        try:
            # Load audio
            y, _ = librosa.load(audio_path, sr=self.sr)
            
            # Calculate total duration
            total_duration_seconds = len(y) / self.sr
            total_duration_minutes = total_duration_seconds / 60
            
            st.info(f"Analyzing audio: {total_duration_minutes:.2f} minutes duration")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Split into segments
            segment_samples = int(self.sr * segment_duration)
            n_segments = int(np.ceil(len(y) / segment_samples))
            
            # Initialize variables
            snoring_segments = []
            snoring_confidences = []
            
            # Process each segment
            for i in range(n_segments):
                # Extract segment
                start_sample = i * segment_samples
                end_sample = min(start_sample + segment_samples, len(y))
                segment = y[start_sample:end_sample]
                
                # Calculate segment times
                segment_start_time = start_sample / self.sr
                segment_end_time = end_sample / self.sr
                
                # Split into 1-second sub-segments
                sub_segment_length = self.sr
                n_sub_segments = int(np.ceil(len(segment) / sub_segment_length))
                
                # Count snoring in sub-segments
                sub_segment_snoring_count = 0
                sub_segment_confidences = []
                
                for j in range(n_sub_segments):
                    # Extract sub-segment
                    sub_start = j * sub_segment_length
                    sub_end = min(sub_start + sub_segment_length, len(segment))
                    sub_segment = segment[sub_start:sub_end]
                    
                    # Skip if too short
                    if len(sub_segment) < 0.5 * sub_segment_length:
                        continue
                    
                    # Extract features
                    mfccs = self.extract_features(sub_segment)
                    
                    if mfccs is None:
                        continue
                    
                    # Prepare for model input
                    mfccs = mfccs[np.newaxis, ..., np.newaxis]
                    
                    # Make prediction
                    pred = self.model.predict(mfccs, verbose=0)[0]
                    pred_class = np.argmax(pred)
                    confidence = pred[pred_class]
                    
                    # If snoring detected
                    if pred_class == 1:  # 1 = snoring
                        sub_segment_snoring_count += 1
                        sub_segment_confidences.append(float(confidence))
                
                # If significant snoring detected in the segment
                if sub_segment_snoring_count > 0.3 * n_sub_segments:
                    avg_confidence = np.mean(sub_segment_confidences) if sub_segment_confidences else 0
                    snoring_segments.append((segment_start_time, segment_end_time))
                    snoring_confidences.append(avg_confidence)
                
                # Update progress
                progress_bar.progress((i + 1) / n_segments)
            
            # Calculate results
            total_snoring_duration = sum(end - start for start, end in snoring_segments)
            snoring_percentage = (total_snoring_duration / total_duration_seconds) * 100 if total_duration_seconds > 0 else 0
            snoring_segments_count = len(snoring_segments)
            
            # Calculate snoring frequency (episodes per hour)
            hours_of_sleep = total_duration_seconds / 3600
            snoring_frequency = snoring_segments_count / hours_of_sleep if hours_of_sleep > 0 else 0
            
            # Prepare results
            results = {
                'total_duration_seconds': total_duration_seconds,
                'total_duration_minutes': total_duration_minutes,
                'snoring_segments': snoring_segments,
                'snoring_confidences': snoring_confidences,
                'total_snoring_duration': total_snoring_duration,
                'snoring_percentage': snoring_percentage,
                'snoring_segments_count': snoring_segments_count,
                'snoring_frequency': snoring_frequency
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error analyzing audio: {str(e)}")
            return None
    
    def generate_suggestions(self, results):
        """Generate suggestions based on analysis results"""
        snoring_percentage = results['snoring_percentage']
        
        # Determine severity
        if snoring_percentage < 10:
            severity = "Minimal"
            assessment = "Your snoring appears to be minimal and likely not a significant concern."
        elif snoring_percentage < 30:
            severity = "Mild"
            assessment = "You have mild snoring that might occasionally disturb your sleep quality."
        elif snoring_percentage < 50:
            severity = "Moderate"
            assessment = "Your moderate snoring could be impacting your sleep quality and may warrant attention."
        else:
            severity = "Severe"
            assessment = "Your snoring is severe and may indicate sleep apnea or other sleep disorders that require medical attention."
        
        # Base suggestions
        base_suggestions = [
            "Maintain a healthy weight through diet and exercise",
            "Avoid alcohol and sedatives before bedtime",
            "Establish a consistent sleep schedule"
        ]
        
        # Additional suggestions based on severity
        additional_suggestions = []
        
        if severity == "Minimal":
            additional_suggestions = [
                "Continue with good sleep hygiene practices",
                "Consider using a humidifier if your environment is dry"
            ]
        elif severity == "Mild":
            additional_suggestions = [
                "Try sleeping on your side instead of your back",
                "Use an extra pillow to elevate your head",
                "Consider nasal strips to improve airflow"
            ]
        elif severity == "Moderate":
            additional_suggestions = [
                "Use a specially designed anti-snoring pillow",
                "Try mouth exercises to strengthen the muscles in your throat",
                "Consider using a mandibular advancement device",
                "Consult with a doctor if snoring persists or worsens"
            ]
        else:  # Severe
            additional_suggestions = [
                "Consult with a sleep specialist as soon as possible",
                "Consider getting tested for sleep apnea",
                "Look into CPAP therapy options",
                "Discuss surgical options with your doctor if appropriate"
            ]
        
        # Combine suggestions
        all_suggestions = base_suggestions + additional_suggestions
        
        # Create suggestions dictionary
        suggestions = {
            'severity': severity,
            'assessment': assessment,
            'suggestions': all_suggestions
        }
        
        return suggestions
    
    def create_visualization(self, audio_path, results):
        """Create visualization of analysis results"""
        try:
            # Load audio for visualization
            y, _ = librosa.load(audio_path, sr=self.sr)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot waveform
            librosa.display.waveshow(y, sr=self.sr, ax=ax1, alpha=0.6)
            ax1.set_title('Audio Waveform with Snoring Segments Highlighted')
            ax1.set_xlabel('Time (s)')
            
            # Highlight snoring segments on waveform
            for start, end in results['snoring_segments']:
                ax1.axvspan(start, end, color='red', alpha=0.3)
            
            # Create snoring timeline
            x_timeline = []
            y_timeline = []
            
            # Create timeline data
            max_time = len(y) / self.sr
            timeline_resolution = max(1, int(max_time / 300))  # One point per timeline_resolution seconds
            
            for i in range(0, int(max_time), timeline_resolution):
                x_timeline.append(i)
                
                # Check if this time point is in a snoring segment
                in_snoring_segment = False
                for start, end in results['snoring_segments']:
                    if start <= i <= end:
                        in_snoring_segment = True
                        break
                
                y_timeline.append(1 if in_snoring_segment else 0)
            
            # Plot timeline
            ax2.plot(x_timeline, y_timeline, 'b-', linewidth=2)
            ax2.set_title('Snoring Timeline (1 = Snoring, 0 = No Snoring)')
            ax2.set_xlabel('Time (s)')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['No Snoring', 'Snoring'])
            ax2.grid(True)
            
            # Add summary statistics as text annotation
            summary_text = (
                f"Total Duration: {results['total_duration_minutes']:.2f} minutes\n"
                f"Snoring Duration: {results['total_snoring_duration'] / 60:.2f} minutes\n"
                f"Snoring Percentage: {results['snoring_percentage']:.2f}%\n"
                f"Snoring Episodes: {results['snoring_segments_count']}\n"
                f"Snoring Frequency: {results['snoring_frequency']:.2f} episodes/hour"
            )
            
            # Add text box for summary
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2.text(0.02, 0.3, summary_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None


# Set up Streamlit application
def main():
    st.set_page_config(page_title="Sleep Audio Analysis", layout="wide")
    
    st.title("Sleep Audio Analysis")
    st.write(f"Date: {CURRENT_DATE}")
    st.write(f"User: {USER_LOGIN}")
    
    # Check if model exists
    model_path = "models/sleep_audio_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Initialize analyzer
    analyzer = SleepAnalyzer(model_path)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Results", "Information"])
    
    with tab1:
        st.header("Upload Sleep Audio")
        st.write("Upload an audio recording of your sleep to analyze snoring patterns.")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a sleep audio file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            # Display audio player
            st.audio(uploaded_file, format="audio/wav")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            # Analysis button
            if st.button("Analyze Sleep Recording"):
                # Store path in session state
                if "temp_path" not in st.session_state:
                    st.session_state.temp_path = temp_path
                
                with st.spinner("Analyzing sleep recording... This may take several minutes for long recordings."):
                    # Perform analysis
                    results = analyzer.analyze_audio(temp_path)
                    
                    if results:
                        # Store results in session state
                        st.session_state.results = results
                        st.session_state.suggestions = analyzer.generate_suggestions(results)
                        
                        # Create visualization
                        fig = analyzer.create_visualization(temp_path, results)
                        if fig:
                            st.session_state.fig = fig
                        
                        # Success message and switch to results tab
                        st.success("Analysis completed! Check the Results tab for details.")
                    else:
                        st.error("Analysis failed. Please try another recording.")
    
    with tab2:
        st.header("Analysis Results")
        
        # Check if results exist in session state
        if "results" in st.session_state and st.session_state.results:
            results = st.session_state.results
            suggestions = st.session_state.suggestions
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sleep Duration", f"{results['total_duration_minutes']:.2f} min")
            with col2:
                st.metric("Snoring Duration", f"{results['total_snoring_duration'] / 60:.2f} min")
            with col3:
                st.metric("Snoring Percentage", f"{results['snoring_percentage']:.2f}%")
            with col4:
                st.metric("Snoring Episodes", f"{results['snoring_segments_count']}")
            
            # Display visualization if available
            if "fig" in st.session_state:
                st.pyplot(st.session_state.fig)
            
            # Display assessment
            st.subheader("Sleep Assessment")
            st.info(f"**Severity**: {suggestions['severity']}")
            st.write(suggestions['assessment'])
            
            # Display suggestions
            st.subheader("Recommendations")
            for suggestion in suggestions['suggestions']:
                st.write(f"- {suggestion}")
            
            # Download results option
            # if st.button("Generate Report PDF"):
            #     st.info("PDF report generation would be implemented here.")
        else:
            st.info("No analysis results available. Please upload and analyze a sleep recording first.")
    
    with tab3:
        st.header("About Sleep Analysis")
        st.write("""
        ## How it works
        
        This application uses a trained CNN-LSTM neural network to detect snoring patterns in sleep audio recordings.
        
        ### Analysis Process:
        1. The sleep recording is divided into 30-second segments
        2. Each segment is further divided into 1-second chunks for analysis
        3. The model determines if each chunk contains snoring
        4. Results are aggregated to calculate overall snoring metrics
        5. Personalized suggestions are generated based on the severity
        
        ### Snoring Severity Levels:
        - **Minimal** (<10%): Occasional snoring, minimal impact on sleep quality
        - **Mild** (10-30%): Intermittent snoring, may slightly affect sleep quality
        - **Moderate** (30-50%): Regular snoring, likely impacts sleep quality
        - **Severe** (>50%): Constant snoring, significant impact on sleep quality, possible sleep apnea
        
        ### Important Note:
        This tool is for informational purposes only and is not a substitute for professional medical advice.
        If you have concerns about your sleep quality or suspect you may have sleep apnea, please consult a healthcare professional.
        """)
    
    # Handle cleanup of temporary files
    if st.session_state.get("temp_path") and os.path.exists(st.session_state.get("temp_path")):
        try:
            # We want to keep the file while the app is running but delete when the session ends
            # Unfortunately, Streamlit doesn't have a clean way to detect session end
            # We'll rely on automatic cleanup by the OS for temporary files
            pass
        except:
            pass


if __name__ == "__main__":
    main()