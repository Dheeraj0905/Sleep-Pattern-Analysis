import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import tempfile
import pickle
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Current information
CURRENT_DATE = "2025-03-07 12:10:48"
USER_LOGIN = "Mangun10"

class SleepAnalyzer:
    def __init__(self, 
             audio_model_path='models/sleep_audio_model.h5',
             disorder_model_path='models/sleep_disorder_model.pkl',
             quality_model_path='models/sleep_quality_model.pkl'):
        """Initialize comprehensive sleep analyzer with pre-trained models"""
        # Load audio model
        try:
            self.audio_model = load_model(audio_model_path)
            self.audio_model_loaded = True
        except Exception as e:
            st.warning(f"Could not load audio model: {e}")
            self.audio_model_loaded = False
        
        # Load disorder prediction model
        try:
            with open(disorder_model_path, 'rb') as f:
                self.disorder_model = pickle.load(f)
            self.disorder_model_loaded = True
        except Exception as e:
            st.warning(f"Could not load sleep disorder model: {e}")
            self.disorder_model_loaded = False
        
        # Load quality prediction model
        try:
            with open(quality_model_path, 'rb') as f:
                self.quality_model = pickle.load(f)
            self.quality_model_loaded = True
        except Exception as e:
            st.warning(f"Could not load sleep quality model: {e}")
            self.quality_model_loaded = False
        
        # Set audio parameters
        self.sr = 8000
        self.n_mfcc = 13
        
        # Initialize scalers for preprocessing
        self.disorder_scaler = StandardScaler()
        self.quality_scaler = StandardScaler()
    
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
        if not self.audio_model_loaded:
            st.error("Audio model not loaded. Cannot analyze audio.")
            return None
        
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
                    pred = self.audio_model.predict(mfccs, verbose=0)[0]
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
    
    def predict_sleep_disorder(self, input_data):
        """Predict sleep disorder from input data"""
        if not self.disorder_model_loaded:
            st.error("Sleep disorder model not loaded.")
            return None
        
        try:
            # First, get all required components from the loaded model data
            model = self.disorder_model['model']
            label_encoders = self.disorder_model['label_encoders']
            target_encoder = self.disorder_model['target_encoder']
            scaler = self.disorder_model['scaler']
            feature_list = self.disorder_model['feature_list']
            numerical_columns = self.disorder_model['numerical_columns']
            categorical_columns = self.disorder_model['categorical_columns']
            
            # Convert input_data to a more suitable format matching the sample implementation
            sample_dict = {
                'Gender': input_data['Gender'].iloc[0],
                'Age': input_data['Age'].iloc[0],
                'Occupation': input_data['Occupation'].iloc[0],
                'Sleep Duration': input_data['Sleep Duration'].iloc[0],
                'Quality of Sleep': input_data['Quality of Sleep'].iloc[0],
                'Physical Activity Level': input_data['Physical Activity Level'].iloc[0],
                'Stress Level': input_data['Stress Level'].iloc[0],
                'BMI Category': input_data['BMI Category'].iloc[0],
                'Systolic BP': input_data['Systolic BP'].iloc[0],
                'Diastolic BP': input_data['Diastolic BP'].iloc[0],
                'Heart Rate': input_data['Heart Rate'].iloc[0],
                'Daily Steps': input_data['Daily Steps'].iloc[0]
            }
            
            # Create a DataFrame with the sample
            sample_df = pd.DataFrame([sample_dict])
            
            # Feature Engineering
            sample_df['Sleep Efficiency'] = sample_df['Quality of Sleep'] / sample_df['Sleep Duration']
            sample_df['Stress Activity Ratio'] = sample_df['Stress Level'] / sample_df['Physical Activity Level']
            sample_df['BP Difference'] = sample_df['Systolic BP'] - sample_df['Diastolic BP']
            
            # BMI and Heart Rate Interaction - Note the BMI values are not yet encoded here
            sample_df['HR_BMI_Factor'] = 0
            if sample_df['BMI Category'].iloc[0] == 'Obese':
                sample_df['HR_BMI_Factor'] = sample_df['Heart Rate'] * 1.5
            elif sample_df['BMI Category'].iloc[0] == 'Overweight':
                sample_df['HR_BMI_Factor'] = sample_df['Heart Rate'] * 1.2
            elif sample_df['BMI Category'].iloc[0] == 'Normal':
                sample_df['HR_BMI_Factor'] = sample_df['Heart Rate'] * 1.0
            
            # Steps Adequacy
            sample_df['Steps Adequacy'] = sample_df['Daily Steps'] / 10000
            
            # Encode categorical features
            for col in categorical_columns:
                if col in sample_df.columns:
                    sample_df[col] = label_encoders[col].transform(sample_df[col])
            
            # Scale numerical features
            sample_df[numerical_columns] = scaler.transform(sample_df[numerical_columns])
            
            # Reorder columns to match the training data
            sample_df = sample_df[feature_list]
            
            # Make prediction
            pred_encoded = model.predict(sample_df)[0]
            prediction = target_encoder.inverse_transform([pred_encoded])[0]
            
            # Get probabilities for each class
            probabilities = None
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(sample_df)[0]
                prob_dict = {target_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(pred_proba)}
                probabilities = prob_dict
            
            return {
                'prediction': prediction,
                'probabilities': probabilities
            }
            
        except Exception as e:
            st.error(f"Error predicting sleep disorder: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            st.write(f"Debug info - model type: {type(self.disorder_model)}")
            st.write(f"Debug info - input columns: {list(input_data.columns)}")
            return None
    
    def predict_sleep_quality(self, input_data):
        """Predict sleep quality from input data"""
        if not self.quality_model_loaded:
            st.error("Sleep quality model not loaded.")
            return None
        
        try:
            # Scale features
            scaled_features = self.quality_scaler.fit_transform(input_data)
            
            # Make prediction
            prediction = self.quality_model.predict(scaled_features)
            
            # Return result
            return {
                'prediction': prediction[0],
                'score': float(prediction[0])
            }
        except Exception as e:
            st.error(f"Error predicting sleep quality: {str(e)}")
            return None
    
    def create_visualization(self, audio_path, results):
        """Create visualization of audio analysis results"""
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

    def generate_suggestions(self, audio_results=None, disorder_results=None, quality_results=None):
        """Generate comprehensive suggestions based on all analysis results"""
        suggestions = []
        severity = "Unknown"
        general_suggestions = []
        
        # Process audio results if available
        if audio_results:
            snoring_percentage = audio_results['snoring_percentage']
            
            # Determine snoring severity
            if snoring_percentage < 10:
                severity = "Minimal"
                suggestions.append("Your snoring is minimal and likely not affecting your sleep quality.")
                suggestions.append("Continue practicing good sleep hygiene.")
            elif snoring_percentage < 30:
                severity = "Mild"
                suggestions.append("You have mild snoring that might occasionally disturb your sleep.")
                suggestions.append("Try sleeping on your side instead of your back.")
                suggestions.append("Consider using nasal strips to improve airflow.")
            elif snoring_percentage < 50:
                severity = "Moderate"
                suggestions.append("Your moderate snoring could be impacting your sleep quality significantly.")
                suggestions.append("Use a specially designed anti-snoring pillow.")
                suggestions.append("Try mouth exercises to strengthen throat muscles.")
                suggestions.append("Consider a humidifier if your environment is dry.")
            else:
                severity = "Severe"
                suggestions.append("Your snoring is severe and may indicate sleep apnea.")
                suggestions.append("Consult with a sleep specialist as soon as possible.")
                suggestions.append("Consider getting tested for sleep apnea.")
        
        # Process disorder prediction if available
        if disorder_results:
            disorder = disorder_results['prediction']

            if disorder == "None":
                suggestions.append("No specific sleep disorder detected from your physiological data.")
            elif disorder == "Insomnia":
                suggestions.append("Your data suggests potential insomnia patterns.")
                suggestions.append("Establish a regular sleep schedule and bedtime routine.")
                suggestions.append("Avoid screens at least 1 hour before bedtime.")
                suggestions.append("Create a comfortable, dark, and quiet sleep environment.")
                suggestions.append("Consider relaxation techniques like meditation before sleep.")
            elif disorder == "Sleep Apnea":
                suggestions.append("Your data indicates patterns consistent with sleep apnea.")
                suggestions.append("Consult with a healthcare provider for proper diagnosis.")
                suggestions.append("Consider weight management strategies if appropriate.")
                suggestions.append("Avoid alcohol and sedatives before sleep.")
                suggestions.append("Sleep on your side rather than your back.")
        
        # Process quality prediction if available
        if quality_results:
            quality_score = quality_results['score']
            
            if quality_score <= 3:
                suggestions.append("Your sleep quality appears to be poor based on physiological markers.")
                suggestions.append("Focus on improving your sleep environment.")
                suggestions.append("Consider stress reduction techniques like meditation or deep breathing.")
            elif quality_score <= 7:
                suggestions.append("Your sleep quality is moderate based on physiological markers.")
                suggestions.append("Small improvements to your bedtime routine could help.")
                suggestions.append("Maintain consistent sleep and wake times.")
            else:
                suggestions.append("Your sleep quality appears to be good based on physiological markers.")
                suggestions.append("Continue your healthy sleep habits.")
        
        # Add general recommendations
        general_suggestions = [
            "Maintain a healthy weight through diet and exercise.",
            "Avoid caffeine and heavy meals at least 4 hours before bedtime.",
            "Exercise regularly, but not within 2 hours of bedtime.",
            "Ensure your bedroom is at a comfortable temperature (around 65°F/18°C).",
            "Use your bed only for sleep and intimacy to strengthen the mental association."
        ]
        
        # Return all suggestions
        return {
            'severity': severity,
            'specific_suggestions': suggestions,
            'general_suggestions': general_suggestions
        }


# Streamlit application
def main():
    st.set_page_config(page_title="Comprehensive Sleep Analysis", layout="wide")
    
    st.title("Comprehensive Sleep Analysis")
    st.write(f"Date: 2025-03-07 12:59:44")
    st.write(f"User: Mangun10")
    
    # Initialize analyzer
    analyzer = SleepAnalyzer(
        audio_model_path='models/sleep_audio_model.h5',
        disorder_model_path='models/sleep_disorder_model.pkl',
        quality_model_path='models/sleep_quality_model.pkl'
    )
    
    # Create tabs (only 2 as requested)
    tab1, tab2 = st.tabs(["Input Data", "Analysis Results"])
    
    # Variables to store inputs and results
    if 'disorder_inputs' not in st.session_state:
        st.session_state.disorder_inputs = {}
    if 'quality_inputs' not in st.session_state:
        st.session_state.quality_inputs = {}
    if 'audio_results' not in st.session_state:
        st.session_state.audio_results = None
    if 'disorder_results' not in st.session_state:
        st.session_state.disorder_results = None
    if 'quality_results' not in st.session_state:
        st.session_state.quality_results = None
    if 'combined_suggestions' not in st.session_state:
        st.session_state.combined_suggestions = None
    
    #############
    # Tab 1: Input Data (combined personal data and audio)
    #############
    with tab1:
        st.header("Sleep Analysis Input Data")
        
        # Create sections with expanders
        with st.expander("Personal & Lifestyle Data", expanded=True):
            st.write("Enter your personal information for sleep disorder assessment:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic information
                gender_options = ["Male", "Female"]
                gender = st.selectbox("Gender", gender_options, 
                                  index=gender_options.index(st.session_state.disorder_inputs.get('Gender', "Male")),
                                  key="gender_disorder")
                st.session_state.disorder_inputs['Gender'] = gender
                
                age = st.number_input("Age", min_value=18, max_value=100, 
                                  value=st.session_state.disorder_inputs.get('Age', 30),
                                  key="age_disorder")
                st.session_state.disorder_inputs['Age'] = age
                
                # Use the occupation list from the data
                occupation_options = ["Software Engineer", "Doctor", "Sales Representative", "Teacher",
                        "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer",
                        "Salesperson", "Manager"]
                        
                occupation = st.selectbox("Occupation", occupation_options,
                                      index=occupation_options.index(st.session_state.disorder_inputs.get('Occupation', "Software Engineer")) if st.session_state.disorder_inputs.get('Occupation') in occupation_options else 0,
                                      key="occupation_disorder")
                st.session_state.disorder_inputs['Occupation'] = occupation
                
                sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 
                                       st.session_state.disorder_inputs.get('Sleep Duration', 7.0), 0.1,
                                       key="sleep_duration_disorder")
                st.session_state.disorder_inputs['Sleep Duration'] = sleep_duration
                
                quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 
                                         st.session_state.disorder_inputs.get('Quality of Sleep', 7),
                                         key="quality_of_sleep_disorder")
                st.session_state.disorder_inputs['Quality of Sleep'] = quality_of_sleep
            
            with col2:
                # Health metrics
                physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 120, 
                                          st.session_state.disorder_inputs.get('Physical Activity Level', 30),
                                          key="physical_activity_disorder")
                st.session_state.disorder_inputs['Physical Activity Level'] = physical_activity
                
                stress_level = st.slider("Stress Level (1-10)", 1, 10, 
                                     st.session_state.disorder_inputs.get('Stress Level', 5),
                                     key="stress_level_disorder")
                st.session_state.disorder_inputs['Stress Level'] = stress_level
                
                bmi_options = ["Normal", "Overweight", "Obese", "Underweight"]
                bmi_category = st.selectbox("BMI Category", bmi_options,
                                        index=bmi_options.index(st.session_state.disorder_inputs.get('BMI Category', "Normal")),
                                        key="bmi_category_disorder")
                st.session_state.disorder_inputs['BMI Category'] = bmi_category
                
                # Blood pressure input (systolic/diastolic)
                col2a, col2b = st.columns(2)
                with col2a:
                    systolic = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=200, 
                                          value=st.session_state.disorder_inputs.get('Systolic BP', 120),
                                          key="systolic_bp_disorder")
                    st.session_state.disorder_inputs['Systolic BP'] = systolic
                
                with col2b:
                    diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=120, 
                                           value=st.session_state.disorder_inputs.get('Diastolic BP', 80),
                                           key="diastolic_bp_disorder")
                    st.session_state.disorder_inputs['Diastolic BP'] = diastolic
                
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, 
                                        value=st.session_state.disorder_inputs.get('Heart Rate', 75),
                                        key="heart_rate_disorder")
                st.session_state.disorder_inputs['Heart Rate'] = heart_rate
                
                daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, 
                                         value=st.session_state.disorder_inputs.get('Daily Steps', 7000),
                                         key="daily_steps_disorder")
                st.session_state.disorder_inputs['Daily Steps'] = daily_steps
        
        with st.expander("Physiological Sleep Metrics", expanded=True):
            st.write("Enter your physiological sleep metrics for sleep quality assessment:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Collect input for quality model based on your preprocessing
                hrv = st.number_input("Heart Rate Variability (ms)", min_value=40.0, max_value=120.0, 
                                  value=st.session_state.quality_inputs.get('Heart_Rate_Variability', 70.0),
                                  key="hrv_quality")
                st.session_state.quality_inputs['Heart_Rate_Variability'] = hrv
                
                body_temp = st.number_input("Body Temperature (°C)", min_value=36.0, max_value=38.0, 
                                       value=st.session_state.quality_inputs.get('Body_Temperature', 36.8), 
                                       format="%.1f",
                                       key="body_temp_quality")
                st.session_state.quality_inputs['Body_Temperature'] = body_temp
                
                movement = st.number_input("Movement During Sleep (index)", min_value=0.0, max_value=5.0, 
                                      value=st.session_state.quality_inputs.get('Movement_During_Sleep', 1.5),
                                      key="movement_quality")
                st.session_state.quality_inputs['Movement_During_Sleep'] = movement
                
                sleep_duration_quality = st.slider("Sleep Duration (hours)", 3.0, 12.0, 
                                           st.session_state.quality_inputs.get('Sleep_Duration', 7.0), 0.1,
                                           key="sleep_duration_quality")
                st.session_state.quality_inputs['Sleep_Duration'] = sleep_duration_quality
            
            with col2:
                caffeine = st.number_input("Caffeine Intake (mg)", min_value=0, max_value=500, 
                                       value=st.session_state.quality_inputs.get('Caffeine_Intake_mg', 100),
                                       key="caffeine_quality")
                st.session_state.quality_inputs['Caffeine_Intake_mg'] = caffeine
                
                stress_quality = st.slider("Stress Level (1-10)", 1, 10, 
                               st.session_state.quality_inputs.get('Stress_Level', 5),
                               key="stress_quality")
                st.session_state.quality_inputs['Stress_Level'] = stress_quality
                
                bedtime_consistency = st.slider("Bedtime Consistency (0-1)", 0.0, 1.0, 
                                           st.session_state.quality_inputs.get('Bedtime_Consistency', 0.7), 0.01,
                                           key="bedtime_consistency_quality")
                st.session_state.quality_inputs['Bedtime_Consistency'] = bedtime_consistency
                
                light_exposure = st.number_input("Light Exposure (hours)", min_value=0.0, max_value=16.0, 
                                            value=st.session_state.quality_inputs.get('Light_Exposure_hours', 8.0),
                                            key="light_exposure_quality")
                st.session_state.quality_inputs['Light_Exposure_hours'] = light_exposure
        
        with st.expander("Sleep Audio Recording", expanded=True):
            st.write("Upload an audio recording of your sleep to analyze snoring patterns:")
            
            # File uploader
            uploaded_file = st.file_uploader("Choose a sleep audio file", type=["wav", "mp3"], key="audio_uploader")
            
            if uploaded_file is not None:
                # Display audio player
                st.audio(uploaded_file, format="audio/wav")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Store path in session state
                st.session_state.temp_audio_path = temp_path
        
        # Analysis buttons - centralized
        st.write("### Run Analysis")
        col1, col2, col3 = st.columns(3)
        
        # Button to process sleep disorder prediction
        with col1:
            # Button to process sleep disorder prediction
            if st.button("Analyze Sleep Disorder Risk", key="analyze_disorder_btn"):
                # Prepare data for model - provide data in the format expected by the predict_sleep_disorder function
                input_dict = {
                    'Age': st.session_state.disorder_inputs['Age'],
                    'Gender': st.session_state.disorder_inputs['Gender'],  # Keep as string
                    'Sleep Duration': st.session_state.disorder_inputs['Sleep Duration'],
                    'Quality of Sleep': st.session_state.disorder_inputs['Quality of Sleep'],
                    'Physical Activity Level': st.session_state.disorder_inputs['Physical Activity Level'],
                    'Stress Level': st.session_state.disorder_inputs['Stress Level'],
                    'BMI Category': st.session_state.disorder_inputs['BMI Category'],  # Keep as string
                    'Heart Rate': st.session_state.disorder_inputs['Heart Rate'],
                    'Daily Steps': st.session_state.disorder_inputs['Daily Steps'],
                    'Systolic BP': st.session_state.disorder_inputs['Systolic BP'],
                    'Diastolic BP': st.session_state.disorder_inputs['Diastolic BP'],
                    'Occupation': st.session_state.disorder_inputs['Occupation']  # Keep as string
                }
                
                # Convert to DataFrame
                df = pd.DataFrame([input_dict])
                
                # Make prediction
                with st.spinner("Analyzing sleep disorder risk..."):
                    st.session_state.disorder_results = analyzer.predict_sleep_disorder(df)      
                    if st.session_state.disorder_results:
                        st.success("Sleep disorder analysis complete! Check the Results tab.")
        
        # Button to process sleep quality prediction
        with col2:
            if st.button("Analyze Sleep Quality", key="analyze_quality_btn"):
                # Prepare data for model - using only the features needed for the quality model
                quality_df = pd.DataFrame([{
                                        'Heart_Rate_Variability': st.session_state.quality_inputs['Heart_Rate_Variability'],
                    'Body_Temperature': st.session_state.quality_inputs['Body_Temperature'],
                    'Movement_During_Sleep': st.session_state.quality_inputs['Movement_During_Sleep'],
                    'Sleep_Duration': st.session_state.quality_inputs['Sleep_Duration'],
                    'Caffeine_Intake_mg': st.session_state.quality_inputs['Caffeine_Intake_mg'],
                    'Stress_Level': st.session_state.quality_inputs['Stress_Level'],
                    'Bedtime_Consistency': st.session_state.quality_inputs['Bedtime_Consistency'],
                    'Light_Exposure_hours': st.session_state.quality_inputs['Light_Exposure_hours']
                }])
                
                # Make prediction
                with st.spinner("Analyzing sleep quality..."):
                    st.session_state.quality_results = analyzer.predict_sleep_quality(quality_df)
                    
                    if st.session_state.quality_results:
                        st.success("Sleep quality analysis complete! Check the Results tab.")
        
        # Button for audio analysis
        with col3:
            if 'temp_audio_path' in st.session_state:
                if st.button("Analyze Audio Recording", key="analyze_audio_btn"):
                    with st.spinner("Analyzing sleep audio... This may take several minutes for long recordings."):
                        # Process audio
                        st.session_state.audio_results = analyzer.analyze_audio(st.session_state.temp_audio_path)
                        
                        if st.session_state.audio_results:
                            # Create visualization
                            fig = analyzer.create_visualization(st.session_state.temp_audio_path, st.session_state.audio_results)
                            if fig:
                                st.session_state.audio_fig = fig
                            
                            st.success("Audio analysis complete! Check the Results tab.")
            else:
                st.button("Analyze Audio Recording", disabled=True, key="analyze_audio_btn_disabled")
                st.info("Upload an audio file first")
        
        # Run all analyses at once
        if st.button("Complete Full Analysis", key="full_analysis_btn", type="primary"):
            # Check if audio file is available
            has_audio = 'temp_audio_path' in st.session_state
            
            with st.spinner("Running comprehensive sleep analysis..."):
                # 1. Sleep disorder prediction
                            
                disorder_input = pd.DataFrame([{
                    'Age': st.session_state.disorder_inputs['Age'],
                    'Gender': st.session_state.disorder_inputs['Gender'],  # Keep as string
                    'Sleep Duration': st.session_state.disorder_inputs['Sleep Duration'],
                    'Quality of Sleep': st.session_state.disorder_inputs['Quality of Sleep'],
                    'Physical Activity Level': st.session_state.disorder_inputs['Physical Activity Level'],
                    'Stress Level': st.session_state.disorder_inputs['Stress Level'],
                    'BMI Category': st.session_state.disorder_inputs['BMI Category'],  # Keep as string
                    'Heart Rate': st.session_state.disorder_inputs['Heart Rate'],
                    'Daily Steps': st.session_state.disorder_inputs['Daily Steps'],
                    'Systolic BP': st.session_state.disorder_inputs['Systolic BP'],
                    'Diastolic BP': st.session_state.disorder_inputs['Diastolic BP'],
                    'Occupation': st.session_state.disorder_inputs['Occupation']  # Keep as string
                }])
                
                st.session_state.disorder_results = analyzer.predict_sleep_disorder(disorder_input)
                
                # 2. Sleep quality prediction
                quality_input = pd.DataFrame([{
                    'Heart_Rate_Variability': st.session_state.quality_inputs['Heart_Rate_Variability'],
                    'Body_Temperature': st.session_state.quality_inputs['Body_Temperature'],
                    'Movement_During_Sleep': st.session_state.quality_inputs['Movement_During_Sleep'],
                    'Sleep_Duration': st.session_state.quality_inputs['Sleep_Duration'],
                    'Caffeine_Intake_mg': st.session_state.quality_inputs['Caffeine_Intake_mg'],
                    'Stress_Level': st.session_state.quality_inputs['Stress_Level'],
                    'Bedtime_Consistency': st.session_state.quality_inputs['Bedtime_Consistency'],
                    'Light_Exposure_hours': st.session_state.quality_inputs['Light_Exposure_hours']
                }])
                
                st.session_state.quality_results = analyzer.predict_sleep_quality(quality_input)
                
                # 3. Audio analysis if audio file available
                if has_audio:
                    st.session_state.audio_results = analyzer.analyze_audio(st.session_state.temp_audio_path)
                    
                    if st.session_state.audio_results:
                        # Create visualization
                        fig = analyzer.create_visualization(st.session_state.temp_audio_path, st.session_state.audio_results)
                        if fig:
                            st.session_state.audio_fig = fig
                
                # Generate comprehensive suggestions
                st.session_state.combined_suggestions = analyzer.generate_suggestions(
                    st.session_state.audio_results,
                    st.session_state.disorder_results,
                    st.session_state.quality_results
                )
                
                st.success("Comprehensive analysis complete! Check the Results tab.")
    
    #############
    # Tab 2: Results & Recommendations
    #############
    with tab2:
        st.header("Comprehensive Sleep Analysis Results")
        
        # Check if any analysis has been done
        if not st.session_state.disorder_results and not st.session_state.quality_results and not st.session_state.audio_results:
            st.info("No analysis results available yet. Complete at least one analysis in the Input Data tab.")
            return
        
        # Generate comprehensive suggestions if they don't exist but we have results
        if (st.session_state.disorder_results or st.session_state.quality_results or st.session_state.audio_results) and not st.session_state.combined_suggestions:
            st.session_state.combined_suggestions = analyzer.generate_suggestions(
                st.session_state.audio_results,
                st.session_state.disorder_results,
                st.session_state.quality_results
            )
        
        # Display results in expandable sections
        with st.expander("Sleep Disorder Analysis", expanded=True):
            if st.session_state.disorder_results:
                disorder = st.session_state.disorder_results['prediction']
                
                # Display disorder prediction prominently
                if disorder == "None":
                    st.success("### No Sleep Disorder Detected")
                elif disorder == "Insomnia":
                    st.warning("### Insomnia Detected")
                elif disorder == "Sleep Apnea":
                    st.error("### Sleep Apnea Detected")
                
                # Display probabilities if available
                if 'probabilities' in st.session_state.disorder_results and st.session_state.disorder_results['probabilities'] is not None:
                    # Get class probabilities - this is now a dictionary
                    probs_dict = st.session_state.disorder_results['probabilities']
                    
                    # Convert the dict to a DataFrame directly
                    prob_data = pd.DataFrame(list(probs_dict.items()), columns=['Disorder', 'Probability'])
                    
                    st.write("#### Disorder Probabilities")
                    chart = st.bar_chart(prob_data.set_index('Disorder'))
                
                # Show input summary
                st.write("#### Key Risk Factors")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sleep Duration", f"{st.session_state.disorder_inputs['Sleep Duration']:.1f} hrs")
                    st.metric("BMI Category", st.session_state.disorder_inputs['BMI Category'])
                with col2:
                    st.metric("Quality of Sleep", f"{st.session_state.disorder_inputs['Quality of Sleep']}/10")
                    st.metric("Physical Activity", f"{st.session_state.disorder_inputs['Physical Activity Level']} min/day")
                with col3:
                    st.metric("Stress Level", f"{st.session_state.disorder_inputs['Stress Level']}/10")
                    st.metric("Heart Rate", f"{st.session_state.disorder_inputs['Heart Rate']} bpm")
            else:
                st.info("Sleep disorder analysis not completed.")
        
        with st.expander("Sleep Quality Analysis", expanded=True):
            if st.session_state.quality_results:
                quality_score = st.session_state.quality_results['score']
                
                # Display result prominently
                if quality_score <= 3:
                    st.error(f"### Poor Sleep Quality: {quality_score:.1f}/10")
                elif quality_score <= 7:
                    st.warning(f"### Moderate Sleep Quality: {quality_score:.1f}/10")
                else:
                    st.success(f"### Good Sleep Quality: {quality_score:.1f}/10")
                
                # Display quality score
                st.slider("Sleep Quality Score", 1, 10, int(quality_score), disabled=True, key="result_quality_score")
                
                # Show key metrics
                st.write("#### Key Physiological Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Heart Rate Variability", f"{st.session_state.quality_inputs['Heart_Rate_Variability']:.1f} ms")
                with col2:
                    st.metric("Body Temperature", f"{st.session_state.quality_inputs['Body_Temperature']:.1f} °C")
                with col3:
                    st.metric("Movement", f"{st.session_state.quality_inputs['Movement_During_Sleep']:.2f} index")
                with col4:
                    st.metric("Sleep Duration", f"{st.session_state.quality_inputs['Sleep_Duration']:.1f} hrs")
            else:
                st.info("Sleep quality analysis not completed.")
        
        with st.expander("Snoring Analysis", expanded=True):
            if st.session_state.audio_results:
                results = st.session_state.audio_results
                
                # Display snoring severity prominently
                if results['snoring_percentage'] < 10:
                    st.success(f"### Minimal Snoring: {results['snoring_percentage']:.1f}%")
                elif results['snoring_percentage'] < 30:
                    st.info(f"### Mild Snoring: {results['snoring_percentage']:.1f}%")
                elif results['snoring_percentage'] < 50:
                    st.warning(f"### Moderate Snoring: {results['snoring_percentage']:.1f}%")
                else:
                    st.error(f"### Severe Snoring: {results['snoring_percentage']:.1f}%")
                
                # Show key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Recording Duration", f"{results['total_duration_minutes']:.1f} min")
                with col2:
                    st.metric("Snoring Duration", f"{results['total_snoring_duration'] / 60:.1f} min")
                with col3:
                    st.metric("Snoring Episodes", f"{results['snoring_segments_count']}")
                
                # Display visualization if available
                if 'audio_fig' in st.session_state:
                    st.write("#### Snoring Pattern Visualization")
                    st.pyplot(st.session_state.audio_fig)
            else:
                st.info("Audio analysis not completed.")
        
        # Display comprehensive recommendations
        with st.expander("Personalized Sleep Recommendations", expanded=True):
            if st.session_state.combined_suggestions:
                # Display specific recommendations
                st.markdown("### Specific Recommendations")
                for suggestion in st.session_state.combined_suggestions['specific_suggestions']:
                    st.write(f"• {suggestion}")
                
                # Display general recommendations
                st.markdown("### General Sleep Hygiene Tips")
                for suggestion in st.session_state.combined_suggestions['general_suggestions']:
                    st.write(f"• {suggestion}")
            else:
                st.info("Complete at least one analysis to get personalized recommendations.")
        
        # Clear all results button
        if st.button("Clear All Results", key="clear_results_btn"):
            # Reset all session state values
            st.session_state.disorder_inputs = {}
            st.session_state.quality_inputs = {}
            st.session_state.audio_results = None
            st.session_state.disorder_results = None
            st.session_state.quality_results = None
            st.session_state.combined_suggestions = None
            
            # Clean up temp files
            if 'temp_audio_path' in st.session_state and os.path.exists(st.session_state.temp_audio_path):
                try:
                    os.unlink(st.session_state.temp_audio_path)
                except:
                    pass
            
            st.experimental_rerun()


if __name__ == "__main__":
    main()