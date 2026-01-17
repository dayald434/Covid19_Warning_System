"""
COVID-19 Early Warning System - Streamlit Interface
Predicts required public health intervention level 7 days in advance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

# Suppress sklearn feature name warnings (model was trained without feature names)
warnings.filterwarnings('ignore', message='X has feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Page configuration
st.set_page_config(
    page_title="COVID-19 Warning System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model and metadata"""
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'trained' / 'best_covid_warning_model.pkl'
        artifact = joblib.load(model_path)
        return artifact
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main app
def main():
    # Load model
    artifact = load_model()
    
    if artifact is None:
        st.error("❌ Model not found. Please train the model first by running: `python scripts/run_pipeline.py`")
        return
    
    model = artifact['model']
    feature_columns = artifact['feature_names']  # New model structure uses 'feature_names'
    target_classes = artifact['target_classes']
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select Page", ["🔮 Prediction", "📊 Batch", "ℹ️ About"], label_visibility="collapsed")
        
        st.markdown("---")
        st.subheader("COVID-19 Warning System")
        st.markdown("**Predict required public health actions 7 days in advance**")
        
        st.markdown("""
        - Analyzes current epidemiological indicators
        - Returns 4-level warning classification
        - Provides policy recommendations
        - Gives policymakers advance notice
        """)
        
        st.warning("⚠️ **Important:** Not a replacement for epidemiological expertise. Use as one input in decision-making.")
        
        st.markdown("---")
        st.subheader("Model Information:")
        st.markdown("""
        - Trained on historical COVID-19 data
        - 8 algorithms compared
        - Best model selected by composite score
        - 80/20 time-based train/test split
        """)
        
        st.markdown("---")
        st.subheader("Performance:")
        st.markdown("""
        - Accuracy: 75-85 percent
        - Critical Recall: 80-95 percent
        """)
    
    # Main content
    if page == "🔮 Prediction":
        page_prediction(model, feature_columns)
    elif page == "📊 Batch":
        page_batch(model, feature_columns)
    else:
        page_about()

def page_prediction(model, feature_columns):
    """Prediction interface"""
    st.title("🦠 COVID-19 Early Warning System")
    st.markdown("### Predict required public health actions 7 days in advance")
    st.markdown("This system analyzes current COVID-19 epidemiological indicators to forecast the level of intervention needed in the next 7 days, giving policymakers time to respond.")
    
    st.markdown("---")
    
    # Create input form
    st.markdown("## 📊 Enter Current Epidemiological Data")
    
    # Collect 16 user inputs
    user_input = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Growth Dynamics")
        user_input['Growth_Rate'] = st.slider("Growth Rate (percent per day)", -1.0, 2.0, 0.10, 0.01,
                                               help="📈 Percentage change in daily cases")
        user_input['Doubling_Time'] = st.number_input("Doubling Time (days)", 1.0, 1000.0, 60.0, 1.0,
                                                       help="📊 Days for cases to double")
        user_input['Acceleration'] = st.slider("Acceleration (/day)", -1.0, 1.0, 0.0, 0.01,
                                                help="🚀 Rate of change in growth speed")
        user_input['Death_Growth'] = st.slider("Death Growth Rate (percent per day)", -1.0, 2.0, 0.02, 0.01,
                                                help="📉 Daily percentage change in deaths")
        
        st.subheader("7-Day Averages")
        user_input['Cases_7d_MA'] = st.number_input("Cases 7-Day MA", 0.0, 100000.0, 4800.0, 100.0,
                                                     help="📊 7-day moving average of daily cases")
        user_input['Deaths_7d_MA'] = st.number_input("Deaths 7-Day MA", 0.0, 5000.0, 45.0, 5.0,
                                                      help="📊 7-day moving average of daily deaths")
        
        st.subheader("Additional Metrics")
        user_input['Deaths_per_100k'] = st.number_input("Deaths per 100k", 0.0, 200.0, 10.0, 1.0,
                                                         help="💀 Deaths per 100k population")
    
    with col2:
        st.subheader("Case Burden")
        user_input['Daily_Cases'] = st.number_input("Daily Cases", 0, 100000, 5000, 100,
                                                     help="📈 New cases reported today")
        user_input['Daily_Deaths'] = st.number_input("Daily Deaths", 0, 5000, 50, 5,
                                                      help="💀 New deaths reported today")
        user_input['Cases_per_100k'] = st.number_input("Cases per 100k Population", 0.0, 5000.0, 30.0, 1.0,
                                                        help="📍 Cases per 100k population")
        
        st.subheader("Temporal Context")
        user_input['Days_Since_100'] = st.number_input("Days Since 100 Cases", 0, 2000, 100, 5,
                                                        help="📅 Days since 100th case")
        user_input['Days_Since_Start'] = st.number_input("Days Since Start", 0, 2000, 200, 10,
                                                          help="📅 Days since outbreak start")
        
        # Day of week selector
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        from datetime import datetime
        current_day = datetime.now().weekday()
        selected_day = st.selectbox("Day of Week", options=days, index=current_day,
                                     help="📅 Current day of week")
        user_input['DayOfWeek'] = days.index(selected_day)
        
        user_input['IsWeekend'] = st.selectbox("Is Weekend?", options=['No', 'Yes'], 
                                                index=1 if user_input['DayOfWeek'] >= 5 else 0,
                                                help="📅 Weekend indicator")
        user_input['IsWeekend'] = 1 if user_input['IsWeekend'] == 'Yes' else 0
        
        st.subheader("Severity Indicators")
        user_input['CFR'] = st.slider("Case Fatality Rate (percent)", 0.0, 15.0, 1.0, 0.1,
                                       help="⚰️ Deaths / Cases * 100")
        user_input['Active_Cases'] = st.number_input("Active Cases", 0, 10000000, 100000, 1000,
                                                      help="📊 Current active cases")
    
    # Auto-calculate remaining 18 features from the 16 user inputs
    input_data = {}
    
    # Direct user inputs (16 features)
    input_data['Growth_Rate'] = user_input['Growth_Rate']
    input_data['Doubling_Time'] = user_input['Doubling_Time']
    input_data['Acceleration'] = user_input['Acceleration']
    input_data['Death_Growth'] = user_input['Death_Growth']
    input_data['Cases_7d_MA'] = user_input['Cases_7d_MA']
    input_data['Deaths_7d_MA'] = user_input['Deaths_7d_MA']
    input_data['Deaths_per_100k'] = user_input['Deaths_per_100k']
    input_data['Daily_Cases'] = user_input['Daily_Cases']
    input_data['Daily_Deaths'] = user_input['Daily_Deaths']
    input_data['Cases_per_100k'] = user_input['Cases_per_100k']
    input_data['Days_Since_100'] = user_input['Days_Since_100']
    input_data['Days_Since_Start'] = user_input['Days_Since_Start']
    input_data['DayOfWeek'] = user_input['DayOfWeek']
    input_data['IsWeekend'] = user_input['IsWeekend']
    input_data['CFR'] = user_input['CFR']
    input_data['Active_Cases'] = user_input['Active_Cases']
    
    # Auto-calculate remaining 18 features
    # Calculate population from Deaths_per_100k and Daily_Deaths
    input_data['Population'] = int((user_input['Daily_Deaths'] * 50 / max(user_input['Deaths_per_100k'], 0.01)) * 100000)
    input_data['Population'] = max(input_data['Population'], 1000000)  # Minimum 1M population
    
    # Calculate cumulative from daily and days
    input_data['Confirmed'] = int(user_input['Daily_Cases'] * user_input['Days_Since_100'])
    input_data['Deaths'] = int(user_input['Daily_Deaths'] * user_input['Days_Since_100'])
    input_data['Recovered'] = int(input_data['Confirmed'] * (1 - user_input['CFR']/100) * 0.95)
    input_data['Daily_Recovered'] = int(user_input['Daily_Cases'] * 0.95)
    
    # Temporal features
    now = datetime.now()
    input_data['Month'] = now.month
    input_data['Quarter'] = (now.month - 1) // 3 + 1
    input_data['Year'] = now.year
    
    # Logarithmic transforms
    input_data['Log_Cases'] = np.log(max(input_data['Confirmed'], 1))
    input_data['Log_Deaths'] = np.log(max(input_data['Deaths'], 1))
    
    # Recovery and death ratios
    input_data['Recovery_Rate'] = (input_data['Recovered'] / max(input_data['Confirmed'], 1)) * 100
    input_data['Death_to_Case_Ratio'] = input_data['Deaths'] / max(input_data['Confirmed'], 1)
    
    # Policy context (defaults)
    input_data['Is_Lockdown'] = 0
    input_data['Is_Post_Vaccine'] = 1
    
    # Future projections (7 days ahead)
    input_data['Growth_Rate_future7d'] = user_input['Growth_Rate'] + user_input['Acceleration'] * 7
    input_data['Cases_per_100k_future7d'] = user_input['Cases_per_100k'] * (1 + user_input['Growth_Rate'] * 7)
    input_data['Doubling_Time_future7d'] = user_input['Doubling_Time'] * 0.9 if user_input['Acceleration'] > 0 else user_input['Doubling_Time'] * 1.1
    input_data['CFR_future7d'] = user_input['CFR'] * 1.05
    
    # Make prediction
    st.markdown("---")
    if st.button("🔮 Predict Warning Level", type="primary"):
        try:
            # Create dataframe with correct feature order
            input_df = pd.DataFrame([input_data])[feature_columns]
            
            # Predict directly (new model has no preprocessing pipeline)
            pred_raw = model.predict(input_df)[0]
            
            # Get class labels and convert prediction to string
            class_labels = model.classes_
            
            # Convert prediction to label string (ensure it's a Python string)
            if isinstance(pred_raw, (int, np.integer)):
                prediction = str(class_labels[pred_raw])
            else:
                prediction = str(pred_raw)
            
            # Normalize prediction format (remove numpy prefix if exists)
            prediction = prediction.strip()
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)[0]
                confidence = probabilities.max()
                prob_dict = {str(label): prob for label, prob in zip(class_labels, probabilities)}
            else:
                confidence = None
                prob_dict = None
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results (7 Days Ahead)")
            
            # Color coding - make more flexible
            color_map = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MODERATE': '🟡',
                'LOW': '🟢'
            }
            
            # Find appropriate color based on prediction content
            color = '⚪'
            for key, emoji in color_map.items():
                if key in prediction.upper():
                    color = emoji
                    break
            
            # Create descriptions based on prediction
            description = ""
            if 'CRITICAL' in prediction.upper():
                description = 'Immediate intervention required - implement strict emergency measures to prevent healthcare system collapse'
            elif 'HIGH' in prediction.upper():
                description = 'Strong measures needed - enhanced social distancing and movement restrictions recommended'
            elif 'MODERATE' in prediction.upper():
                description = 'Enhanced monitoring required - implement moderate restrictions and increase testing capacity'
            elif 'LOW' in prediction.upper():
                description = 'Routine surveillance - maintain standard public health protocols and monitoring'
            else:
                description = f'Warning level: {prediction}'
            
            # Recommendations based on prediction
            actions = []
            if 'CRITICAL' in prediction.upper():
                actions = [
                    "⚠️ Implement immediate emergency measures",
                    "🏥 Prepare healthcare system for surge capacity",
                    "📢 Issue public health emergency alert",
                    "🚫 Close non-essential businesses and schools",
                    "💉 Accelerate vaccination and testing programs"
                ]
            elif 'HIGH' in prediction.upper():
                actions = [
                    "📊 Enhanced social distancing protocols required",
                    "😷 Mandatory mask mandates in public spaces",
                    "👥 Limit large gatherings (max 10-20 people)",
                    "🏢 Encourage work-from-home policies",
                    "🧪 Increase testing capacity by 50%"
                ]
            elif 'MODERATE' in prediction.upper():
                actions = [
                    "📈 Closely monitor case trends daily",
                    "🏥 Ensure healthcare resources are adequate",
                    "😷 Recommend masks in crowded indoor spaces",
                    "✅ Maintain current restrictions",
                    "📣 Public awareness campaigns"
                ]
            elif 'LOW' in prediction.upper():
                actions = [
                    "✅ Continue routine surveillance",
                    "📊 Monitor data weekly",
                    "💉 Maintain vaccination programs",
                    "🏥 Standard healthcare protocols",
                    "📢 Regular public health updates"
                ]
            
            # Display prediction with big visual indicator
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Show exact prediction label
                st.markdown(f"### {color} {prediction}")
                if confidence:
                    st.metric("Model Confidence", f"{confidence:.1%}")
                    
                    # Confidence interpretation
                    if confidence >= 0.8:
                        st.success("✅ High confidence prediction")
                    elif confidence >= 0.6:
                        st.info("ℹ️ Moderate confidence")
                    else:
                        st.warning("⚠️ Low confidence - monitor closely")
            
            with col2:
                st.markdown("#### Situation Assessment")
                st.info(description)
            
            # Recommended Actions
            if actions:
                st.markdown("---")
                st.markdown("### 📋 Recommended Actions")
                
                for action in actions:
                    st.markdown(f"- {action}")
            
            # Show all probabilities with visual bars
            if prob_dict:
                st.markdown("---")
                st.markdown("### 📊 Detailed Risk Assessment")
                st.markdown("*Probability of each warning level in 7 days*")
                
                prob_df = pd.DataFrame({
                    'Warning Level': list(prob_dict.keys()),
                    'Probability': list(prob_dict.values())
                })
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                for idx, row in prob_df.iterrows():
                    level = str(row['Warning Level'])
                    prob = row['Probability']
                    
                    # Find color for this level
                    level_color = '⚪'
                    for key, emoji in color_map.items():
                        if key in level.upper():
                            level_color = emoji
                            break
                    
                    # Color-coded progress bar with exact label
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(prob, text=f"{level_color} {level}")
                    with col_b:
                        st.markdown(f"**{prob:.1%}**")
            
            # Key metrics summary
            st.markdown("---")
            st.markdown("### 📈 Input Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Growth Rate", f"{input_data['Growth_Rate']*100:.2f}%")
            with col2:
                st.metric("Cases per 100k", f"{input_data['Cases_per_100k']:.1f}")
            with col3:
                st.metric("CFR (percent)", f"{input_data['CFR']:.1f}")
            with col4:
                st.metric("Deaths per 100k", f"{input_data['Deaths_per_100k']:.1f}")
            
            # Model information
            with st.expander("ℹ️ About This Prediction"):
                st.markdown("""
                **Model:** Random Forest Classifier (100 decision trees)
                
                **How it works:**
                - Analyzes 34 epidemiological features from current data
                - Predicts intervention level needed 7 days from now
                - Trained on 8,066 samples from 337,185+ historical COVID-19 records
                - Optimized to detect critical situations early (91.7% Critical Recall)
                
                **Prediction horizon:** 7 days ahead
                
                **Confidence level:** Higher confidence (greater than 80 percent) indicates the model is very certain about the prediction based on historical patterns.
                
                **Important:** This is a decision support tool. Always combine with expert epidemiological judgment and local context.
                """)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            
            # Debug info
            with st.expander("🔍 Debug Information"):
                st.write("Expected features:", feature_columns)
                st.write("Provided features:", list(input_data.keys()))
                st.write("Error details:", str(e))
                st.write("Error type:", type(e).__name__)

def page_batch(model, feature_columns):
    """Batch prediction interface"""
    st.title("📊 Batch Predictions")
    st.markdown("### Upload CSV file for multiple predictions")
    
    st.info("Upload a CSV file with the same features used in training. The system will predict warning levels for all rows.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {len(df)} records")
            
            st.subheader("Preview of uploaded data")
            st.dataframe(df.head(10))
            
            if st.button("🔮 Run Batch Predictions", type="primary"):
                # Check if required features exist
                missing_features = [f for f in feature_columns if f not in df.columns]
                
                if missing_features:
                    st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                else:
                    # Prepare data
                    X = df[feature_columns].copy()
                    
                    # Predictions (new model has no preprocessing pipeline)
                    predictions = model.predict(X)
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)
                        confidences = probabilities.max(axis=1)
                    else:
                        confidences = None
                    
                    # Add results to dataframe
                    df['Predicted_Warning_Level'] = predictions
                    if confidences is not None:
                        df['Confidence'] = confidences
                    
                    st.success(f"✅ Predictions complete!")
                    
                    # Show results
                    st.subheader("Results")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name='covid_predictions.csv',
                        mime='text/csv'
                    )
                    
                    # Summary statistics
                    st.subheader("Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    warning_counts = pd.Series(predictions).value_counts()
                    
                    with col1:
                        st.metric("🔴 Critical", warning_counts.get('CRITICAL_EMERGENCY', 0))
                    with col2:
                        st.metric("🟠 High", warning_counts.get('HIGH_ALERT', 0))
                    with col3:
                        st.metric("🟡 Moderate", warning_counts.get('MODERATE_MEASURES', 0))
                    with col4:
                        st.metric("🟢 Low", warning_counts.get('LOW_MONITORING', 0))
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")



def page_about():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## COVID-19 Early Warning System
    
    ### Purpose
    This machine learning system predicts the required public health intervention level 
    **7 days in advance** to give policymakers time to respond to emerging COVID-19 threats.
    
    ### How It Works
    
    1. **Input Current Metrics**: Enter today's COVID-19 situation metrics
    2. **ML Prediction**: Random Forest model analyzes 34 epidemiological features
    3. **7-Day Forecast**: System predicts intervention level needed in 7 days
    4. **Actionable Output**: Clear warning level with confidence score
    
    ### Risk Scoring System
    
    The system calculates a composite risk score based on:
    - **Growth Rate (40 percent)**: Speed of case increase
    - **Case Burden (30 percent)**: Cases per 100,000 population
    - **Doubling Time (20 percent)**: How fast cases double
    - **Case Fatality Rate (10 percent)**: Disease severity
    
    ### Warning Levels
    
    | Level | Risk Score | Action Required |
    |-------|-----------|-----------------|
    | 🔴 CRITICAL | ≥10 | Immediate lockdown |
    | 🟠 HIGH | 6-9 | Strong restrictions |
    | 🟡 MODERATE | 3-5 | Enhanced monitoring |
    | 🟢 LOW | 0-2 | Routine surveillance |
    
    ### Model Performance
    
    - **Dataset**: 337,185 total records → 10,071 valid samples (2020-2023)
    - **Training**: 8,066 samples (80 percent split)
    - **Testing**: 2,005 samples (20 percent split)
    - **Countries**: 201 countries and regions
    - **Algorithm**: Random Forest (100 trees)
    - **Optimization**: Focused on detecting critical situations
    - **Critical Recall**: 91.7 percent (detects 92 percent of critical situations)
    
    ### Data Sources
    
    Training data from Johns Hopkins University CSSE COVID-19 repository:
    - Daily confirmed cases
    - Daily deaths
    - Daily recovered cases
    - Geographic information
    
    ### Limitations
    
    ⚠️ **Important Notes**:
    - Model trained on historical data (may not capture new variants)
    - Requires accurate input data
    - Should complement, not replace, expert judgment
    - Performance depends on data quality
    - Not validated for real-time deployment
    
    ### Technical Stack
    
    - **ML Framework**: scikit-learn
    - **Model**: Random Forest Classifier
    - **Interface**: Streamlit
    - **Data Processing**: pandas, numpy
    
    ---
    
    **Developed as an early warning decision support tool for public health officials**
    """)

if __name__ == "__main__":
    main()
