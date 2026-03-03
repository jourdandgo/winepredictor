import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
import json

# No longer need 'userdata' directly in app.py for flexible deployment

# Page Config
st.set_page_config(page_title="Vinho Verde Quality Predictor", page_icon="🍷", layout="wide")

# Custom CSS for "Wine Vibe"
st.markdown("""
    <style>
    .main { background-color: #fdfcfc; }
    .stButton>button { background-color: #722f37; color: white; border-radius: 5px; }
    .prediction-box { padding: 20px; border-radius: 10px; border: 1px solid #722f37; background-color: #fff4f4; text-align: center; }
    .chat-box { padding: 15px; border-radius: 8px; background-color: #e6e6e6; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Configure Gemini API
# Attempt to get API key from Streamlit secrets (for Streamlit Cloud deployment)
API_KEY = st.secrets.get('GOOGLE_API_KEY')
# Fallback to environment variable (for local testing or other deployments)
if not API_KEY:
    API_KEY = os.environ.get('GOOGLE_API_KEY')

if not API_KEY:
    st.error("Gemini API key not found. Please set 'GOOGLE_API_KEY' in Streamlit secrets or as an environment variable.")
    st.stop()

genai.configure(api_key=API_KEY)

# Logic to find a suitable Gemini model
model_gemini = None
try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m)

    if not available_models:
        st.error("No Gemini models supporting 'generateContent' found. Please check your API key and regional availability.")
    else:
        preferred_model_name = None
        # Prioritize 'flash' models
        for model_info in available_models:
            if 'flash' in model_info.name:
                preferred_model_name = model_info.name
                break
        
        # Fallback to 'gemini-pro' if no 'flash' model is found
        if not preferred_model_name:
            for model_info in available_models:
                if 'gemini-pro' in model_info.name:
                    preferred_model_name = model_info.name
                    break
        
        # Fallback to the first available model if neither 'flash' nor 'gemini-pro' is found
        if not preferred_model_name:
            preferred_model_name = available_models[0].name

        model_gemini = genai.GenerativeModel(preferred_model_name)
except Exception as e:
    st.error(f"Error configuring Gemini model: {e}")


# LLM Context and Prompt Template (as defined in previous steps)
system_instruction = (
    "You are a knowledgeable wine expert. Your primary role is to provide concise, informative, and relevant answers about red 'Vinho Verde' wine, its characteristics, and the machine learning model used to predict its quality. "
    "Always refer to the provided context when answering questions. If a question cannot be answered from the context, state that explicitly."
    "Focus on clarity and accuracy, avoiding jargon where possible, but explaining it when necessary."
)

# Load the LLM context from the JSON file
try:
    with open('llm_context.json', 'r') as f:
        llm_context_data = json.load(f)
except FileNotFoundError:
    st.error("LLM context file 'llm_context.json' not found. Please run the context gathering steps first.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding 'llm_context.json'. Please ensure it's a valid JSON file.")
    st.stop()

prompt_template = f"""{system_instruction}

Here is the relevant context:
{{llm_context}}

Based on this context, please answer the following question:
{{user_question}}
"""

# Function to query Gemini
def query_gemini(user_question: str):
    if model_gemini is None:
        return "LLM is not configured properly. Please check the API key and model availability."
    
    full_context = json.dumps(llm_context_data, indent=2) # Use loaded context
    prompt = prompt_template.format(llm_context=full_context, user_question=user_question)
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error querying Gemini: {e}"


# Load Model and Metadata
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('champion_wine_model.joblib')
        features = joblib.load('features.joblib')
        return model, features
    except:
        return None, None

model, features = load_assets()

# Sidebar: Input Panel
st.sidebar.header("🍷 Chemical Properties")
st.sidebar.markdown("Adjust the physicochemical traits below to predict wine quality.")

def user_input_features():
    data = {}
    data['fixed acidity'] = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 8.3)
    data['volatile acidity'] = st.sidebar.slider('Volatile Acidity', 0.1, 1.6, 0.5)
    data['citric acid'] = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.27)
    data['residual sugar'] = st.sidebar.slider('Residual Sugar', 0.9, 15.5, 2.5)
    data['chlorides'] = st.sidebar.slider('Chlorides', 0.01, 0.6, 0.08)
    data['free sulfur dioxide'] = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 15.0)
    data['total sulfur dioxide'] = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.0)
    data['density'] = st.sidebar.slider('Density', 0.990, 1.004, 0.996)
    data['pH'] = st.sidebar.slider('pH', 2.7, 4.0, 3.3)
    data['sulphates'] = st.sidebar.slider('Sulphates', 0.3, 2.0, 0.6)
    data['alcohol'] = st.sidebar.slider('Alcohol (%)', 8.0, 15.0, 10.4)
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Content
st.title("🍷 Professional Wine Quality Predictor")
st.markdown("""
    This interactive application leverages machine learning to predict the quality of red 'Vinho Verde' wine based on its physicochemical properties.
    By adjusting various chemical parameters in the sidebar, you can observe how different characteristics influence the predicted quality score.
    The underlying model has been trained on a dataset of red wines, aiming to provide insights into what makes a good quality wine.
    The predicted quality score is on a scale from 0 (very poor) to 10 (excellent).
    """)

if model is None:
    st.error("Model files not found. Please run 'train_model.py' first.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Summary")
        st.write(input_df)
        prediction = model.predict(input_df)[0]
        st.markdown("---")
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color: #722f37;'>Predicted Quality Score</h3>
                <h1 style='font-size: 60px; color: #722f37;'>{prediction:.2f}</h1>
                <p>Scale: 0 (Poor) - 10 (Excellent)</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Quality Analytics")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            gauge = {'axis': {'range': [None, 10]}, 'bar': {'color': "#722f37"}, 'steps': [
                {'range': [0, 4], 'color': 'lightgray'},
                {'range': [4, 7], 'color': 'gray'},
                {'range': [7, 10], 'color': '#722f37'}
            ]},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Model Performance Overview")
    st.markdown("Our prediction system compared three distinct machine learning models:")
    st.markdown("""
    *   **Baseline (Linear Regression)**: Achieved a Test RMSE of **0.6565** and R2 of **0.3915**.
    *   **Random Forest**: Showed superior performance with a Test RMSE of **0.6193** and R2 of **0.4585**.
    *   **XGBoost**: Performed with a Test RMSE of **0.6378** and R2 of **0.4256**.

    Based on these evaluations, the **Random Forest** model was selected as the champion due to its lowest Test RMSE, indicating the best predictive accuracy on unseen data.
    """)

    st.subheader("🧬 Decision Drivers")
    st.markdown("The chart below illustrates the relative importance of each chemical property in the model's decision-making process, ordered from most to least influential.")
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=True) # Sort ascending for plotly bar chart to show highest at top
        fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#722f37'],
                         title="Feature Importance in Predicting Wine Quality")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance is not available for the selected model type.")


# LLM Interaction Section
st.markdown("""
    <hr style="height:2px;border:none;color:#333;background-color:#333;" />
    """, unsafe_allow_html=True)
st.subheader("🗣️ Chat with our Wine Expert (powered by Gemini)")
st.markdown("Ask our AI wine expert anything about this dataset, the models, or wine quality in general!")

user_question = st.text_input("Your Question:", "What are the most important features for predicting wine quality?")

if st.button("Ask Gemini"):
    if user_question:
        with st.spinner('Thinking...'):
            gemini_response = query_gemini(user_question)
            st.markdown(f"<div class='chat-box'><b>Gemini:</b> {gemini_response}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a question to ask Gemini.")
