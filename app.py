import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
import json

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
# Ensure GOOGLE_API_KEY is set in your environment variables
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model_gemini = genai.GenerativeModel('gemini-pro')

# LLM Context and Prompt Template (as defined in previous steps)
system_instruction = (
    "You are a knowledgeable wine expert. Your primary role is to provide concise, informative, and relevant answers about red 'Vinho Verde' wine, its characteristics, and the machine learning model used to predict its quality. "
    "Always refer to the provided context when answering questions. If a question cannot be answered from the context, state that explicitly."
    "Focus on clarity and accuracy, avoiding jargon where possible, but explaining it when necessary."
)

llm_context_data = {
    'project_objective': "To predict the quality of red 'Vinho Verde' wine based on its physicochemical properties using machine learning models.",
    'feature_names': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
    'dataset_descriptive_statistics': {
        'fixed acidity': {'count': 1359.0, 'mean': 8.310596026490067, 'std': 1.736989807532466, 'min': 4.6, '25%': 7.1, '50%': 7.9, '75%': 9.2, 'max': 15.9},
        'volatile acidity': {'count': 1359.0, 'mean': 0.5294775570272259, 'std': 0.18303131761907185, 'min': 0.12, '25%': 0.39, '50%': 0.52, '75%': 0.64, 'max': 1.58},
        'citric acid': {'count': 1359.0, 'mean': 0.2723325974981604, 'std': 0.1955365445504639, 'min': 0.0, '25%': 0.09, '50%': 0.26, '75%': 0.43, 'max': 1.0},
        'residual sugar': {'count': 1359.0, 'mean': 2.5234009094922737, 'std': 1.352467556942476, 'min': 0.9, '25%': 1.9, '50%': 2.2, '75%': 2.6, 'max': 15.5},
        'chlorides': {'count': 1359.0, 'mean': 0.08812435614422369, 'std': 0.04938634177435191, 'min': 0.012, '25%': 0.07, '50%': 0.08, '75%': 0.091, 'max': 0.611},
        'free sulfur dioxide': {'count': 1359.0, 'mean': 15.893156732892126, 'std': 10.447343513337965, 'min': 1.0, '25%': 7.0, '50%': 14.0, '75%': 21.0, 'max': 72.0},
        'total sulfur dioxide': {'count': 1359.0, 'mean': 46.24282560706402, 'std': 32.29033320646626, 'min': 6.0, '25%': 22.0, '50%': 38.0, '75%': 62.0, 'max': 289.0},
        'density': {'count': 1359.0, 'mean': 0.9967098528329654, 'std': 0.001864197262261646, 'min': 0.99, '25%': 0.9956, '50%': 0.9967, '75%': 0.9978,
        'max': 1.00369},
        'pH': {'count': 1359.0, 'mean': 3.309786607799853, 'std': 0.15598418042571216, 'min': 2.74, '25%': 3.21, '50%': 3.31, '75%': 3.4,
        'max': 4.01},
        'sulphates': {'count': 1359.0, 'mean': 0.6587417218543046, 'std': 0.17435555620942474, 'min': 0.33, '25%': 0.55, '50%': 0.62, '75%': 0.73,
        'max': 2.0},
        'alcohol': {'count': 1359.0, 'mean': 10.432376747608535, 'std': 1.0829330691509935, 'min': 8.4, '25%': 9.5, '50%': 10.2, '75%': 11.1,
        'max': 14.9}
    },
    'model_performance': {
        'Baseline (Linear Regression)': {'Test RMSE': np.float64(0.656512760797608), 'R2 Score': np.float64(0.3915360499058188)},
        'Random Forest': {'Test RMSE': np.float64(0.6193438178474718), 'R2 Score': np.float64(0.45848296601599026)},
        'XGBoost': {'Test RMSE': np.float64(0.6378441442688663), 'R2 Score': np.float64(0.42564862966537476)}
    },
    'champion_model': 'Random Forest'
}

# Convert numpy types in llm_context_data to native Python types for JSON serialization
def convert_numpy_to_python(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_python(elem) for elem in obj]
    return obj

llm_context_data_serializable = convert_numpy_to_python(llm_context_data)

prompt_template = f"""{system_instruction}

Here is the relevant context:
{{llm_context}}

Based on this context, please answer the following question:
{{user_question}}
"""

# Function to query Gemini
def query_gemini(user_question: str):
    full_context = json.dumps(llm_context_data_serializable, indent=2)
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
