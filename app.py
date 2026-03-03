import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Vinho Verde Quality Predictor", page_icon="🍷", layout="wide")

# Custom CSS for "Wine Vibe"
st.markdown("""
    <style>
    .main { background-color: #fdfcfc; }
    .stButton>button { background-color: #722f37; color: white; border-radius: 5px; }
    .prediction-box { padding: 20px; border-radius: 10px; border: 1px solid #722f37; background-color: #fff4f4; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

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
st.markdown("This application uses a Machine Learning model to predict red wine quality.")

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
            gauge = {'axis': {'range': [None, 10]}, 'bar': {'color': "#722f37"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧬 Decision Drivers")
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
        fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#722f37'])
        st.plotly_chart(fig_imp, use_container_width=True)
