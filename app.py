import streamlit as st
import pandas as pd
from analyzer import analyze_data, visualize_data
from ml_model import train_and_predict

st.title("📊 Smart Data Analyzer with ML")

uploaded = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("📈 Summary Statistics")
    results = analyze_data(df)
    st.write(results["Description"])
    st.write("Missing values per column:")
    st.write(results["Missing Values"])

    st.subheader("📊 Data Visualization")
    visualize_data(df)

    st.subheader("🤖 Machine Learning Prediction")
    if st.checkbox("Enable ML Prediction"):
        target = st.selectbox("Select target column", options=df.columns)
        test_size = st.slider("Test size (%)", 10, 50, 20)
        metric, model_info = train_and_predict(df, target, test_size / 100.0)
        st.write(f"**Task**: {model_info['task_type']}")
        st.write(f"**Model**: {model_info['model_name']}")
        st.write(f"**Performance ({metric['name']})**: {metric['value']:.4f}")
