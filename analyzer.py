import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def analyze_data(df):
    desc = df.describe()
    nulls = df.isnull().sum()
    return {"Description": desc, "Missing Values": nulls}

def visualize_data(df):
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        st.write(f"Histogram of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    if len(numeric_cols) > 1:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
