import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import database as db

feature_cols = [
    'Gender', 
    'Age', 
    'Height', 
    'Weight', 
    'lumbar spine(L1-L4)', 
    'BMD(Bone Mineral Density)', 
    'T-score'
]
def main():
    st.title("Dự đoán bệnh loãng xương")
    df = pd.read_csv("C:\\Users\\Khoa Bi\\Downloads\\du lieu.csv")
    df = df.dropna()
    X = df[feature_cols]
    y = df['Osteoporosis'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, scaler, poly, accuracy = db.train_until_reach_accuracy(X_train, y_train, target_accuracy=0.85)
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.poly = poly
    st.session_state.accuracy = accuracy
    st.write("Nhập thông tin bệnh nhân để dự đoán nguy cơ loãng xương:")
    db.enter_input_page()
    if __name__ == "__main__":
        main()
