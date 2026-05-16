import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import database as db
# App
st.set_page_config(page_title="Perceptron Trainer", page_icon=":bar_chart:", layout="centered")

def main():
    if "logged_in_user" not in st.session_state or st.session_state.logged_in_user is None:
        db.login_page()
    else:
        file_da_chon = db.enter_csv_page()
        if file_da_chon:
            df = pd.read_csv(file_da_chon)
            df = df.dropna()  
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model, scaler, accuracy = db.train_until_reach_accuracy(X_train, y_train, X_test, y_test, target_accuracy=0.9)
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.accuracy = accuracy
if __name__ == "__main__":
    main()
