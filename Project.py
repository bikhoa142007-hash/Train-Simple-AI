import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import database as db
# App
def main():
    db.login_page()
    file_da_chon = db.enter_csv_page()
    if file_da_chon:
        df = pd.read_csv(file_da_chon)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model, scaler, accuracy = db.train_until_reach_accuracy(X_train, y_train, X_test, y_test, target_accuracy=0.9)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.accuracy = accuracy
if __name__ == "__main__":
    main()


