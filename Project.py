import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from database import train_perceptron, predict_perceptron, train_until_reach_accuracy
def load_users():
    try:
       if os.path.exists("users.txt"):
           with open("users.txt", "r", encoding="utf-8") as f:
               users = [line.strip() for line in f.readlines() if line.strip()]
           return users
       else:
           st.error("File 'users.txt' không tồn tại")
           return []
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi đọc file: {e}")
        return []
def login_page():
    st.title("Đăng nhập")
    # Input username
    username = st.text_input("Tên đăng nhập: ")
    if st.button("Đăng nhập"):
        if username:
            users =load_users()
            if username in users:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.session_state.show_greeting = True
                st.session_state.username = username
                st.rerun()
        else:
            st.warning("Vui lòng nhập tên người dùng!")
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

def main():
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_until_reach_accuracy(X_train, y_train, X_test, y_test, target_accuracy=0.9)


