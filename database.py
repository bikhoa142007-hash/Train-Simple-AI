import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import hashlib
# Hàm huấn luyện mô hình Perceptron
def train_perceptron(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = Perceptron(max_iter=1000, eta0=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler
def predict_perceptron(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    return model.predict(X_test_scaled)
def train_until_reach_accuracy(X_train, y_train, X_test, y_test, target_accuracy=0.9):
    best_accuracy = 0
    attempt = 0
    max_attempts = 100
    progress_text = st.empty()
    while best_accuracy < target_accuracy and attempt < max_attempts:
        attempt += 1
        model, scaler = train_perceptron(X_train, y_train)
        y_pred = predict_perceptron(model, scaler, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_scaler = scaler    
        progress_text.text(f"Đang thử lần {attempt}... Accuracy hiện tại: {accuracy:.2f}")
        if best_accuracy >= target_accuracy:
            break     
    return best_model, best_scaler, best_accuracy
# Hàm lưu dữ USERNAME và PASSWORD vào file JSON
DB_FILE = "users.json"
def load_data():
    if not os.path.exists(DB_FILE):
        return {"users": []}
    with open(DB_FILE, "r") as f:
        return json.load(f)
def save_data(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
# Giao diện đăng nhập, đăng ký và quên mật khẩu
def login_page():
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"
    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None
    data = load_data()
    # MÀN HÌNH ĐĂNG NHẬP
    if st.session_state.auth_page == "login":
        st.subheader("Đăng Nhập")
        username = st.text_input("Tên đăng nhập", key="login_user")
        password = st.text_input("Mật khẩu", type="password", key="login_pass")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Đăng nhập"):
                hashed = hash_password(password)
                user = next((u for u in data["users"] if u["username"] == username and u["password"] == hashed), None)
                if user:
                    st.session_state.logged_in_user = username
                    st.success(f"Xin chào {username}!")
                    st.rerun()
                else:
                    st.error("Sai tài khoản hoặc mật khẩu!")
        with col2:
            if st.button("Chưa có tài khoản?"):
                st.session_state.auth_page = "register"
                st.rerun()          
        if st.button("Quên mật khẩu?", variant="secondary"):
            st.session_state.auth_page = "forgot_password"
            st.rerun()
    # MÀN HÌNH ĐĂNG KÝ
    elif st.session_state.auth_page == "register":
        st.subheader("Đăng Ký Tài Khoản")
        new_user = st.text_input("Tên đăng nhập mới", key="reg_user")
        email = st.text_input("Email", key="reg_email")
        new_pass = st.text_input("Mật khẩu", type="password", key="reg_pass")
        confirm_pass = st.text_input("Xác nhận mật khẩu", type="password", key="reg_confirm")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Tạo tài khoản", use_container_width=True):
                if not new_user or not email or not new_pass:
                    st.warning("Vui lòng điền đầy đủ thông tin.")
                elif new_pass != confirm_pass:
                    st.error("Mật khẩu xác nhận không trùng khớp.")
                elif any(u["username"] == new_user for u in data["users"]):
                    st.error("Tên đăng nhập đã tồn tại!")
                else:
                    user_obj = {"username": new_user, "email": email, "password": hash_password(new_pass)}
                    data["users"].append(user_obj)
                    save_data(data)
                    st.success("Đăng ký thành công!")
                    st.session_state.auth_page = "login"
                    st.rerun()
        with col2:
            if st.button("Quay lại Đăng nhập", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()
    # MÀN HÌNH QUÊN MẬT KHẨU
    elif st.session_state.auth_page == "forgot_password":
        st.subheader("Khôi Phục Mật Khẩu")
        username = st.text_input("Tên đăng nhập", key="forgot_user")
        email = st.text_input("Email đã đăng ký", key="forgot_email")
        new_pass = st.text_input("Mật khẩu mới", type="password", key="forgot_pass")        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Đổi mật khẩu", use_container_width=True):
                user = next((u for u in data["users"] if u["username"] == username and u["email"] == email), None)
                if user:
                    user["password"] = hash_password(new_pass)
                    save_data(data)
                    st.success("Đổi mật khẩu thành công!")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error("Thông tin không chính xác!")
        with col2:
            if st.button("Hủy bỏ", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()
# Hàm bổ trợ app
def greeting_page():
    st.title("Chào mừng bạn đến với ứng dụng!")
    st.write(f"Xin chào, {st.session_state.username}! Bạn đã đăng nhập thành công.")
def enter_input_page():
    st.title("Nhập dữ liệu")
    Gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
    Age = st.number_input("Tuổi", min_value=0, max_value=120)
    Height = st.number_input("Chiều cao (cm)", min_value=0.0)
    Weight = st.number_input("Cân nặng (kg)", min_value=0.0)
    BMI = st.number_input("Chỉ số BMI", min_value=0.0)
    lumbar_spine = st.number_input("Đau lưng dưới", min_value=0.0)
    Femoral_Neck = st.number_input("Đau cổ đùi ", min_value=0.0)
    T_score = st.number_input("Điểm T-score", min_value=0.0)
    LDL_C = st.number_input("LDL-C (mg/dL)", min_value=0.0)
    Ca = st.number_input("Canxi (mg/dL)", min_value=0.0)
    P = st.number_input("Phospho (mg/dL)", min_value=0.0)
    Mg = st.number_input("Magie (mg/dL)", min_value=0.0)
    Alanine_Aminotransferase = st.number_input("Alanine Aminotransferase (U/L)", min_value=0.0)
    Blood_Urea_Nitrogen = st.number_input("Blood Urea Nitrogen (mg/dL)", min_value=0.0)
    if st.button("Chẩn đoán kết quả"):
        gender_val = 1 if Gender == "Nam" else 2
        input_data = [
        gender_val, Age, Height, Weight, BMI, 
        lumbar_spine, Femoral_Neck, T_score, LDL_C, 
        Ca, P, Mg, Alanine_Aminotransferase, Blood_Urea_Nitrogen
        ]
        input_array = [input_data] 
        if "model" in st.session_state and "scaler" in st.session_state:
            model = st.session_state.model
            scaler = st.session_state.scaler
            prediction = predict_perceptron(model, scaler, input_array)
            st.subheader("Kết quả chẩn đoán:")
            if prediction[0] == 1:
                st.error("Cảnh báo: Có nguy cơ mắc bệnh (Positive)")
            else:
                st.success("Chúc mừng: Chỉ số bình thường (Negative)")
        else:
            st.warning("Vui lòng thực hiện huấn luyện (Train) mô hình trước!")
def enter_csv_page():
    st.title("Nhập dữ liệu từ file CSV")
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    return uploaded_file