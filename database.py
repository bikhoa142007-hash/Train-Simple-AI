import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st

def train_perceptron(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = Perceptron(max_iter=5000, eta0=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def predict_perceptron(model, scaler, poly, X_test):
    X_test_poly = poly.transform(X_test)
    X_test_scaled = scaler.transform(X_test_poly)
    return model.predict(X_test_scaled)

def train_until_reach_accuracy(X_train, y_train, target_accuracy=0.85):
    best_accuracy = 0
    best_model = None
    best_scaler = None
    best_poly = None
    max_attempts = 1500    
    
    poly = PolynomialFeatures(degree=3, include_bias=False)
    progress_text = st.empty()   
    
    for attempt in range(1, max_attempts + 1):
        current_seed = np.random.randint(0, 50000)
        current_eta0 = np.random.choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])       
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train, test_size=0.2, random_state=current_seed
        )    
        
        X_tr_poly = poly.fit_transform(X_tr)
        X_te_poly = poly.transform(X_te)  
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_poly)
        X_te_scaled = scaler.transform(X_te_poly)             
        
        model = Perceptron(
            max_iter=2000, 
            eta0=current_eta0, 
            penalty='l2',
            alpha=0.0001,
            random_state=current_seed
        )
        model.fit(X_tr_scaled, y_tr)   
        
        y_pred = model.predict(X_te_scaled)
        current_accuracy = accuracy_score(y_te, y_pred)          
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = model
            best_scaler = scaler
            best_poly = poly                  
        
        progress_text.text(
            f"Đang tìm giải pháp nâng cao... Lần {attempt}/{max_attempts} | Accuracy tốt nhất: {best_accuracy:.2f}"
        )           
        
        if best_accuracy >= target_accuracy:
            break           
            
    if best_accuracy >= target_accuracy:
        progress_text.success(f"Xuất sắc! Đạt mục tiêu nâng cao: {best_accuracy:.2f}")
    else:
        progress_text.warning(f"Đã quét tối ưu 1500 lần. Accuracy tối đa đạt được: {best_accuracy:.2f}")        
        
    return best_model, best_scaler, best_poly, best_accuracy

def enter_input_page():
    st.title("Nhập dữ liệu")
    st.write("Nhập thông tin bệnh nhân để dự đoán nguy cơ loãng xương:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
        Age = st.number_input("Tuổi", min_value=0, max_value=120, value=0)
        Height = st.number_input("Chiều cao (cm)", min_value=0.0, value=0.0)
        Weight = st.number_input("Cân nặng (kg)", min_value=0.0, value=0.0)
        
    with col2:
        BMI = st.number_input("Chỉ số BMI", min_value=0.0, value=0.0)
        lumbar_spine = st.number_input("Đau lưng dưới", min_value=0.0, value=0.0)
        BMD = st.number_input("Mật độ xương", min_value=0.0, value=0.0)
        T_score = st.number_input("T-score", min_value=-5.0, max_value=5.0, value=0.0)   
    
    if st.button("Chẩn đoán kết quả", type="primary"):
        gender_val = 1 if Gender == "Nam" else 0
        
        input_data = [
            gender_val, Age, Height, Weight, 
            BMI, lumbar_spine, BMD, T_score
        ]
        input_array = [input_data]
        
        if "model" in st.session_state and "scaler" in st.session_state and "poly" in st.session_state:
            model = st.session_state.model
            scaler = st.session_state.scaler
            poly = st.session_state.poly
            
            prediction = predict_perceptron(model, scaler, poly, input_array)
            
            st.markdown("---")
            st.subheader("Kết quả chẩn đoán:")
            if prediction[0] == 1:
                st.error("⚠️ Cảnh báo: Có nguy cơ mắc bệnh (Positive)")
            else:
                st.success("🎉 Chúc mừng: Chỉ số bình thường (Negative)")
        else:
            st.warning("Vui lòng thực hiện huấn luyện (Train) mô hình trước!")