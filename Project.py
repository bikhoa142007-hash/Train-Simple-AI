import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from database import train_perceptron, predict_perceptron, train_until_reach_accuracy, login_page
# Hàm bổ trợ 
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
# Hàm chính
def main():
    login_page()
    file_da_chon = enter_csv_page()
    if file_da_chon:
        df = pd.read_csv(file_da_chon)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model, scaler, accuracy = train_until_reach_accuracy(X_train, y_train, X_test, y_test, target_accuracy=0.9)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.accuracy = accuracy
        greeting_page()
        enter_input_page()
if __name__ == "__main__":
    main()


