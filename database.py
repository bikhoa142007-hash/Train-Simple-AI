import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# Hàm chính để chạy toàn bộ quy trình
