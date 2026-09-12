import pandas as pd
import streamlit as st

import database as db


st.set_page_config(page_title="Osteoporosis Risk Demo", page_icon="🦴")


@st.cache_resource
def train_model(data_path: str):
    """Train once per Streamlit session instead of retraining on every interaction."""
    return db.train_and_evaluate(data_path)


def main():
    st.title("Dự đoán nguy cơ loãng xương")
    st.caption("Dự án học tập Machine Learning — không thay thế chẩn đoán y khoa.")

    try:
        artifacts = train_model("du_lieu.csv")
    except Exception as exc:
        st.error(f"Không thể huấn luyện mô hình: {exc}")
        st.stop()

    metrics = artifacts["metrics"]
    st.subheader("Đánh giá trên tập kiểm thử cố định")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    cols[3].metric("F1-score", f"{metrics['f1']:.3f}")

    st.caption(
        f"Mô hình được chọn bằng 5-fold cross-validation trên tập train: "
        f"{artifacts['model_name']}. Tập test chỉ được dùng để báo cáo cuối cùng."
    )

    st.subheader("Nhập thông tin để thử mô hình")
    input_df = db.patient_input_form()

    if input_df is not None:
        prediction = artifacts["model"].predict(input_df)[0]
        st.markdown("---")
        if prediction == 1:
            st.error("Mô hình dự đoán: Có nguy cơ (Positive)")
        else:
            st.success("Mô hình dự đoán: Không có nguy cơ (Negative)")

        st.info(
            "Kết quả chỉ phục vụ minh họa kỹ thuật. Hãy tham khảo nhân viên y tế "
            "để được đánh giá và chẩn đoán phù hợp."
        )


if __name__ == "__main__":
    main()
