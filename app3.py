import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\HP\OneDrive\Nick final year document\Main Detect\ml-streamlit_app\model\fraud_model.pkl") 
    return model

# App Title
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("\U0001F512 Financial Fraud Detection - Real-World Testing")

# Sidebar for file upload
st.sidebar.header("Step 1: Upload Your Dataset")
file = st.sidebar.file_uploader("Upload a CSV file for prediction", type=["csv"])

if file:
    df = pd.read_csv(file)
    original_df = df.copy()

    st.subheader("\U0001F4C8 Dataset Preview")
    st.write(df.head())

    # Show class distribution if Label exists
    if "Label" in df.columns:
        label_counts = df["Label"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(label_counts, labels=["Legit", "Fraud"], autopct="%1.2f%%", startangle=140, colors=["#3cb371", "#ff6347"])
        ax.set_title("Transaction Class Distribution")
        st.pyplot(fig)

    st.sidebar.header("Step 2: Predict Fraud")
    if st.sidebar.button("Run Prediction"):
        model = load_model()

        # Separate features and preprocess
        X = df.drop(columns=["Label"], errors='ignore')
        y = df["Label"] if "Label" in df.columns else None

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        df["Predicted_Label"] = y_pred
        df["Fraud_Probability"] = y_proba

        st.subheader("\U0001F4CA Post-Prediction Dataset Preview")
        st.write(df.head())

        # Show confusion matrix if true labels available
        if y is not None:
            cm = confusion_matrix(y, y_pred)
            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            st.text("Classification Report")
            st.code(classification_report(y, y_pred))
            st.metric("ROC AUC", round(roc_auc_score(y, y_proba), 3))

        # Show predicted fraud only
        st.subheader("\U0001F50D Detected Fraudulent Transactions")
        frauds = df[df["Predicted_Label"] == 1]
        st.write(frauds)

        # Download button
        csv = frauds.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="\U0001F4E5 Download Detected Frauds",
            data=csv,
            file_name='fraudulent_transactions.csv',
            mime='text/csv',
        )

# Tips Section
st.sidebar.markdown("---")
st.sidebar.markdown("\U0001F4A1 **Tips:**")
st.sidebar.markdown("- Ensure your dataset has the same structure as the training data.")
st.sidebar.markdown("- For better prediction, include columns like `Time`, `Amount`, and anonymized features.")
