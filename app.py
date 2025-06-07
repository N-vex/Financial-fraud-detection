import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load trained model
model = joblib.load(r"C:\Users\HP\OneDrive\Nick final year document\Main Detect\ml-streamlit_app\model\Financial_fraud_model.pkl")

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# App title
st.title("💳 Financial Fraud Detection App")

# Tabs for two functionalities
tab1, tab2 = st.tabs(["📊 Dataset Analysis", "🧪 Manual Transaction Test"])

# ======================= TAB 1 =========================
with tab1:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Overview")
        st.write(data.head())

        st.write(f"Number of Rows: {data.shape[0]}")
        st.write(f"Number of Columns: {data.shape[1]}")

        st.subheader("Class Distribution (Legit vs Fraud)")
        fig, ax = plt.subplots()
        sns.countplot(x='Class', data=data, ax=ax, palette='Set2')
        ax.set_xticklabels(['Legit', 'Fraud'])
        st.pyplot(fig)

        st.subheader("Amount Distribution by Class")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Class', y='Amount', data=data, ax=ax2, palette='Set2')
        ax2.set_xticklabels(['Legit', 'Fraud'])
        st.pyplot(fig2)

        # ========= Preprocessing for Prediction =========
        data_for_pred = data.copy()

        if 'Amount' in data_for_pred.columns:
            sc = StandardScaler()
            data_for_pred['Amount'] = sc.fit_transform(pd.DataFrame(data_for_pred['Amount']))
        if 'Time' in data_for_pred.columns:
            data_for_pred = data_for_pred.drop('Time', axis=1)

        X = data_for_pred.drop('Class', axis=1)
        y = data_for_pred['Class']

        # ========= Predict with Pre-trained Model =========
        prediction = model.predict(X)

        # ========= Show Detected Frauds =========
        fraud_data = data[prediction == 1]  # Use original data for output
        st.subheader("✅ Detected Fraud Transactions")
        st.write(fraud_data)

        st.subheader("💰 Fraud Transaction Amounts")
        st.write(fraud_data['Amount'])

        # ========= Download Button =========
        csv = fraud_data.to_csv(index=False)
        st.download_button("⬇️ Download Fraud Transactions", data=csv, file_name="fraud_transactions.csv", mime="text/csv")

        # ========= Optional: SMOTE Visualization =========
        st.subheader("📈 Class Distribution After SMOTE (for training purposes)")

        # Apply SMOTE for visualization only
        X_res, y_res = SMOTE().fit_resample(X, y)
        smote_df = pd.DataFrame(X_res, columns=X.columns)
        smote_df['Class'] = y_res

        fig3, ax3 = plt.subplots()
        sns.countplot(x='Class', data=smote_df, ax=ax3, palette='coolwarm')
        ax3.set_xticklabels(['Legit', 'Fraud'])
        st.pyplot(fig3)

        st.info("This SMOTE-balanced class distribution is only for visualization/training, not used for prediction.")


# ======================= TAB 2 =========================
with tab2:
    st.subheader("Enter Transaction Details Manually")
    
    # Define features (excluding 'Time' and 'Class')
    feature_columns = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]

    user_input = []
    col1, col2 = st.columns(2)

    for i, col in enumerate(feature_columns):
        if i % 2 == 0:
            val = col1.number_input(f"{col}", value=0.0, format="%.6f")
        else:
            val = col2.number_input(f"{col}", value=0.0, format="%.6f")
        user_input.append(val)

    if st.button("Predict Transaction"):
        input_df = pd.DataFrame([user_input], columns=feature_columns)
        prediction = model.predict(input_df)

        if prediction[0] == 0:
            st.success("✅ The transaction is LEGIT.")
        else:
            st.error("🚨 The transaction is FRAUDULENT!")
