import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv(r'C:\Users\HP\OneDrive\Nick final year document\Main Detect\ml-streamlit_app\df\creditCards.csv')

# Rename for clarity (do it early)
df = df.rename(columns={"Class": "Label"})  # 0 = Legit, 1 = Fraud

# Split features and target
# ---------------------------------------
# Drop 'Time' column (if it's not needed for model training)
# ---------------------------------------
X = df.drop(columns=["Label", "Time"])  # Drop both the target and 'Time' column
y = df["Label"]

# ---------------------------------------
# Hourly Distribution of Fraud
# ---------------------------------------
# Create the 'Hour' feature without using 'Time' in the model
df["Hour"] = (df["Time"] / 3600) % 24  # We keep this for visualization purposes only

# Continue with the rest of the code as it is...


# ---------------------------------------
# 1. First 5 rows
# ---------------------------------------
print("First 5 rows of the dataset:")
print(df.head())

# ---------------------------------------
# 2. Class Distribution (Countplot)
# ---------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Label", data=df, palette=["#3cb371", "#ff6347"])
plt.xticks([0, 1], ["Legit", "Fraud"])
plt.title("Transaction Class Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------------------------------
# 3. Pie Chart of Legit vs Fraud
# ---------------------------------------
label_counts = df["Label"].value_counts()
labels = ["Legit", "Fraud"]
colors = ["#3cb371", "#ff6347"]

plt.figure(figsize=(5, 5))
plt.pie(label_counts, labels=labels, autopct="%1.2f%%", colors=colors, startangle=140)
plt.title("Transaction Distribution: Legit vs Fraud")
plt.show()

# ---------------------------------------
# 4. Hourly Distribution of Fraud
# ---------------------------------------
df["Hour"] = (df["Time"] / 3600) % 24

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Hour", hue="Label", bins=48, palette=colors, alpha=0.6, multiple="stack")
plt.title("Fraud vs Legit Transactions by Hour of Day")
plt.xlabel("Hour (0–24)")
plt.ylabel("Number of Transactions")
plt.legend(title="Type", labels=["Legit", "Fraud"])
plt.tight_layout()
plt.show()

# ---------------------------------------
# Train-test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# ---------------------------------------
# Scale Time and Amount
# ---------------------------------------
# ---------------------------------------
# Scale Amount column only (Time column is dropped)
# ---------------------------------------
scale_columns = ["Amount"]  # Only scale 'Amount' now
preprocessor = ColumnTransformer([
    ("scaler", StandardScaler(), scale_columns)
], remainder='passthrough')

# Build Pipeline (with SMOTE + XGBoost)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", XGBClassifier(
        n_estimators=100,
        scale_pos_weight=15,  # Use imbalance ratio for adjustment
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# Train Model
pipeline.fit(X_train, y_train)

# ---------------------------------------
# Predictions
# ---------------------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ---------------------------------------
# Save Predictions
# ---------------------------------------
X_test_with_preds = X_test.copy()
X_test_with_preds["Actual"] = y_test.values
X_test_with_preds["Predicted"] = y_pred
X_test_with_preds["Fraud_Probability"] = y_proba

X_test_with_preds.to_csv("fraud_predictions.csv", index=False)
joblib.dump(pipeline, filename=r"C:\Users\HP\OneDrive\Nick final year document\Main Detect\ml-streamlit_app\model\fraud_model.pkl")

# ---------------------------------------
# Evaluation Metrics
# ---------------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC:", pr_auc)

# ---------------------------------------
# Confusion Matrix Heatmap
# ---------------------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ---------------------------------------
# XGBoost Feature Importance Plot
# ---------------------------------------
xgb_model = pipeline.named_steps['classifier']
plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Important Features (XGBoost)")
plt.tight_layout()
plt.show()
