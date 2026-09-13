import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# ===================================
# LOAD DATASET
# ===================================

print("\nLoading dataset...\n")

df = pd.read_csv("data/creditcard.csv")

# ===================================
# BASIC DATASET INFO
# ===================================

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== FRAUD vs NORMAL ==========\n")
print(df["Class"].value_counts())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ===================================
# FRAUD PERCENTAGE
# ===================================

fraud_percentage = (df["Class"].value_counts()[1] / len(df)) * 100

print(f"\nFraud Transactions Percentage: {fraud_percentage:.4f}%")

# ===================================
# VISUALIZATION BEFORE SMOTE
# ===================================



plt.show()

# ===================================
# FEATURE SCALING
# ===================================

print("\nScaling Time and Amount columns...\n")

scaler = StandardScaler()

df["Time"] = scaler.fit_transform(df[["Time"]])
df["Amount"] = scaler.fit_transform(df[["Amount"]])

print("\n========== SCALED DATA ==========\n")

print(df[["Time", "Amount"]].head())

# ===================================
# FEATURES & TARGET
# ===================================

X = df.drop("Class", axis=1)

y = df["Class"]

print("\n========== ORIGINAL CLASS COUNT ==========\n")

print(y.value_counts())

# ===================================
# SMOTE - IMBALANCE HANDLING
# ===================================

print("\nApplying SMOTE...\n")

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\n========== AFTER SMOTE ==========\n")

print(pd.Series(y_resampled).value_counts())

# ===================================
# VISUALIZATION AFTER SMOTE
# ===================================

plt.figure(figsize=(6, 4))

sns.countplot(x=y_resampled)

plt.title("After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()

# ===================================
# TRAIN TEST SPLIT
# ===================================

print("\nSplitting training and testing data...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ===================================
# LOGISTIC REGRESSION MODEL
# ===================================

print("\n===================================")
print("TRAINING LOGISTIC REGRESSION MODEL")
print("===================================\n")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

# Prediction

lr_pred = lr_model.predict(X_test)

# Accuracy

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n========== LOGISTIC REGRESSION ACCURACY ==========\n")

print(f"Accuracy: {lr_accuracy * 100:.2f}%")

# Classification Report

print("\n========== LOGISTIC REGRESSION REPORT ==========\n")

print(classification_report(y_test, lr_pred))

# Confusion Matrix

lr_cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    lr_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ===================================
# LINEAR SVM MODEL
# ===================================

print("\n===================================")
print("TRAINING LINEAR SVM MODEL")
print("===================================\n")

svm_model = LinearSVC(
    max_iter=5000,
    random_state=42
)

svm_model.fit(X_train, y_train)

# Prediction

svm_pred = svm_model.predict(X_test)

# Accuracy

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n========== SVM ACCURACY ==========\n")

print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

# Classification Report

print("\n========== SVM CLASSIFICATION REPORT ==========\n")

print(classification_report(y_test, svm_pred))

# Confusion Matrix

svm_cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    svm_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Confusion Matrix - Linear SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ===================================
# K-MEANS ANOMALY CLUSTERING
# ===================================

print("\n===================================")
print("K-MEANS CLUSTERING")
print("===================================\n")

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# Add cluster column

df["Cluster"] = clusters

print("\n========== CLUSTER COUNTS ==========\n")

print(df["Cluster"].value_counts())

# ===================================
# CLUSTER VISUALIZATION
# ===================================

plt.figure(figsize=(6, 4))

sns.countplot(x="Cluster", data=df)

plt.title("K-Means Clusters")
plt.xlabel("Cluster")
plt.ylabel("Count")

plt.show()

# ===================================
# FINAL MESSAGE
# ===================================

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================\n")