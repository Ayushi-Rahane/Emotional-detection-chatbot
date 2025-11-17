# baseline_confusion.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------
# Load Data
# ---------------------------
def load_data(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ";" in line:
                text, label = line.strip().split(";", 1)
                texts.append(text)
                labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})

base_dir = os.path.dirname(os.path.abspath(__file__))
train_df = load_data(os.path.join(base_dir, "../data/train.txt"))
test_df  = load_data(os.path.join(base_dir, "../data/test.txt"))

# ---------------------------
# TF-IDF + Logistic Regression
# ---------------------------
print("Training TF-IDF Baseline Model...")

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["text"])
X_test  = vectorizer.transform(test_df["text"])

model = LogisticRegression(max_iter=500)
model.fit(X_train, train_df["label"])

# ---------------------------
# Predictions
# ---------------------------
y_true = test_df["label"]
y_pred = model.predict(X_test)

# ---------------------------
# Confusion Matrix
# ---------------------------
labels = sorted(train_df["label"].unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\nLabels:", labels)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Baseline TF-IDF Model")
plt.tight_layout()
plt.show()

# ---------------------------
# Classification Report
# ---------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, digits=3))


# ---------------------------
# Save Baseline Report to Excel
# ---------------------------
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

output_path = os.path.join(base_dir, "../results/baseline_classification_report.xlsx")
report_df.to_excel(output_path, index=True)

print("Baseline classification report saved to:", output_path)
