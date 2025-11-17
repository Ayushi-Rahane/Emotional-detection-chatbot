# transformer_confusion.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------
# Load Test Data
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
test_df  = load_data(os.path.join(base_dir, "../data/test.txt"))

# ---------------------------
# Load Transformer Model
# ---------------------------
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

label_map = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

# ---------------------------
# Predict Function
# ---------------------------
def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    preds = torch.argmax(outputs, dim=1).numpy()
    return preds

# ---------------------------
# Make Predictions
# ---------------------------
y_true = test_df["label"].tolist()
y_pred_idx = predict(test_df["text"].tolist())
y_pred = [label_map[i] for i in y_pred_idx]

# ---------------------------
# Align labels between model and dataset
# ---------------------------
dataset_labels = sorted(list(set(y_true)))
model_labels = [label_map[i] for i in range(len(label_map))]

# intersection of labels
eval_labels = [label for label in model_labels if label in dataset_labels]

# warn if missing
missing_labels = set(dataset_labels) - set(eval_labels)
if missing_labels:
    print("Warning: These dataset labels are not predicted by the transformer:", missing_labels)

print("\nEvaluating on labels:", eval_labels)

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_true, y_pred, labels=eval_labels)

print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=eval_labels, yticklabels=eval_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – DistilRoBERTa Model (Aligned Labels)")
plt.tight_layout()
plt.show()

# ---------------------------
# Classification Report
# ---------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, labels=eval_labels, digits=3))


# ---------------------------
# Save Transformer Report to Excel
# ---------------------------
report = classification_report(y_true, y_pred, labels=eval_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

output_path = os.path.join(base_dir, "../results/transformer_classification_report.xlsx")
report_df.to_excel(output_path, index=True)

print("Transformer classification report saved to:", output_path)
