"""
Emotion Detection Baseline Model Training Script
Trains a TF-IDF + Logistic Regression baseline model for emotion classification.

Features:
- Text preprocessing with stopword removal
- TF-IDF vectorization
- Logistic Regression classifier
- Comprehensive evaluation metrics
- Model persistence for deployment
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing utilities
try:
    from preprocessing import clean_text
except ImportError:
    # Fallback to simple cleaning if preprocessing module not available
    import nltk
    from nltk.corpus import stopwords
    import string
    
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass
    
    def clean_text(text):
        """Basic text cleaning: lowercase, remove punctuation and stopwords"""
        if not text or not isinstance(text, str):
            return ""
        stop_words = set(stopwords.words('english')) if 'stopwords' in dir() else set()
        text = text.lower()
        text = ''.join([ch for ch in text if ch not in string.punctuation])
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)


def evaluate_model(y_true, y_pred, class_names):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=class_names, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': {
            class_name: {
                'precision': prec,
                'recall': rec,
                'f1': f,
                'support': supp
            }
            for class_name, prec, rec, f, supp in zip(class_names, precision, recall, f1, support)
        }
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted Emotion", fontsize=12, fontweight='bold')
    plt.ylabel("Actual Emotion", fontsize=12, fontweight='bold')
    plt.title("Emotion Detection Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_class_distribution(y, save_path):
    """
    Plot emotion class distribution in dataset.
    
    Args:
        y: Labels array
        save_path: Path to save the figure
    """
    emotion_counts = pd.Series(y).value_counts()
    
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.xlabel("Emotion", fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title("Emotion Distribution in Dataset", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to {save_path}")


def main():
    """Main training function"""
    print("=" * 60)
    print("Emotion Detection Baseline Model Training")
    print("=" * 60)
    
    # Load dataset
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emotion_dataset.csv")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please run convert_to_csv.py first to generate the dataset.")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Remove any rows with missing values
    df = df.dropna()
    print(f"\nDataset shape after removing missing values: {df.shape}")
    
    # Check emotion distribution
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    # Preprocessing
    print("\n" + "=" * 60)
    print("Preprocessing")
    print("=" * 60)
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("\nSample cleaned text:")
    print(df[['text', 'clean_text']].head(5))
    
    # Prepare data
    X = df['clean_text']
    y = df['emotion']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print("\n" + "=" * 60)
    print("Data Split")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Vectorization
    print("\n" + "=" * 60)
    print("Feature Extraction (TF-IDF)")
    print("=" * 60)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,
        max_df=0.95
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_vec.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Training
    print("\n" + "=" * 60)
    print("Model Training")
    print("=" * 60)
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    
    print("Training Logistic Regression model...")
    model.fit(X_train_vec, y_train)
    print("Training completed!")
    
    # Cross-validation
    print("\n" + "=" * 60)
    print("Cross-Validation")
    print("=" * 60)
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Comprehensive metrics
    class_names = sorted(y.unique())
    metrics = evaluate_model(y_test, y_pred, class_names)
    
    print(f"\nMacro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    
    print("\nPer-class metrics:")
    for emotion in class_names:
        m = metrics['per_class'][emotion]
        print(f"\n{emotion}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall: {m['recall']:.4f}")
        print(f"  F1-Score: {m['f1']:.4f}")
        print(f"  Support: {m['support']}")
    
    # Classification report
    print("\n" + "=" * 60)
    print("Detailed Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Visualizations
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        os.path.join(MODEL_DIR, "confusion_matrix.png")
    )
    
    plot_class_distribution(
        y,
        os.path.join(MODEL_DIR, "class_distribution.png")
    )
    
    # Save model
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    joblib.dump(model, os.path.join(MODEL_DIR, "emotion_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    
    print(f"Model saved to: {os.path.join(MODEL_DIR, 'emotion_model.pkl')}")
    print(f"Vectorizer saved to: {os.path.join(MODEL_DIR, 'vectorizer.pkl')}")
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    sample_texts = [
        "I am really happy today!",
        "Everything is going wrong and I feel terrible.",
        "I'm scared for my exam tomorrow.",
        "This is disgusting! I can't believe it.",
        "What a surprise! This is amazing!"
    ]
    
    sample_clean = [clean_text(t) for t in sample_texts]
    sample_vec = vectorizer.transform(sample_clean)
    sample_preds = model.predict(sample_vec)
    sample_probs = model.predict_proba(sample_vec)
    
    for text, pred, probs in zip(sample_texts, sample_preds, sample_probs):
        max_prob = max(probs)
        print(f"\nText: '{text}'")
        print(f"Predicted: {pred} (confidence: {max_prob:.4f})")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
