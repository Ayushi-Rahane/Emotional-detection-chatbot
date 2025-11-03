"""
Emotion Detection Model Module
Uses transformer-based model (DistilRoBERTa) for emotion classification,
combined with sentiment analysis and Markov chain modeling for emotion transitions.

Key Features:
- Transformer-based emotion classification (7 emotions)
- Sentiment analysis using TextBlob
- Emotion transition tracking (Markov chain model)
- K-means clustering for emotion embedding visualization
- PCA for dimensionality reduction
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import torch
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

# Model Configuration
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Initialize model components (loaded once at module import)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Set to evaluation mode
    logger.info(f"Loaded emotion model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Emotion transition memory (Markov-like model)
emotion_texts = []  # Store original text for clustering
emotion_memory = []  # Store emotion labels for transition tracking
transition_matrix = {e1: {e2: 0 for e2 in EMOTION_LABELS} for e1 in EMOTION_LABELS}


def transformer_emotion(text):
    """
    Predict emotion using transformer model.
    
    Args:
        text (str): Input text to classify
        
    Returns:
        tuple: (predicted_emotion_label, emotion_probability_distribution)
    """
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = EMOTION_LABELS[pred_idx]
            
        return pred_label, probs.numpy().flatten()
    except Exception as e:
        logger.error(f"Error in transformer_emotion: {e}")
        return "neutral", np.array([1.0/len(EMOTION_LABELS)] * len(EMOTION_LABELS))


def sentiment_tone(text):
    """
    Analyze sentiment polarity of text using TextBlob.
    
    Args:
        text (str): Input text
        
    Returns:
        str: 'positive', 'negative', or 'neutral'
    """
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.3:
            return "positive"
        elif polarity < -0.3:
            return "negative"
        return "neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "neutral"


def update_transition(prev_emotion, current_emotion):
    """
    Update emotion transition matrix (Markov chain model).
    
    Args:
        prev_emotion (str): Previous emotion state
        current_emotion (str): Current emotion state
    """
    if prev_emotion and current_emotion and prev_emotion in transition_matrix:
        if current_emotion in transition_matrix[prev_emotion]:
            transition_matrix[prev_emotion][current_emotion] += 1


def predict_emotion(text):
    """
    Main emotion prediction function with hybrid approach.
    Combines transformer model, sentiment analysis, and rule-based refinement.
    
    Args:
        text (str): User input text
        
    Returns:
        tuple: (predicted_emotion_label, emotion_embedding_probabilities)
    """
    if not text or not text.strip():
        return "neutral", np.array([1.0/len(EMOTION_LABELS)] * len(EMOTION_LABELS))
    
    # Get transformer-based prediction
    transformer_label, embedding = transformer_emotion(text)
    
    # Get sentiment for rule-based refinement
    sentiment = sentiment_tone(text)
    
    # Get previous emotion for transition tracking
    last_emotion = emotion_memory[-1] if emotion_memory else None
    
    # Refinement: If transformer says neutral but sentiment is negative,
    # check for specific keywords that indicate sadness
    if transformer_label == "neutral" and sentiment == "negative":
        sadness_keywords = ["tired", "lonely", "work", "stress", "sad", "depressed", 
                           "unhappy", "down", "miserable", "hopeless"]
        if any(keyword in text.lower() for keyword in sadness_keywords):
            transformer_label = "sadness"
    
    # Store for transition tracking and clustering
    emotion_texts.append(text)
    emotion_memory.append(transformer_label)
    update_transition(last_emotion, transformer_label)
    
    return transformer_label, embedding


def get_emotion_clusters():
    """
    Cluster emotion embeddings using K-means and visualize with PCA.
    
    Returns:
        str: Path to saved cluster visualization image, or None if insufficient data
    """
    if len(emotion_texts) < 5:
        return None
    
    try:
        # Extract embeddings for all stored texts
        X = []
        for text in emotion_texts:
            _, emb = transformer_emotion(text)
            X.append(emb)
        
        X = np.array(X)
        
        # Determine optimal number of clusters (3-5)
        n_clusters = min(5, max(3, len(emotion_texts) // 3))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_reduced[:, 0], 
            X_reduced[:, 1], 
            c=cluster_labels, 
            cmap="viridis",
            alpha=0.6,
            s=100
        )
        
        # Add emotion labels as annotations
        for i, (x, y) in enumerate(X_reduced):
            if i < len(emotion_memory):
                plt.annotate(
                    emotion_memory[i][:3].upper(), 
                    (x, y), 
                    fontsize=8,
                    alpha=0.7
                )
        
        plt.colorbar(scatter, label='Cluster')
        plt.title("Emotion Embedding Clusters (PCA Visualization)", fontsize=14, fontweight='bold')
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save visualization
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_dir = os.path.join(BASE_DIR, "backend", "static")
        os.makedirs(static_dir, exist_ok=True)
        
        cluster_path = os.path.join(static_dir, "emotion_clusters.png")
        plt.savefig(cluster_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster visualization saved to {cluster_path}")
        return "static/emotion_clusters.png"
        
    except Exception as e:
        logger.error(f"Error generating clusters: {e}")
        return None


def get_transition_matrix():
    """
    Get normalized emotion transition probabilities (Markov chain).
    
    Returns:
        dict: Normalized transition probability matrix
    """
    normalized = {}
    for e1 in EMOTION_LABELS:
        total = sum(transition_matrix[e1].values())
        if total > 0:
            normalized[e1] = {
                e2: round(transition_matrix[e1][e2] / total, 3) 
                for e2 in EMOTION_LABELS
            }
        else:
            # If no transitions from this emotion, return uniform distribution
            normalized[e1] = {e2: round(1.0 / len(EMOTION_LABELS), 3) for e2 in EMOTION_LABELS}
    
    return normalized


def get_emotion_statistics():
    """
    Get statistics about emotion distribution in conversation history.
    
    Returns:
        dict: Statistics including counts, percentages, and most common emotions
    """
    if not emotion_memory:
        return {
            "total_interactions": 0,
            "emotion_counts": {},
            "emotion_percentages": {},
            "most_common": None
        }
    
    emotion_counts = Counter(emotion_memory)
    total = len(emotion_memory)
    
    return {
        "total_interactions": total,
        "emotion_counts": dict(emotion_counts),
        "emotion_percentages": {
            emotion: round(count / total * 100, 2) 
            for emotion, count in emotion_counts.items()
        },
        "most_common": emotion_counts.most_common(1)[0][0] if emotion_counts else None,
        "unique_emotions": len(emotion_counts)
    }


def reset_memory():
    """
    Reset conversation memory (useful for testing or starting fresh).
    """
    global emotion_texts, emotion_memory, transition_matrix
    emotion_texts = []
    emotion_memory = []
    transition_matrix = {e1: {e2: 0 for e2 in EMOTION_LABELS} for e1 in EMOTION_LABELS}
    logger.info("Emotion memory reset")
