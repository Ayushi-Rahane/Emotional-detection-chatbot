import os
import json
from collections import Counter
from typing import List, Tuple
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_emotion.pt")

# Simple whitespace tokenizer
def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()

def build_vocab(texts, min_freq=1, max_size=20000):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_size):
        if freq < min_freq:
            break
        if word not in vocab:
            vocab[word] = len(vocab)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    logger.info(f"Saved vocab ({len(vocab)}) to {VOCAB_PATH}")
    return vocab

def load_vocab():
    if not os.path.exists(VOCAB_PATH):
        return None
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def encode_text(text: str, vocab: dict, max_len: int = 100):
    toks = simple_tokenize(text)
    ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in toks][:max_len]
    if len(ids) < max_len:
        ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    return ids

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, num_classes=len(EMOTION_LABELS), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x)                       # (B, L, E)
        out, _ = self.lstm(emb)                       # out: (B, L, H*2)
        pooled = out.mean(dim=1)                      # (B, H*2)
        logits = self.fc(self.dropout(pooled))
        return logits

# Lightweight loader/cacher
_state = {"model": None, "vocab": None, "device": None}

def load_model(device=None):
    if _state["model"] is not None and _state["vocab"] is not None:
        return _state
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device
    vocab = load_vocab()
    if not vocab or not os.path.exists(MODEL_PATH):
        logger.warning("LSTM model or vocab not found. Run training to create model at backend/models/")
        return _state
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = LSTMClassifier(vocab_size=ckpt.get("vocab_size", len(vocab)), embed_dim=ckpt.get("embed_dim",128), hidden_dim=ckpt.get("hidden_dim",128), num_layers=ckpt.get("num_layers",1))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    _state["model"] = model
    _state["vocab"] = vocab
    logger.info("Loaded LSTM model for inference")
    return _state

def predict(text: str) -> Tuple[str, List[float]]:
    """
    Returns (label, probs_array). If model not available returns neutral uniform.
    """
    import numpy as np
    if not text or not text.strip():
        return "neutral", np.array([1.0/len(EMOTION_LABELS)] * len(EMOTION_LABELS))
    state = load_model()
    model = state.get("model")
    vocab = state.get("vocab")
    device = state.get("device") or torch.device("cpu")
    if model is None or vocab is None:
        logger.warning("LSTM model/vocab missing, returning neutral fallback")
        return "neutral", np.array([1.0/len(EMOTION_LABELS)] * len(EMOTION_LABELS))
    ids = torch.tensor([encode_text(text, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
        pred = EMOTION_LABELS[int(probs.argmax())]
    return pred, probs