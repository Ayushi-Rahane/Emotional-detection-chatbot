"""
Setup helper script to download NLTK data and pre-load transformer model.
This script is called by setup.bat to ensure all required resources are available.
"""

import sys

print("\n=== Downloading NLTK Data ===")
try:
    import nltk
    print("Downloading punkt...")
    nltk.download('punkt', quiet=True)
    print("Downloading stopwords...")
    nltk.download('stopwords', quiet=True)
    print("Downloading wordnet...")
    nltk.download('wordnet', quiet=True)
    print("Downloading omw-1.4...")
    nltk.download('omw-1.4', quiet=True)
    print("✓ NLTK data downloaded successfully!")
except Exception as e:
    print(f"✗ Error downloading NLTK data: {e}")
    sys.exit(1)

print("\n=== Pre-downloading Transformer Model ===")
print("Model: j-hartmann/emotion-english-distilroberta-base")
print("This may take a few minutes (~500MB)...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    print(f"Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("✓ Transformer model downloaded successfully!")
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    print("Note: The model will be downloaded automatically on first run.")
    # Don't exit with error - model can be downloaded later

print("\n=== Setup Complete ===")
sys.exit(0)

