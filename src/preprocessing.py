"""
Text Preprocessing Module
Provides utilities for cleaning and preprocessing text data for emotion classification.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Note: Some NLTK data may not be available: {e}")

# Initialize lemmatizer
try:
    lemmatizer = WordNetLemmatizer()
except:
    lemmatizer = None


def clean_text(text, remove_stopwords=True, lemmatize=True):
    """
    Clean and preprocess text for emotion classification.
    
    Args:
        text (str): Input text to clean
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize words
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation (optional - can keep for emotion detection)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords if requested
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except:
            pass
    
    # Lemmatize if requested
    if lemmatize and lemmatizer:
        try:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except:
            pass
    
    # Join tokens back
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text.strip()


def remove_emojis(text):
    """
    Remove emojis from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without emojis
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def normalize_text(text):
    """
    Normalize text by expanding contractions and standardizing format.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Common contractions
    contractions = {
        "don't": "do not",
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'m": " am",
        "'d": " would",
    }
    
    text = text.lower()
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    return text


def extract_features(text):
    """
    Extract additional features from text that might be useful for emotion detection.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of features
    """
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'has_question': '?' in text,
        'has_exclamation': '!' in text,
        'capital_letters': sum(1 for c in text if c.isupper()),
    }
    
    return features

