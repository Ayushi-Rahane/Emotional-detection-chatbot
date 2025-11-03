"""
Unit tests for emotion detection model
Tests emotion classification, sentiment analysis, and clustering functionality
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from emotion_model import (
    predict_emotion,
    sentiment_tone,
    get_transition_matrix,
    get_emotion_statistics,
    reset_memory,
    EMOTION_LABELS
)


class TestEmotionPrediction:
    """Test emotion prediction functionality"""
    
    def test_predict_joy(self):
        """Test prediction of joy emotion"""
        emotion, embedding = predict_emotion("I'm so happy today! Everything is wonderful!")
        assert emotion in EMOTION_LABELS
        assert emotion == "joy" or "joy" in emotion.lower()
        assert embedding is not None
        assert len(embedding) == len(EMOTION_LABELS)
    
    def test_predict_sadness(self):
        """Test prediction of sadness emotion"""
        emotion, embedding = predict_emotion("I'm feeling really down and depressed today.")
        assert emotion in EMOTION_LABELS
        assert embedding is not None
    
    def test_predict_anger(self):
        """Test prediction of anger emotion"""
        emotion, embedding = predict_emotion("I'm so furious and angry about this situation!")
        assert emotion in EMOTION_LABELS
        assert embedding is not None
    
    def test_predict_neutral(self):
        """Test prediction of neutral emotion"""
        emotion, embedding = predict_emotion("The weather is nice today.")
        assert emotion in EMOTION_LABELS
        assert embedding is not None
    
    def test_predict_empty_string(self):
        """Test handling of empty string"""
        emotion, embedding = predict_emotion("")
        assert emotion == "neutral"
        assert embedding is not None
    
    def test_predict_whitespace(self):
        """Test handling of whitespace-only input"""
        emotion, embedding = predict_emotion("   ")
        assert emotion == "neutral"
        assert embedding is not None


class TestSentimentAnalysis:
    """Test sentiment analysis functionality"""
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        sentiment = sentiment_tone("I love this amazing product! It's fantastic!")
        assert sentiment == "positive"
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        sentiment = sentiment_tone("I hate this terrible experience. It's awful!")
        assert sentiment == "negative"
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        sentiment = sentiment_tone("The book is on the table.")
        assert sentiment == "neutral"


class TestTransitionMatrix:
    """Test emotion transition tracking"""
    
    def test_transition_matrix_structure(self):
        """Test that transition matrix has correct structure"""
        reset_memory()
        
        # Make some predictions to populate transitions
        predict_emotion("I'm happy!")
        predict_emotion("Now I'm sad.")
        predict_emotion("I'm angry now!")
        
        matrix = get_transition_matrix()
        
        assert isinstance(matrix, dict)
        assert len(matrix) == len(EMOTION_LABELS)
        
        for emotion in EMOTION_LABELS:
            assert emotion in matrix
            assert isinstance(matrix[emotion], dict)
            assert len(matrix[emotion]) == len(EMOTION_LABELS)
            
            # Check that probabilities sum to approximately 1
            total_prob = sum(matrix[emotion].values())
            assert 0.95 <= total_prob <= 1.05  # Allow small floating point errors


class TestEmotionStatistics:
    """Test emotion statistics generation"""
    
    def test_statistics_structure(self):
        """Test that statistics have correct structure"""
        reset_memory()
        
        # Make some predictions
        predict_emotion("I'm happy!")
        predict_emotion("I'm sad.")
        predict_emotion("I'm happy again!")
        
        stats = get_emotion_statistics()
        
        assert "total_interactions" in stats
        assert "emotion_counts" in stats
        assert "emotion_percentages" in stats
        
        assert stats["total_interactions"] == 3
        assert isinstance(stats["emotion_counts"], dict)
        assert isinstance(stats["emotion_percentages"], dict)
    
    def test_statistics_empty(self):
        """Test statistics when no interactions"""
        reset_memory()
        stats = get_emotion_statistics()
        
        assert stats["total_interactions"] == 0
        assert stats["emotion_counts"] == {}
        assert stats["emotion_percentages"] == {}


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_long_text(self):
        """Test handling of very long text input"""
        long_text = "This is a test. " * 100
        emotion, embedding = predict_emotion(long_text)
        assert emotion in EMOTION_LABELS
        assert embedding is not None
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special_text = "I'm feeling!!! @#$%^&*() amazing!!! 🎉🎊"
        emotion, embedding = predict_emotion(special_text)
        assert emotion in EMOTION_LABELS
        assert embedding is not None
    
    def test_mixed_case(self):
        """Test handling of mixed case text"""
        mixed_text = "I'M So HaPPy and EXCITED!!!"
        emotion, embedding = predict_emotion(mixed_text)
        assert emotion in EMOTION_LABELS
        assert embedding is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

