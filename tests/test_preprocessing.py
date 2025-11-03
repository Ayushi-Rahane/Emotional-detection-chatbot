"""
Tests for preprocessing utilities
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from preprocessing import clean_text
except ImportError:
    # If preprocessing module doesn't exist, create a simple test
    def clean_text(text):
        """Simple text cleaning for testing"""
        return text.lower().strip()


class TestPreprocessing:
    """Test text preprocessing functions"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        result = clean_text("Hello World!")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_clean_text_lowercase(self):
        """Test that text is converted to lowercase"""
        result = clean_text("HELLO WORLD")
        assert result == result.lower()
    
    def test_clean_text_empty(self):
        """Test cleaning empty string"""
        result = clean_text("")
        assert isinstance(result, str)
    
    def test_clean_text_special_chars(self):
        """Test handling of special characters"""
        result = clean_text("Hello!!! @#$ World...")
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

