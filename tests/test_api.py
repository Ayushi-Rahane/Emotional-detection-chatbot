"""
Integration tests for Flask API endpoints
Tests the web API functionality
"""

import pytest
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHomeEndpoint:
    """Test home page endpoint"""
    
    def test_home_page(self, client):
        """Test that home page loads successfully"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Emotion Detection Chatbot' in response.data


class TestPredictEndpoint:
    """Test emotion prediction endpoint"""
    
    def test_predict_with_valid_message(self, client):
        """Test prediction with valid message"""
        response = client.post(
            '/predict',
            data=json.dumps({'message': 'I am very happy today!'}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'predicted_emotion' in data
        assert 'bot_reply' in data
        assert 'transition_probs' in data
        assert 'emotion_statistics' in data
        assert data['predicted_emotion'] in ['joy', 'sadness', 'anger', 'fear', 'neutral', 'surprise', 'disgust']
    
    def test_predict_with_empty_message(self, client):
        """Test prediction with empty message"""
        response = client.post(
            '/predict',
            data=json.dumps({'message': ''}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_without_message_field(self, client):
        """Test prediction without message field"""
        response = client.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_without_json(self, client):
        """Test prediction without JSON data"""
        response = client.post('/predict')
        assert response.status_code == 400
    
    def test_predict_multiple_emotions(self, client):
        """Test prediction with different emotion inputs"""
        test_cases = [
            ('I am so sad and depressed', 'sadness'),
            ('I am furious and angry!', 'anger'),
            ('I am scared and frightened', 'fear'),
            ('This is amazing! I love it!', 'joy')
        ]
        
        for message, expected_emotion in test_cases:
            response = client.post(
                '/predict',
                data=json.dumps({'message': message}),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'predicted_emotion' in data


class TestStatsEndpoint:
    """Test statistics endpoint"""
    
    def test_stats_endpoint(self, client):
        """Test that stats endpoint returns data"""
        # Make a prediction first to populate statistics
        client.post(
            '/predict',
            data=json.dumps({'message': 'I am happy!'}),
            content_type='application/json'
        )
        
        response = client.get('/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'transition_matrix' in data
        assert 'statistics' in data


class TestClustersEndpoint:
    """Test clusters endpoint"""
    
    def test_clusters_with_insufficient_data(self, client):
        """Test clusters endpoint with insufficient data"""
        response = client.get('/clusters')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        # Should return a message indicating insufficient data
        assert 'message' in data or 'cluster_image' in data
    
    def test_clusters_after_many_predictions(self, client):
        """Test clusters endpoint after multiple predictions"""
        # Make multiple predictions to populate data
        messages = [
            'I am very happy!',
            'I feel sad today.',
            'I am angry about this!',
            'I am scared.',
            'I am surprised!',
            'This is disgusting.'
        ]
        
        for message in messages:
            client.post(
                '/predict',
                data=json.dumps({'message': message}),
                content_type='application/json'
            )
        
        response = client.get('/clusters')
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

