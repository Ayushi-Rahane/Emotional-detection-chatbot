"""
Emotion Detection Chatbot - Flask Web Application
Main entry point for the emotion detection chatbot API and web interface.

This module provides RESTful API endpoints for emotion prediction and serves
the interactive web interface for real-time emotion detection in conversations.
"""

from flask import Flask, request, jsonify, render_template
from emotion_model import predict_emotion, get_emotion_clusters, get_transition_matrix, get_emotion_statistics
import random
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Load emotion responses configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES_FILE = os.path.join(BASE_DIR, "data", "emotion_responses.json")

try:
    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        RESPONSES = json.load(f)
    logger.info(f"Loaded emotion responses from {RESPONSES_FILE}")
except FileNotFoundError:
    logger.error(f"Emotion responses file not found: {RESPONSES_FILE}")
    RESPONSES = {
        "neutral": ["I'm here to listen. How can I help you today?"],
        "joy": ["That's wonderful! I'm glad you're feeling positive!"],
        "sadness": ["I'm sorry you're feeling this way. You're not alone."],
        "anger": ["I understand you're feeling frustrated. Let's work through this."],
        "fear": ["It's okay to feel afraid. You're brave for expressing it."]
    }


@app.route("/")
def home():
    """Serve the main chatbot interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict emotion from user message and generate appropriate response.
    
    Request body:
        - message (str): User's text input
        
    Returns:
        JSON response containing:
        - predicted_emotion (str): Detected emotion label
        - bot_reply (str): Generated response
        - transition_probs (dict): Emotion transition probabilities
        - emotion_statistics (dict): Statistics about emotion history
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"error": "Message field missing or empty"}), 400
        
        # Predict emotion using the ML model
        emotion, embedding = predict_emotion(message)
        
        # Get appropriate response based on detected emotion
        response_list = RESPONSES.get(emotion, RESPONSES.get("neutral", ["I understand."]))
        bot_reply = random.choice(response_list)
        
        # Get transition probabilities and statistics
        transition_probs = get_transition_matrix()
        emotion_stats = get_emotion_statistics()
        
        logger.info(f"Predicted emotion: {emotion} for message: '{message[:50]}...'")
        
        return jsonify({
            "predicted_emotion": emotion,
            "bot_reply": bot_reply,
            "transition_probs": transition_probs,
            "emotion_statistics": emotion_stats,
            "confidence": float(max(embedding)) if embedding is not None else 0.0
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """Get emotion statistics and transition matrix."""
    try:
        return jsonify({
            "transition_matrix": get_transition_matrix(),
            "statistics": get_emotion_statistics()
        })
    except Exception as e:
        logger.error(f"Error in stats endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/generate_clusters", methods=["GET"])
def generate_clusters():
    """Generate and return emotion cluster visualization."""
    try:
        cluster_path = get_emotion_clusters()
        if cluster_path:
            return jsonify({
                "cluster_image": cluster_path,
                "message": "Clusters generated successfully"
            })
        else:
            return jsonify({
                "message": "Not enough data for clustering (need at least 5 interactions)"
            }), 200
    except Exception as e:
        logger.error(f"Error generating clusters: {str(e)}")
        return jsonify({"error": f"Error generating clusters: {str(e)}"}), 500


@app.route("/export", methods=["GET"])
def export_conversation():
    """Export conversation history as JSON or CSV."""
    from emotion_model import emotion_texts, emotion_memory
    try:
        export_format = request.args.get('format', 'json')  # 'json' or 'csv'
        
        if not emotion_memory or not emotion_texts:
            return jsonify({"error": "No conversation data to export"}), 400
        
        # Prepare data
        conversation_data = []
        for i, (text, emotion) in enumerate(zip(emotion_texts, emotion_memory)):
            conversation_data.append({
                "timestamp": i,  # Simple index-based timestamp
                "text": text,
                "emotion": emotion
            })
        
        if export_format == 'csv':
            # Return CSV
            csv_output = "timestamp,text,emotion\n"
            for item in conversation_data:
                escaped_text = item['text'].replace('"', '""')
                csv_output += f"{item['timestamp']},\"{escaped_text}\",{item['emotion']}\n"
            
            from flask import Response
            return Response(
                csv_output,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=emotion_conversation.csv'}
            )
        else:
            # Return JSON
            return jsonify(conversation_data)
            
    except Exception as e:
        logger.error(f"Error exporting conversation: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    """Get conversation history with timestamps."""
    from emotion_model import emotion_texts, emotion_memory
    try:
        history = []
        for i, (text, emotion) in enumerate(zip(emotion_texts, emotion_memory)):
            history.append({
                "index": i,
                "text": text,
                "emotion": emotion
            })
        return jsonify({"history": history})
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Emotion Detection Chatbot server...")
    app.run(debug=True, host="0.0.0.0", port=5001)
