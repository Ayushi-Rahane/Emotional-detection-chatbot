"""
Emotion Detection Chatbot - Flask Web Application
Main entry point for the emotion detection chatbot API and web interface.
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

# ---------------- LOAD RESPONSE + RECOMMENDATION FILES ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESPONSES_FILE = os.path.join(BASE_DIR, "data", "emotion_responses.json")
RECOMMENDATIONS_FILE = os.path.join(BASE_DIR, "data", "emotion_recommendations.json")

# Load emotion responses
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

# Load recommendations (songs + quotes)
try:
    with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
        RECOMMENDATIONS = json.load(f)
    logger.info(f"Loaded emotion recommendations from {RECOMMENDATIONS_FILE}")
except FileNotFoundError:
    logger.error(f"emotion_recommendations.json NOT FOUND, continuing without recommendations.")
    RECOMMENDATIONS = {}

# ----------------------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Message field missing or empty"}), 400

        # Predict emotion
        emotion, embedding = predict_emotion(message)

        # Select bot response
        response_list = RESPONSES.get(emotion, RESPONSES.get("neutral", ["I understand."]))
        bot_reply = random.choice(response_list)

        # Stats + transitions
        transition_probs = get_transition_matrix()
        emotion_stats = get_emotion_statistics()

        logger.info(f"Predicted emotion: {emotion}")

        # Important: Add EXACT predictions, but DO NOT show recommendations in chat.
        return jsonify({
            "predicted_emotion": emotion,
            "bot_reply": bot_reply,
            "transition_probs": transition_probs,
            "emotion_statistics": emotion_stats,
            "confidence": float(max(embedding)) if embedding is not None else 0.0
        })

    except Exception as e:
        logger.error(f"Predict error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/stats", methods=["GET"])
def stats():
    try:
        stats = get_emotion_statistics()

        # Most common emotion = final emotion
        final_emotion = stats.get("most_common")

        # Load recommendations ONLY for stats page
        recs = RECOMMENDATIONS.get(final_emotion, {}) if final_emotion else {}

        return jsonify({
            "transition_matrix": get_transition_matrix(),
            "statistics": stats,
            "recommendations": recs
        })

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/generate_clusters", methods=["GET"])
def generate_clusters():
    try:
        cluster_path = get_emotion_clusters()
        if cluster_path:
            return jsonify({
                "cluster_image": cluster_path,
                "message": "Clusters generated successfully"
            })
        else:
            return jsonify({"message": "Not enough data for clustering (need at least 5 interactions)"})
    except Exception as e:
        logger.error(f"Cluster error: {str(e)}")
        return jsonify({"error": f"Error generating clusters: {str(e)}"}), 500


@app.route("/export", methods=["GET"])
def export_conversation():
    from emotion_model import emotion_texts, emotion_memory
    try:
        export_format = request.args.get('format', 'json')

        if not emotion_memory or not emotion_texts:
            return jsonify({"error": "No conversation data to export"}), 400

        conversation_data = []
        for i, (text, emotion) in enumerate(zip(emotion_texts, emotion_memory)):
            conversation_data.append({
                "timestamp": i,
                "text": text,
                "emotion": emotion
            })

        if export_format == "csv":
            csv_output = "timestamp,text,emotion\n"
            for item in conversation_data:
                escaped = item["text"].replace('"', '""')
                csv_output += f'{item["timestamp"]},"{escaped}",{item["emotion"]}\n'

            from flask import Response
            return Response(
                csv_output,
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=emotion_conversation.csv"}
            )

        return jsonify(conversation_data)

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    from emotion_model import emotion_texts, emotion_memory
    try:
        history = []
        for i, (text, emotion) in enumerate(zip(emotion_texts, emotion_memory)):
            history.append({"index": i, "text": text, "emotion": emotion})
        return jsonify({"history": history})
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Emotion Detection Chatbot server...")
    app.run(debug=True, host="0.0.0.0", port=5001)
