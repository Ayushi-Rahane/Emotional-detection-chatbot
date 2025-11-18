/**
 * Emotion Detection Chatbot - Frontend JavaScript
 * Handles user interactions, API communication, and visualization
 */

const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const chatBox = document.getElementById("chatBox");
const statsBtn = document.getElementById("statsBtn");
const historyBtn = document.getElementById("historyBtn");
const exportBtn = document.getElementById("exportBtn");
const themeBtn = document.getElementById("themeBtn");
const voiceBtn = document.getElementById("voiceBtn");
const statsPanel = document.getElementById("statsPanel");
const historyPanel = document.getElementById("historyPanel");
const clusterBtn = document.getElementById("clusterBtn"); // ✅ New button

const API_URL = "http://127.0.0.1:5001/predict";
const STATS_URL = "http://127.0.0.1:5001/stats";
const EXPORT_URL = "http://127.0.0.1:5001/export";
const HISTORY_URL = "http://127.0.0.1:5001/history";
const CLUSTER_URL = "http://127.0.0.1:5001/generate_clusters"; // ✅ New endpoint

// Voice input variables
let recognition = null;
let isListening = false;

// Emotion color mapping for visual feedback
const emotionColors = {
  joy: "#FFF9C4",
  sadness: "#BBDEFB",
  anger: "#FFCDD2",
  fear: "#E1BEE7",
  disgust: "#C5E1A5",
  surprise: "#FFE0B2",
  neutral: "#E0E0E0"
};

// Event listeners
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

if (statsBtn) statsBtn.addEventListener("click", toggleStats);
if (historyBtn) historyBtn.addEventListener("click", toggleHistory);
if (exportBtn) exportBtn.addEventListener("click", exportConversation);
if (themeBtn) themeBtn.addEventListener("click", toggleTheme);
if (voiceBtn) voiceBtn.addEventListener("click", toggleVoiceInput);
if (clusterBtn) clusterBtn.addEventListener("click", showClusterImage); // ✅ new

// Initialize Web Speech API
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
    sendMessage();
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    voiceBtn.classList.remove("recording");
    isListening = false;
  };

  recognition.onend = () => {
    voiceBtn.classList.remove("recording");
    isListening = false;
  };
}

/**
 * Toggle voice input
 */
function toggleVoiceInput() {
  if (!recognition) {
    addMessage("Sorry, voice input is not supported in your browser.", "bot");
    return;
  }

  if (isListening) {
    recognition.stop();
    voiceBtn.classList.remove("recording");
    isListening = false;
  } else {
    recognition.start();
    voiceBtn.classList.add("recording");
    isListening = true;
  }
}

/**
 * Add a message bubble to the chat interface
 */
function addMessage(text, sender, emotion = null, confidence = null) {
  const bubble = document.createElement("div");
  bubble.classList.add("chat-bubble", sender);

  const messageText = document.createElement("div");
  messageText.textContent = text;
  bubble.appendChild(messageText);

  if (sender === "bot" && emotion) {
    const color = emotionColors[emotion.toLowerCase()] || "#F1F0F0";
    bubble.style.backgroundColor = color;

    const tag = document.createElement("div");
    tag.classList.add("emotion-tag");
    tag.innerHTML = `<strong>${emotion}</strong>`;
    if (confidence !== null) {
      tag.innerHTML += ` <span style="font-size:0.9em; opacity:0.7;">(${(confidence * 100).toFixed(1)}%)</span>`;
    }
    bubble.appendChild(tag);
  }

  chatBox.appendChild(bubble);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * Send user message to backend and handle response
 */
async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";
  sendBtn.disabled = true;

  // Show typing indicator
  const typing = document.createElement("div");
  typing.classList.add("chat-bubble", "bot");
  typing.textContent = "Thinking...";
  typing.id = "typing-indicator";
  chatBox.appendChild(typing);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) chatBox.removeChild(typingIndicator);

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();

    if (data.predicted_emotion && data.bot_reply) {
      addMessage(data.bot_reply, "bot", data.predicted_emotion, data.confidence);

      // Update panels if open
      if (statsPanel && statsPanel.style.display !== "none") await loadStats();
      if (historyPanel && historyPanel.style.display !== "none") await loadHistory();
    } else if (data.error) {
      addMessage(`Error: ${data.error}`, "bot");
    }
  } catch (err) {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) chatBox.removeChild(typingIndicator);
    addMessage("Unable to connect to server. Make sure Flask is running.", "bot");
    console.error("Error:", err);
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}

/**
 * Toggle statistics panel visibility
 */
function toggleStats() {
  if (!statsPanel) return;
  if (historyPanel && historyPanel.style.display !== "none") historyPanel.style.display = "none";
  statsPanel.style.display = statsPanel.style.display === "none" ? "block" : "none";
  if (statsPanel.style.display === "block") loadStats();
}

/**
 * Load and display emotion statistics
 */
async function loadStats() {
  if (!statsPanel) return;

  try {
    const response = await fetch(STATS_URL);
    if (!response.ok) throw new Error("Failed to fetch stats");

    const data = await response.json();
    displayStatistics(data);
  } catch (error) {
    console.error("Error loading stats:", error);
  }
}


function displayStatistics(stats) {
  const content = statsPanel.querySelector(".stats-content");
  if (!content) return;

  const s = stats.statistics || {};   // <-- FIX

  let html = `<p><strong>Total Interactions:</strong> ${s.total_interactions || 0}</p>`;

  if (s.emotion_percentages) {
    html += '<div class="emotion-bars">';
    for (const [emotion, percentage] of Object.entries(s.emotion_percentages)) {
      const color = emotionColors[emotion.toLowerCase()] || "#E0E0E0";
      html += `
        <div class="emotion-bar-item">
          <span><strong>${emotion}</strong></span>
          <div class="progress">
            <div class="progress-bar" style="width:${percentage}%; background:${color};">
              ${percentage.toFixed(1)}%
            </div>
          </div>
        </div>
      `;
    }
    html += "</div>";
  }

  if (s.most_common) {
    html += `<p style="margin-top:10px;"><strong>Final Emotion:</strong> ${s.most_common}</p>`;
  }

  // RECOMMENDATIONS (unchanged)
  if (stats.recommendations) {
    const recs = stats.recommendations;
    html += `<div style="margin-top:15px; padding:12px; background:#f7f7f7; border-radius:8px;">
      <strong>Recommendations:</strong>`;

    if (recs.songs && recs.songs.length) {
      html += `<p style="margin-top:8px;"><strong>Songs:</strong></p><ul>`;
      recs.songs.forEach(s => {
        html += `<li><a href="${s.url}" target="_blank">${s.title}</a></li>`;
      });
      html += `</ul>`;
    }

    if (recs.quotes && recs.quotes.length) {
      html += `<p style="margin-top:8px;"><strong>Quotes:</strong></p><ul>`;
      recs.quotes.forEach(q => {
        html += `<li>${q}</li>`;
      });
      html += `</ul>`;
    }

    html += `</div>`;
  }

  content.innerHTML = html;
}

/**
 * Toggle history panel visibility
 */
function toggleHistory() {
  if (!historyPanel) return;
  if (statsPanel && statsPanel.style.display !== "none") statsPanel.style.display = "none";
  historyPanel.style.display = historyPanel.style.display === "none" ? "block" : "none";
  if (historyPanel.style.display === "block") loadHistory();
}

/**
 * Load and display conversation history
 */
async function loadHistory() {
  const historyContent = document.getElementById("historyContent");
  if (!historyContent) return;
  try {
    const response = await fetch(HISTORY_URL);
    if (!response.ok) throw new Error("Failed to fetch history");
    const data = await response.json();
    if (data.history && data.history.length > 0) displayHistory(data.history);
    else historyContent.innerHTML = "<p>No conversation history yet. Start chatting!</p>";
  } catch (err) {
    console.error("Error loading history:", err);
    historyContent.innerHTML = "<p>Error loading history</p>";
  }
}

/**
 * Display conversation history in timeline
 */
function displayHistory(history) {
  const historyContent = document.getElementById("historyContent");
  if (!historyContent) return;
  let html = "";
  history.forEach((item) => {
    const color = emotionColors[item.emotion.toLowerCase()] || "#E0E0E0";
    html += `
      <div class="timeline-item" style="border-left-color: ${color};">
        <div class="timeline-text">"${item.text}"</div>
        <div class="timeline-emotion" style="color: ${color};">
          ${item.emotion.charAt(0).toUpperCase() + item.emotion.slice(1)}
        </div>
      </div>`;
  });
  historyContent.innerHTML = html;
}

/**
 * Export conversation as CSV
 */
async function exportConversation() {
  try {
    const response = await fetch(`${EXPORT_URL}?format=csv`);
    if (!response.ok) throw new Error("Failed to export conversation");
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "emotion_conversation.csv";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    exportBtn.textContent = "Saved!";
    setTimeout(() => (exportBtn.textContent = "Export"), 2000);
  } catch (err) {
    console.error("Error exporting conversation:", err);
    alert("Failed to export conversation.");
  }
}

/**
 * Toggle dark/light theme
 */
function toggleTheme() {
  const body = document.body;
  if (body.classList.contains("dark-theme")) {
    body.classList.remove("dark-theme");
    themeBtn.textContent = "Dark";
    localStorage.setItem("theme", "light");
  } else {
    body.classList.add("dark-theme");
    themeBtn.textContent = "Light";
    localStorage.setItem("theme", "dark");
  }
}

/**
 * Show Emotion Cluster Image (with loader)
 */
async function showClusterImage() {
  const loader = document.getElementById("clusterLoader");
  const imgElement = document.getElementById("clusterImage");
  const msgElement = document.getElementById("clusterMessage");

  loader.style.display = "block";
  imgElement.style.display = "none";
  msgElement.textContent = "";

  const modal = new bootstrap.Modal(document.getElementById("clusterModal"));
  modal.show();

  try {
    const response = await fetch(CLUSTER_URL);
    if (!response.ok) throw new Error("Failed to fetch cluster image");
    const data = await response.json();

    setTimeout(() => {
      loader.style.display = "none";
      imgElement.src = data.cluster_image;
      imgElement.style.display = "block";
      msgElement.textContent = data.message || "Clusters generated successfully.";
    }, 700);
  } catch (err) {
    console.error("Error loading clusters:", err);
    loader.style.display = "none";
    msgElement.textContent = "Failed to load emotion clusters.";
  }
}

/**
 * Initialize
 */
document.addEventListener("DOMContentLoaded", () => {
  userInput.focus();
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "dark") {
    document.body.classList.add("dark-theme");
    themeBtn.textContent = "Light";
  }
});
