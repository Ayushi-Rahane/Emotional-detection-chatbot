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

const API_URL = "http://127.0.0.1:5001/predict";
const STATS_URL = "http://127.0.0.1:5001/stats";
const EXPORT_URL = "http://127.0.0.1:5001/export";
const HISTORY_URL = "http://127.0.0.1:5001/history";

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

const emotionIcons = {
  joy: "😊",
  sadness: "😢",
  anger: "😠",
  fear: "😨",
  disgust: "🤢",
  surprise: "😲",
  neutral: "😐"
};

// Event listeners
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

if (statsBtn) {
  statsBtn.addEventListener("click", toggleStats);
}

if (historyBtn) {
  historyBtn.addEventListener("click", toggleHistory);
}

if (exportBtn) {
  exportBtn.addEventListener("click", exportConversation);
}

if (themeBtn) {
  themeBtn.addEventListener("click", toggleTheme);
}

if (voiceBtn) {
  voiceBtn.addEventListener("click", toggleVoiceInput);
}

// Initialize Web Speech API
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
    sendMessage();
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    voiceBtn.classList.remove('recording');
    isListening = false;
  };

  recognition.onend = () => {
    voiceBtn.classList.remove('recording');
    isListening = false;
  };
}

/**
 * Toggle voice input
 */
function toggleVoiceInput() {
  if (!recognition) {
    addMessage('Sorry, voice input is not supported in your browser.', 'bot');
    return;
  }

  if (isListening) {
    recognition.stop();
    voiceBtn.classList.remove('recording');
    isListening = false;
  } else {
    recognition.start();
    voiceBtn.classList.add('recording');
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
  
  // Add reaction buttons for user messages
  if (sender === "user") {
    const reactionContainer = document.createElement("div");
    reactionContainer.classList.add("reaction-container");
    reactionContainer.style.display = "none";
    
    const reactions = ["👍", "❤️", "😊", "😢", "😡", "🤯"];
    reactions.forEach(emoji => {
      const reactionBtn = document.createElement("span");
      reactionBtn.classList.add("reaction-btn");
      reactionBtn.textContent = emoji;
      reactionBtn.addEventListener("click", () => {
        reactionBtn.classList.toggle("active");
        bubble.classList.add("reacted");
      });
      reactionContainer.appendChild(reactionBtn);
    });
    
    bubble.appendChild(reactionContainer);
    
    // Show/hide reactions on hover
    bubble.addEventListener("mouseenter", () => {
      reactionContainer.style.display = "flex";
    });
    
    bubble.addEventListener("mouseleave", () => {
      if (!bubble.classList.contains("reacted")) {
        reactionContainer.style.display = "none";
      }
    });
  }
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
    if (typingIndicator) {
      chatBox.removeChild(typingIndicator);
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.predicted_emotion && data.bot_reply) {
      addMessage(
        data.bot_reply, 
        "bot", 
        data.predicted_emotion,
        data.confidence
      );
      
      // Update panels if open
      if (statsPanel && statsPanel.style.display !== "none") {
        await loadStats();
      }
      if (historyPanel && historyPanel.style.display !== "none") {
        await loadHistory();
      }
    } else if (data.error) {
      addMessage(`Error: ${data.error}`, "bot");
    }
  } catch (err) {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) {
      chatBox.removeChild(typingIndicator);
    }
    addMessage(
      "Unable to connect to server. Make sure Flask is running on port 5000.",
      "bot"
    );
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
  
  // Hide history panel if showing
  if (historyPanel && historyPanel.style.display !== "none") {
    historyPanel.style.display = "none";
  }
  
  if (statsPanel.style.display === "none" || !statsPanel.style.display) {
    statsPanel.style.display = "block";
    loadStats();
  } else {
    statsPanel.style.display = "none";
  }
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
    
    if (data.statistics) {
      displayStatistics(data.statistics);
    }
  } catch (err) {
    console.error("Error loading stats:", err);
    const content = statsPanel.querySelector(".stats-content");
    if (content) {
      content.innerHTML = "<p>Error loading statistics</p>";
    }
  }
}

// Make loadStats available globally
window.loadStats = loadStats;

/**
 * Display emotion statistics in the stats panel
 */
function displayStatistics(stats) {
  if (!statsPanel) return;

  const content = statsPanel.querySelector(".stats-content");
  if (!content) return;

  let html = `<p><strong>Total Interactions:</strong> ${stats.total_interactions || 0}</p>`;
  
  if (stats.emotion_percentages) {
    html += '<div class="emotion-bars">';
    for (const [emotion, percentage] of Object.entries(stats.emotion_percentages)) {
      const color = emotionColors[emotion.toLowerCase()] || "#E0E0E0";
      html += `
        <div class="emotion-bar-item">
          <span>${emotion}</span>
          <div class="progress">
            <div class="progress-bar" role="progressbar" 
                 style="width: ${percentage}%; background-color: ${color};" 
                 aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
              ${percentage.toFixed(1)}%
            </div>
          </div>
        </div>
      `;
    }
    html += '</div>';
  }

  if (stats.most_common) {
    html += `<p><strong>Most Common:</strong> ${stats.most_common}</p>`;
  }

  content.innerHTML = html;
}

/**
 * Toggle history panel visibility
 */
function toggleHistory() {
  if (!historyPanel) return;
  
  // Hide stats panel if showing
  if (statsPanel && statsPanel.style.display !== "none") {
    statsPanel.style.display = "none";
  }
  
  if (historyPanel.style.display === "none" || !historyPanel.style.display) {
    historyPanel.style.display = "block";
    loadHistory();
  } else {
    historyPanel.style.display = "none";
  }
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
    
    if (data.history && data.history.length > 0) {
      displayHistory(data.history);
    } else {
      historyContent.innerHTML = "<p>No conversation history yet. Start chatting!</p>";
    }
  } catch (err) {
    console.error("Error loading history:", err);
    if (historyContent) {
      historyContent.innerHTML = "<p>Error loading history</p>";
    }
  }
}

/**
 * Display conversation history in timeline format
 */
function displayHistory(history) {
  const historyContent = document.getElementById("historyContent");
  if (!historyContent) return;

  let html = '';
  
  history.forEach((item) => {
    const color = emotionColors[item.emotion.toLowerCase()] || "#E0E0E0";
    html += `
      <div class="timeline-item" style="border-left-color: ${color};">
        <div class="timeline-text">"${item.text}"</div>
        <div class="timeline-emotion" style="color: ${color};">${item.emotion.charAt(0).toUpperCase() + item.emotion.slice(1)}</div>
      </div>
    `;
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
    const a = document.createElement('a');
    a.href = url;
    a.download = 'emotion_conversation.csv';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    // Show feedback
    if (exportBtn) {
      const originalText = exportBtn.textContent;
      exportBtn.textContent = "Saved!";
      setTimeout(() => {
        exportBtn.textContent = originalText;
      }, 2000);
    }
  } catch (err) {
    console.error("Error exporting conversation:", err);
    alert("Failed to export conversation. Make sure you have some messages first!");
  }
}

/**
 * Toggle between dark and light theme
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
 * Initialize the application
 */
document.addEventListener("DOMContentLoaded", () => {
  userInput.focus();
  
  // Load saved theme preference
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "dark") {
    document.body.classList.add("dark-theme");
    themeBtn.textContent = "Light";
  }
});
