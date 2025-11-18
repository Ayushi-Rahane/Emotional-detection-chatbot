# ✨ Emotion Detection Chatbot 🤖
An AI-powered chatbot that detects human emotions from text and responds in an empathetic, context-aware manner.

---

## 🎯 Project Title & Objective
**Emotion Detection Chatbot**  
To build a real-time system that classifies user text into 7 emotions *joy, sadness, anger, fear, disgust, surprise, neutral*  using advanced NLP and Deep Learning techniques.

---

## 📂 Dataset Details
- Total Samples: **~20,000**
- Derived from multiple open-source emotion datasets
- Final consolidated file: `emotion_dataset.csv`
- Emotions balanced using stratified splitting
- Used for training, validation, and baseline testing

---

## 🧠 Algorithm / Model Used

### 🔹 Primary Model
- **DistilRoBERTa (j-hartmann/emotion-english-distilroberta-base)**
- Accuracy: **~92%**
- Libraries: Hugging Face Transformers, PyTorch

### 🔹 Supporting Techniques
- **TextBlob** for sentiment polarity
- **Rule-based refinement** for edge cases
- **Markov Chains** for emotion transitions
- **K-Means Clustering + PCA** for visualization and pattern analysis

### 🔹 Baseline Model
- **TF-IDF + Logistic Regression**
- Accuracy: **~75%**

---

## 📊 Results

### 📌 Performance Metrics
| Metric | Score |
|--------|--------|
| Accuracy | ~92% |
| Macro F1-Score | ~0.90 |
| Weighted F1 | ~0.91 |

### 📌 Highlights
- Best performance on: **joy, sadness, neutral**
- Moderate confusion between **fear** and **surprise**
- **Disgust** is hardest due to fewer samples

### 📌 Visual Outputs (Generated During Project)
- PCA-based emotion clusters  
- Emotion frequency graphs  
- Transition matrices  
- Interactive statistics dashboard

---

## 🏁 Conclusion
The chatbot successfully demonstrates how transformer-based models can understand subtle emotional cues in text.  
Integrating NLP, ML, sentiment analysis, and Markov modeling results in accurate, real-time emotion-aware conversations.

---

## 🚀 Future Scope
- Multilingual emotion detection  
- Voice + text-based emotion recognition  
- Cloud deployment for scalable usage  
- Fine-tuning with domain-specific data  
- Personalized responses using emotion history  
- Remove rule-based logic and use small LLM refinement layer  

---

## 📚 References
- Hugging Face Transformers  
- DistilRoBERTa emotion model by j-hartmann  
- NLTK, TextBlob documentation  
- “Attention Is All You Need” — Transformer Architecture  
- AIML Lab Hackathon Resources  

