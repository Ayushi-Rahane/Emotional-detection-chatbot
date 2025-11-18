# Emotion Detection Chatbot  


A chatbot designed to understand emotional tone in text and respond with context-aware, empathetic messages.  
Built using transformer models, sentiment analysis, and basic conversational state tracking.

---

## 🎯 Project Objective  

**Emotion Detection Chatbot**

This project aims to overcome the limitations of typical chatbots that fail to detect mood shifts.  
The goals include:

- Detect emotions accurately with DistilRoBERTa  
- Refine predictions using sentiment polarity  
- Track mood transitions through the conversation  
- Generate emotionally consistent replies  
- Visualize emotional behaviour patterns  

---

## 📂 Dataset  


- 2000 unseen text entries  
- Six emotion categories: **anger, fear, joy, love, sadness, surprise**  
- Stored in a simple `text ; label` format  
- Used for both training and baseline testing  

---

## 🧠 Models & Methods  

### **Primary Model**
- DistilRoBERTa (j-hartmann/emotion-english-distilroberta-base)  
- Accuracy: **~84.1%**  
- Libraries: Transformers, PyTorch  

### **Supporting Techniques**
- Sentiment polarity → **TextBlob**  
- Edge-case refinement → rule-based logic  
- Transition modeling → **Markov Chains**  
- Pattern analysis → **K-means + PCA**  

### **Baseline**
- TF-IDF + Logistic Regression  
- Accuracy around **86.9%**

---

## 📊 Results  
### **Metrics**
- Accuracy: 84.1%  
- Weighted F1: 0.876  
- Macro F1: 0.838  

### **Class-wise Performance**
- Sadness: 0.907  
- Joy: 0.877  
- Fear: 0.832  
- Anger: 0.845  

The model handles subtle emotional cues better than simple keyword-based systems.

### **Visual Outputs**
- PCA emotion clusters  
- Emotion frequency graphs  
- Transition matrices  
- Interactive mini-dashboard  

---

## 🏁 Conclusion  

This chatbot shows how transformer-based models can effectively capture emotional signals in text.  
By combining NLP, ML, sentiment checks, and transition tracking, it maintains emotional continuity in conversation and produces reliable predictions.

---

## 🚀 Future Enhancements  

- Multimodal emotion detection (voice + face)  
- Domain-specific fine-tuning  
- Removing rule-based steps with dynamic AI response layers  
- Web & mobile deployment  
- Emotional intensity score (0–100)  
- Mood-based recommendations  
- Long-term emotion analytics for individuals  

---

## 📚 References  

- A. Pophale, S. Gite, A. Thombre, 2021 – Emotion recognition using chatbot system.
- J. Antony, S. G. Sudha, R. Prabha, 2021 – Emotion recognition-based mental healthcare chatbots: A survey. Link
- P. Zhong, D. Wang, C. Miao, 2019 – Knowledge-enriched transformer for emotion detection in textual conversations. Link
- C. Cortiz, 2021 – Exploring transformers in emotion recognition: Comparison of BERT, DistilBERT, RoBERTa, XLNet and ELECTRA. Link
- L. Bulla, T. Biesialska, J. D. Williams, M. Wiegand, 2023 – Towards distribution-shift robust text classification of emotions. ACL 2023.

