
# 🤖 BERT Fine-Tuning for Emotion Detection

This repository contains the training pipeline for a BERT-based emotion classifier using the [SMILE dataset](https://github.com/huggingface/datasets/blob/main/datasets/smile/README.md). The resulting model was used in my deployed [Multimodal Emotion Classifier](https://github.com/sshweta13/Multimodal-Emotion-Classifier), which performs real-time emotion recognition using both text and facial expressions.

---

## 🚀 Project Overview

This project demonstrates how to:
- Preprocess and tokenize emotion-labeled text
- Fine-tune a BERT model (`bert-base-uncased`) using Hugging Face Transformers
- Evaluate model performance for emotion classification
- Save and export the model for later deployment

---

## 🧠 Technologies Used

- Python 🐍
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas
- Google Colab (for training environment)

---

## 📂 Project Structure

```
bert-emotion-training/
├── Sentiment_Analysis_with_Deep_Learning_using_BERT.ipynb   # Main training notebook
├── sentiment_analysis_with_deep_learning_using_bert.py      # Script version
├── smile-annotations-final.csv                              # Training dataset
├── model_output/                                            # Optional: for saved model
├── requirements.txt                                         # Package dependencies
└── README.md
```

---

## 🧪 Model Summary

| Property           | Value                  |
|--------------------|------------------------|
| Base Model         | `bert-base-uncased`    |
| Dataset            | SMILE                  |
| Task               | Emotion classification |
| Output Labels      | happy, sad, angry, fear, neutral |
| Evaluation Metric  | Accuracy, F1-score     |

---

## 🔗 Related Work

🎭 This model is integrated into the **Multimodal Emotion Classifier** app:  
[GitHub Repo](https://github.com/sshweta13/Multimodal-Emotion-Classifier)  
[Live Demo (Hugging Face)](https://huggingface.co/spaces/sshweta13/Multimodal_Emotion_Classifier)

---

## 📈 Future Improvements

- Integrate emotion intensity scoring
- Explore multilingual fine-tuning (e.g., BERT multilingual)
- Upload model weights to Hugging Face Model Hub

---

## 📜 License

This project is shared for educational and learning purposes. Contact me if you'd like to collaborate or build upon it.
