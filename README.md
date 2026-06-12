# 🎬 Movie Review Sentiment Analysis

A web-based Sentiment Analysis application built using **PyTorch-rnn**, **Flask**, and **TF-IDF Vectorization**. The model predicts whether a movie review expresses a **Positive** or **Negative** sentiment.

## 🚀 Live Demo

🔗 https://movie-review-sentiment-analysis-crwf.onrender.com

---

## 📌 Features

- Predicts sentiment of movie reviews
- PyTorch-based Recurrent Neural Network (RNN)
- TF-IDF feature extraction
- Flask web application
- Interactive user interface
- Deployed on Render

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Flask
- Scikit-learn
- NLTK
- Pandas
- HTML/CSS
- Render

---

## 📂 Dataset

This project uses the **IMDB Movie Review Dataset** containing 50,000 movie reviews labeled as:

- Positive
- Negative

---

## 🧠 Model Architecture

```text
Movie Review
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
PyTorch RNN
      ↓
Sigmoid Activation
      ↓
Positive / Negative
```

---

## 📁 Project Structure

```text
movie-review-sentiment-analysis/
│
├── app.py
├── rnn_model.pth
├── tfidf_vectorizer.pkl
├── requirements.txt
├── Procfile
├── RNN_For_Sentiment_Analysis.ipynb
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/ritesh-kr18/movie-review-sentiment-analysis.git

cd movie-review-sentiment-analysis
```

### Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📝 Sample Inputs

### Positive Review

```text
This movie was amazing and I loved every minute of it.
```

Output:

```text
Positive 😊
```

### Negative Review

```text
This movie was terrible and a complete waste of time.
```

Output:

```text
Negative 😞
```

---

## 🌐 Deployment

The application is deployed on **Render** using:

- Flask
- Gunicorn
- Render Web Services

---

## 📈 Future Improvements

- Replace TF-IDF with Word Embeddings
- Implement LSTM/GRU architecture
- Improve generalization beyond movie reviews
- Support multi-class sentiment analysis

---

## 👨‍💻 Author

**Ritesh Kumar**

GitHub: https://github.com/ritesh-kr18

---
