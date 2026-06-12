# app.py
import re
import pickle
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify, render_template_string

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"<.*?>", "", text)
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    tokens = [w for w in tokens if w not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

input_size = len(vectorizer.get_feature_names_out())  # 5000
model = RNN(input_size)
model.load_state_dict(torch.load("rnn_model.pth", map_location="cpu"))
model.eval()

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Sentiment Analyzer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .card {
      background: #1e293b;
      border-radius: 16px;
      padding: 40px;
      width: 600px;
      max-width: 95vw;
      box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }
    h1 { font-size: 1.6rem; margin-bottom: 8px; color: #f8fafc; }
    p.sub { font-size: 0.9rem; color: #94a3b8; margin-bottom: 28px; }
    textarea {
      width: 100%;
      height: 160px;
      padding: 14px;
      border-radius: 10px;
      border: 2px solid #334155;
      background: #0f172a;
      color: #e2e8f0;
      font-size: 1rem;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: #6366f1; }
    button {
      margin-top: 16px;
      width: 100%;
      padding: 14px;
      border: none;
      border-radius: 10px;
      background: #6366f1;
      color: white;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #4f46e5; }
    button:disabled { background: #475569; cursor: not-allowed; }
    #result {
      margin-top: 24px;
      padding: 20px;
      border-radius: 10px;
      text-align: center;
      font-size: 1.4rem;
      font-weight: 700;
      display: none;
    }
    .positive { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
    .negative { background: #4c0519; color: #fca5a5; border: 1px solid #dc2626; }
    #confidence { font-size: 0.85rem; font-weight: 400; margin-top: 6px; color: inherit; opacity: 0.8; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎬 Sentiment Analyzer</h1>
    <p class="sub">Type a movie review below to predict its sentiment.</p>
    <textarea id="review" placeholder="e.g. This movie was absolutely brilliant! The acting was top-notch..."></textarea>
    <button onclick="predict()" id="btn">Analyze Sentiment</button>
    <div id="result">
      <div id="label"></div>
      <div id="confidence"></div>
    </div>
  </div>

  <script>
    async function predict() {
      const text = document.getElementById("review").value.trim();
      if (!text) { alert("Please enter a review first."); return; }

      const btn = document.getElementById("btn");
      btn.disabled = true;
      btn.textContent = "Analyzing...";

      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await resp.json();

      const result = document.getElementById("result");
      const label = document.getElementById("label");
      const conf = document.getElementById("confidence");

      result.className = data.sentiment === "Positive" ? "positive" : "negative";
      label.textContent = data.sentiment === "Positive" ? "😊 Positive" : "😞 Negative";
      conf.textContent = `Confidence: ${data.confidence}%`;
      result.style.display = "block";

      btn.disabled = false;
      btn.textContent = "Analyze Sentiment";
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_text = data.get("text", "")

    cleaned = preprocess(raw_text)
    vector = vectorizer.transform([cleaned]).toarray()
    tensor = torch.from_numpy(vector).float().unsqueeze(1)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output.squeeze()).item()

    sentiment = "Positive" if prob > 0.5 else "Negative"
    confidence = round(prob * 100 if prob > 0.5 else (1 - prob) * 100, 1)

    return jsonify({"sentiment": sentiment, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)