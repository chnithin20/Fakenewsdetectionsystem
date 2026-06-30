# 📰 VerifyAI - AI-Powered Fake News Detection System

<div align="center">




\

**An Explainable AI-powered Fake News Detection System using RoBERTa, Google Fact Check API, and Source Credibility Analysis**

</div>

---

# 📑 Table of Contents

* Project Overview
* Key Features
* System Architecture
* Technology Stack
* Project Structure
* Dataset
* Machine Learning Pipeline
* Detection Workflow
* Performance Evaluation
* Explainability
* REST API
* Installation
* Environment Variables
* Running the Project
* Training & Evaluation
* Sample Output
* Challenges
* Future Improvements
* Contributors
* License

---

# 🚀 Project Overview

VerifyAI is a modern AI-powered misinformation detection platform designed to classify news articles, headlines, and online news URLs using Natural Language Processing and external credibility verification.

Instead of relying solely on a machine learning model, VerifyAI combines **three independent verification signals**:

* 🤖 RoBERTa Deep Learning Classifier
* 🌐 Google Fact Check Tools API
* 📰 Domain Source Credibility Analysis

These three signals are combined into a weighted credibility engine that generates a final trust score and explainable verdict.

---

# ✨ Key Features

* Fine-tuned RoBERTa Transformer Model
* Explainable AI Predictions
* Fake News Classification
* Headline Analysis
* Full Article Analysis
* URL-based News Analysis
* Automatic Article Extraction
* Google Fact Check Integration
* Domain Reputation Analysis
* Weighted Credibility Score
* SQLite Analysis History
* REST API
* Interactive Dashboard
* Optional Redis Caching

---

# 🏗 System Architecture

```text
                  User Input
                       │
     ┌─────────────────┼──────────────────┐
     │                 │                  │
 Headline          Full Article          URL
     │                 │                  │
     └─────────────────┼──────────────────┘
                       │
              Article Extraction
                       │
                Text Preprocessing
                       │
                RoBERTa Tokenizer
                       │
                Fine-tuned RoBERTa
                       │
        ┌──────────────┼──────────────┐
        │              │              │
 Fact Check API   Source Analysis   NLP Score
        │              │              │
        └──────────────┼──────────────┘
                       │
          Weighted Credibility Engine
                       │
         Explainable Final Prediction
```

---

# 💻 Technology Stack

## Backend

* Python 3.10+
* Flask
* Flask-CORS
* SQLAlchemy
* Pydantic

## Machine Learning

* Hugging Face Transformers
* RoBERTa
* PyTorch
* Scikit-Learn
* NLTK
* Pandas
* NumPy

## Web Scraping

* BeautifulSoup4
* newspaper3k
* lxml

## Database

* SQLite
* Redis (Optional)

## APIs

* Google Fact Check Tools API

---

# 📂 Project Structure

```text
VerifyAI/
│
├── backend/
│   ├── data/                         # Train, Validation & Test splits
│   ├── dataset/                      # Fake.csv & True.csv
│   ├── model_cache/                  # HuggingFace downloaded models
│   ├── my-fakenews-model/            # Fine-tuned RoBERTa Model
│   ├── services/                     # NLP & API Services
│   ├── train.py
│   ├── evaluate.py
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   └── .env
│
├── docs/
├── tmp-train-smoke/
├── tmp-train-smoke2/
├── tmp-training-checkpoints/
│
├── fake-news-detector.html
├── test.json
├── verifyai.db
└── README.md
```

---

# 📊 Dataset

The model is trained using two balanced datasets.

| Dataset  | Description           |
| -------- | --------------------- |
| Fake.csv | Fake News Articles    |
| True.csv | Genuine News Articles |

Dataset Split

```text
Training      : 80%

Validation    : 10%

Testing        : 10%
```

---

# 🧠 Machine Learning Pipeline

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Text Normalization
      │
      ▼
RoBERTa Tokenization
      │
      ▼
Fine-Tuning
      │
      ▼
Validation
      │
      ▼
Testing
      │
      ▼
Model Saving
```

---

# 🔍 Fake News Detection Workflow

```text
User Input
     │
     ▼
Article Extraction
     │
     ▼
Cleaning
     │
     ▼
RoBERTa Prediction
     │
     ▼
Fact Check API
     │
     ▼
Source Credibility
     │
     ▼
Weighted Credibility
     │
     ▼
Explainable Result
```

---

# 📈 Performance Evaluation

## Current Baseline Results

| Metric            | Score     |
| ----------------- | --------- |
| Accuracy          | **88.0%** |
| Macro Precision   | **85.0%** |
| Macro Recall      | **82.0%** |
| Weighted F1-Score | **80.0%** |

### Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| REAL  | 1.00      | 0.74   | 0.81     |
| FAKE  | 0.84      | 1.00   | 0.88     |

> **Current Status:** Baseline fine-tuned RoBERTa implementation. Performance is expected to improve through hyperparameter tuning, learning-rate scheduling, class balancing, and expanded training data.

---

# 🧩 Explainability Engine

The final credibility score is computed using three independent verification signals.

| Component              | Weight |
| ---------------------- | ------ |
| RoBERTa Classification | 50%    |
| Google Fact Check      | 30%    |
| Source Credibility     | 20%    |

```text
Final Credibility Score

=
0.50 × NLP Prediction

+
0.30 × Fact Check Score

+
0.20 × Source Reputation
```

---

# 🌐 REST API

| Method | Endpoint           | Description           |
| ------ | ------------------ | --------------------- |
| GET    | /health            | Health Check          |
| POST   | /api/analyze       | Analyze News          |
| GET    | /api/history       | View Analysis History |
| GET    | /api/analysis/{id} | Analysis Details      |

---

# ⚙ Installation

```bash
git clone https://github.com/yourusername/VerifyAI.git

cd VerifyAI/backend

pip install -r requirements.txt
```

---

# 🔑 Environment Variables

```env
DATABASE_URL=sqlite:///verifyai.db

GOOGLE_FACT_CHECK_API_KEY=

REDIS_URL=redis://localhost:6379/0

HF_MODEL_NAME=hamzab/roberta-fake-news-classification

WEIGHT_NLP_MODEL=0.50

WEIGHT_FACT_CHECK=0.30

WEIGHT_SOURCE_CREDIBILITY=0.20
```

---

# ▶ Running the Project

```bash
python main.py
```

Open your browser:

```text
http://localhost:8000
```

---

# 🏋 Model Training

Training

```bash
python train.py
```

Evaluation

```bash
python evaluate.py
```

---

# 📷 Example Prediction

```text
Headline

Government announces miracle cure overnight.

Prediction

FAKE NEWS

Confidence

96.8%

Reasons

✔ High NLP confidence

✔ No fact-check available

✔ Low source credibility

✔ Sensational wording detected

Credibility Score

18 / 100
```

---

# ⚠ Challenges & Limitations

* Limited dataset diversity may affect generalization.
* Performance depends on article extraction quality.
* Fact-check coverage varies by topic and region.
* Domain reputation alone cannot determine article authenticity.

---

# 🚀 Future Improvements

* Fine-tune on larger multilingual datasets
* Explainable AI (LIME / SHAP)
* Browser Extension
* Mobile Application
* Image-based Fake News Detection
* Social Media Verification
* Knowledge Graph Validation
* Continual Learning Pipeline
* Docker Deployment
* CI/CD Integration

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 👨‍💻 Author

**Challa Naga Sai Nithin**

AI • Machine Learning • Full Stack Development

---

# 📄 License

This project is licensed under the MIT License.

---

# 🙏 Acknowledgements

* Hugging Face Transformers
* PyTorch
* Google Fact Check Tools API
* Flask
* BeautifulSoup
* newspaper3k
* NLTK
* Scikit-Learn
* Pandas
* NumPy

---

⭐ **If you found this project useful, consider giving it a Star on GitHub!**
#   F a k e n e w s d e t e c t i o n s y s t e m  
 