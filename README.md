# ❤️ Heart Disease Prediction — Random Forest Project

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── backend.py          ← ML brain: generate data, train model, evaluate, plot
├── app.py              ← Streamlit web UI
├── requirements.txt    ← Python libraries needed
├── data/
│   └── heart_disease.csv    ← Auto-generated dataset (10,000 patients)
├── models/
│   └── heart_model.joblib   ← Saved trained model
└── README.md           ← This guide
```

---

## 🚀 Getting Started — Step by Step

---

### ✅ STEP 1 — Install Python & VSCode

**Python:**
1. Download from https://python.org/downloads
2. During install → ✅ check **"Add Python to PATH"**

**VSCode:**
1. Download from https://code.visualstudio.com
2. Install the **Python** extension from Extensions panel

---

### ✅ STEP 2 — Clone this Repository

Open terminal and run:
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

---

### ✅ STEP 3 — Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

---

### ✅ STEP 4 — Install Libraries

```bash
pip install -r requirements.txt
```

---

### ✅ STEP 5 — Train the Model

```bash
python backend.py
```

Expected output:
```
✅ Dataset generated: data/heart_disease.csv  (10000 rows)
📊 Loaded: 10000 rows × 14 columns
✅ After cleaning: 10000 rows remain
✂️  Split: 8000 train | 2000 test
🌲 Training Random Forest (100 trees, depth 10)...
✅ Training complete!
💾 Model saved → models/heart_model.joblib

   Test Accuracy : 97.3%
   Train Accuracy: 99.1%
   AUC Score     : 98.7%

✅  ALL STEPS PASSED!
👉  Now run:  streamlit run app.py
```

---

### ✅ STEP 6 — Run the App

```bash
streamlit run app.py
```

Browser opens at **http://localhost:8501** 🎉

---

## 📊 Dataset Features

| Column | Type | Meaning |
|--------|------|---------|
| `age` | Number | Patient age (25–80) |
| `sex` | 0 or 1 | 0 = Female, 1 = Male |
| `cp` | 0–3 | Chest pain type |
| `trestbps` | Number | Resting blood pressure (mmHg) |
| `chol` | Number | Serum cholesterol (mg/dl) |
| `fbs` | 0 or 1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | 0–2 | Resting ECG results |
| `thalach` | Number | Maximum heart rate achieved |
| `exang` | 0 or 1 | Exercise induced angina |
| `oldpeak` | Float | ST depression induced by exercise |
| `slope` | 0–2 | Slope of peak exercise ST segment |
| `ca` | 0–3 | Number of major vessels colored by fluoroscopy |
| `thal` | 0–3 | Thalassemia type |
| `target` | 0 or 1 | **TARGET** — Heart disease (1) or Healthy (0) |

---

## 🧪 Experiments for Students

| Experiment | What to change | What to observe |
|------------|---------------|-----------------|
| More trees | n_estimators 10 → 200 | Accuracy vs training time |
| Depth limit | max_depth 3 → 20 | Overfitting gap changes |
| Less data | test_size 10% → 40% | Does accuracy drop? |
| Feature importance | Check the bar chart | Which feature matters most? |

---

## ❓ Common Problems & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No saved model` error | Run `python backend.py` first |
| Port in use | Run `streamlit run app.py --server.port 8502` |

---

## 🔗 Difference from Diabetes Project

| Feature | Diabetes Project | Heart Disease Project |
|---------|-----------------|----------------------|
| Dataset | Kaggle API | Auto-generated (no API key needed!) |
| Target | Diabetes (0/1) | Heart Disease (0/1) |
| Features | 8 features | 13 features |
| Extra | Kaggle credentials needed | Works immediately |

---

*Built with Python · scikit-learn · Streamlit · pandas · matplotlib · seaborn*
