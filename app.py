# ============================================================
# app.py  —  STREAMLIT FRONTEND FOR HEART DISEASE PREDICTION
# ============================================================

import streamlit as st
import pandas as pd

from backend import (
    MODEL_PATH,
    evaluate_model,
    load_data,
    load_saved_model,
    plot_age_thalach_scatter,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_roc_curve,
    plot_single_tree,
    preprocess_data,
    split_data,
    train_and_save_model,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

# ============================================================
# CUSTOM DARK THEME CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0d0d1a, #1a0a0a);
    color: white;
}

h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

[data-testid="stSidebar"] {
    background-color: #110808;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #c0392b, #e74c3c);
    color: white;
    border: none;
    border-radius: 10px;
    height: 3rem;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(to right, #922b21, #c0392b);
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}

.success-box {
    background-color: rgba(46, 204, 113, 0.2);
    border: 2px solid #2ecc71;
}

.danger-box {
    background-color: rgba(231, 76, 60, 0.2);
    border: 2px solid #e74c3c;
}

.metric-card {
    background-color: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown("""
<h1 style='text-align:center;'>
❤️ Heart Disease Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px; color:#cccccc;'>
Enter patient health information to predict heart disease risk.
</p>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return load_saved_model()
    model, _ = train_and_save_model()
    return model


try:
    model = get_model()
except Exception as exc:
    st.error("Model not ready. Run `python backend.py` once first.")
    st.exception(exc)
    st.stop()


@st.cache_resource
def get_graph_data():
    df = load_data()
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean, 20, 42)
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    return df, df_clean, results

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## 📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=20, max_value=85, value=45)
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 420, 200)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])

with col2:
    sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
    cp = st.selectbox("Chest Pain Type", [
        "0 - Typical Angina",
        "1 - Atypical Angina",
        "2 - Non-Anginal Pain",
        "3 - Asymptomatic",
    ])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
    restecg = st.selectbox("Resting ECG", ["Normal (0)", "ST Abnormality (1)", "LV Hypertrophy (2)"])
    exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
    slope = st.selectbox("Slope of Peak ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
    thal = st.selectbox("Thalassemia", ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Other (3)"])

# ============================================================
# ENCODING
# ============================================================

sex_enc     = 0 if sex.startswith("Female") else 1
cp_enc      = int(cp[0])
fbs_enc     = 0 if fbs.startswith("No") else 1
restecg_enc = int(restecg[0])
exang_enc   = 0 if exang.startswith("No") else 1
slope_enc   = int(slope[0])
thal_enc    = int(thal[0])

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict Heart Disease Risk"):

    input_data = pd.DataFrame([{
        "age": age, "sex": sex_enc, "cp": cp_enc,
        "trestbps": trestbps, "chol": chol, "fbs": fbs_enc,
        "restecg": restecg_enc, "thalach": thalach, "exang": exang_enc,
        "oldpeak": oldpeak, "slope": slope_enc, "ca": ca, "thal": thal_enc,
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box danger-box">
            🚨 High Risk of Heart Disease<br><br>
            Probability: {probability:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box success-box">
            ✅ Low Risk of Heart Disease<br><br>
            Probability: {probability:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # METRICS CARDS
    st.markdown("## 📈 Health Metrics")

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in [
        (m1, "Age", age), (m2, "Cholesterol", chol),
        (m3, "BP", trestbps), (m4, "Max HR", thalach)
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <h2>{val}</h2>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# GRAPHS
# ============================================================

st.markdown("## 📊 Model Analytics")

try:
    df, df_clean, results = get_graph_data()

    tabs = st.tabs([
        "Class Distribution", "Correlation", "Confusion Matrix",
        "Feature Importance", "ROC Curve", "Age vs Heart Rate", "Decision Tree"
    ])

    with tabs[0]:
        st.pyplot(plot_class_distribution(df), use_container_width=True)
    with tabs[1]:
        st.pyplot(plot_correlation_heatmap(df_clean), use_container_width=True)
    with tabs[2]:
        st.pyplot(plot_confusion_matrix(results["conf_matrix"]), use_container_width=True)
    with tabs[3]:
        feat_fig, _, _ = plot_feature_importance(model)
        st.pyplot(feat_fig, use_container_width=True)
    with tabs[4]:
        st.pyplot(plot_roc_curve(results["roc_fpr"], results["roc_tpr"], results["auc_score"]),
                  use_container_width=True)
    with tabs[5]:
        st.pyplot(plot_age_thalach_scatter(df_clean), use_container_width=True)
    with tabs[6]:
        st.pyplot(plot_single_tree(model), use_container_width=True)

except Exception as exc:
    st.warning("Graphs not available yet. Run `python backend.py` first.")
    st.caption(str(exc))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#aaaaaa;'>
Built with ❤️ using Streamlit + Random Forest Machine Learning
</div>
""", unsafe_allow_html=True)
