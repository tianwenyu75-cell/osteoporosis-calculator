# ============================================================
# EO-CDSS: Explainable Osteoimmune Clinical Decision Support System
# Version 2.1（自动适配，无需 joblib）
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

# 尝试导入 joblib（如果可用则加载真实模型，否则使用内置系数）
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="EO-CDSS - Explainable Osteoimmune CDSS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 自定义CSS（保持不变，省略...）
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .eo-title { font-size: 3.5rem; font-weight: 800; color: #0a1628; text-align: center; letter-spacing: -1px; margin-bottom: 0.2rem; }
    .eo-subtitle { font-size: 1.2rem; color: #4a6a8b; text-align: center; font-weight: 400; letter-spacing: 4px; margin-bottom: 0.5rem; }
    .eo-tagline { font-size: 1rem; color: #7a8fa3; text-align: center; margin-bottom: 2.5rem; font-style: italic; }
    .flow-container { display: flex; justify-content: center; align-items: center; gap: 10px; margin: 2rem 0; flex-wrap: wrap; }
    .flow-box { background: white; padding: 12px 24px; border-radius: 10px; border: 1.5px solid #dde7f0; font-weight: 500; color: #1a3a5c; font-size: 0.9rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    .flow-arrow { color: #8aa3bc; font-size: 1.5rem; font-weight: 300; }
    .stButton > button { border-radius: 30px !important; font-weight: 600 !important; padding: 0.6rem 2.5rem !important; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(46, 134, 171, 0.3); }
    .result-card { background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #eef3f8; margin: 1rem 0; }
    .result-number { font-size: 3.5rem; font-weight: 800; text-align: center; }
    .result-label { font-size: 0.9rem; color: #7a8fa3; text-align: center; letter-spacing: 1px; }
    .risk-high { color: #c0392b; }
    .risk-moderate { color: #e67e22; }
    .risk-low { color: #27ae60; }
    .contrib-container { margin: 8px 0; }
    .contrib-label { display: flex; justify-content: space-between; font-size: 0.9rem; color: #2c3e50; margin-bottom: 2px; }
    .contrib-bar-bg { background: #e9ecef; border-radius: 8px; overflow: hidden; height: 20px; }
    .contrib-bar-fill { height: 20px; border-radius: 8px; transition: width 0.6s ease; }
    .disclaimer-box { background: #fff8e7; padding: 1rem 1.5rem; border-radius: 10px; border-left: 4px solid #f1c40f; font-size: 0.85rem; color: #5d4e37; margin: 1.5rem 0; }
    .disclaimer-box strong { color: #8a6d3b; }
    .eo-footer { text-align: center; color: #9aafc4; font-size: 0.75rem; padding-top: 2rem; margin-top: 3rem; border-top: 1px solid #e8edf3; }
    .step-indicator { display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem; }
    .step-dot { display: flex; align-items: center; gap: 8px; color: #9aafc4; font-size: 0.85rem; }
    .step-dot.active { color: #1a3a5c; font-weight: 600; }
    .step-dot .dot { width: 10px; height: 10px; border-radius: 50%; background: #dde7f0; }
    .step-dot.active .dot { background: #2E86AB; }
    .step-dot.done .dot { background: #27ae60; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State 初始化
# ============================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'result' not in st.session_state:
    st.session_state.result = None

# ============================================================
# 加载模型（自动适配）
# ============================================================
@st.cache_resource
def load_models():
    model = None
    scaler = None
    if HAS_JOBLIB:
        try:
            model = joblib.load('logistic_model.joblib')
            scaler = joblib.load('scaler.joblib')
        except:
            pass
    return model, scaler

model, scaler = load_models()

# 特征名称
FEATURE_NAMES = [
    'BMXBMI', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2',
    'INDFMPIR', 'NPAR', 'SII', 'PLR', 'NLR',
    '吸烟_new', '饮酒_new', '高血压_new', '糖尿病_new', '关节炎_new'
]

# 内置系数（备用）
COEFS_INTERCEPT = -2.135957
COEFS = [-0.896414, 0.769691, 0.682537, -0.081783, 0.064456,
         -0.104677, 0.259990, 0.018336, -0.047041, -0.075340,
         0.029675, -0.129385, -0.046589, 0.053170, -0.024440]
MEANS = [28.696, 1.4788, 63.825, 3.0356, 3.2473, 2.6343,
         13.938, 533.55, 127.88, 2.1958, 0.4990, 0.4693,
         0.5182, 0.1913, 0.4094]
STDS = [5.6667, 0.4996, 9.2401, 1.1252, 1.3149, 1.5375,
        2.5951, 483.12, 54.183, 1.2798, 0.5000, 0.4991,
        0.4997, 0.3933, 0.4917]

# ============================================================
# 计算函数
# ============================================================
def calculate_risk(input_dict):
    raw_values = [
        input_dict['bmi'], input_dict['sex'], input_dict['age'],
        input_dict['race'], input_dict['education'], input_dict['income'],
        input_dict['npar'], input_dict['sii'], input_dict['plr'], input_dict['nlr'],
        input_dict['smoking'], input_dict['drinking'], input_dict['hypertension'],
        input_dict['diabetes'], input_dict['arthritis']
    ]
    X_raw = np.array([raw_values])
    
    if scaler is not None:
        X_scaled = scaler.transform(X_raw)[0]
    else:
        X_scaled = (X_raw - MEANS) / STDS
        X_scaled = X_scaled[0]
    
    if model is not None:
        prob = model.predict_proba([X_scaled])[0][1]
    else:
        logit = COEFS_INTERCEPT + np.sum(COEFS[i] * X_scaled[i] for i in range(len(FEATURE_NAMES)))
        prob = 1 / (1 + np.exp(-logit))
    return prob

def get_risk_level(prob):
    if prob >= 0.30:
        return 'High', '🔴', 'Consider DXA and comprehensive assessment'
    elif prob >= 0.15:
        return 'Moderate', '🟡', 'Consider DXA based on additional risk factors'
    else:
        return 'Low', '🟢', 'Routine follow-up. Lifestyle optimization recommended.'

def get_shap_contributions(input_dict):
    raw_values = [
        input_dict['bmi'], input_dict['sex'], input_dict['age'],
        input_dict['race'], input_dict['education'], input_dict['income'],
        input_dict['npar'], input_dict['sii'], input_dict['plr'], input_dict['nlr'],
        input_dict['smoking'], input_dict['drinking'], input_dict['hypertension'],
        input_dict['diabetes'], input_dict['arthritis']
    ]
    X_raw = np.array([raw_values])
    if scaler is not None:
        X_scaled = scaler.transform(X_raw)[0]
    else:
        X_scaled = (X_raw - MEANS) / STDS
        X_scaled = X_scaled[0]
    
    display_names = ['BMI', 'Sex (Female)', 'Age', 'Race', 'Education',
                     'Income', 'NPAR', 'SII', 'PLR', 'NLR',
                     'Smoking', 'Drinking', 'Hypertension', 'Diabetes', 'Arthritis']
    contributions = []
    for i, (name, display) in enumerate(zip(FEATURE_NAMES, display_names)):
        contrib = COEFS[i] * X_scaled[i]
        if abs(contrib) > 0.005:
            direction = '↑' if contrib > 0 else '↓'
            contributions.append({'feature': display, 'value': contrib, 'direction': direction, 'abs_value': abs(contrib)})
    contributions.sort(key=lambda x: x['abs_value'], reverse=True)
    return contributions[:6]

# ============================================================
# 页面导航
# ============================================================
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# ============================================================
# 首页
# ============================================================
def show_home():
    st.markdown('<div class="eo-title">🏥 EO-CDSS</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-subtitle">EXPLAINABLE OSTEOIMMUNE</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-subtitle" style="font-size:1.4rem;margin-bottom:0;">Clinical Decision Support System</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-tagline">Based on NHANES 2005-2020 · n=10,189 · AUC=0.7891</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="flow-container">
        <div class="flow-box">📋 Routine Clinical Variables</div>
        <span class="flow-arrow">→</span>
        <div class="flow-box">🤖 Explainable AI</div>
        <span class="flow-arrow">→</span>
        <div class="flow-box">📊 Clinical Recommendation</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div style="text-align:center;padding:1rem;"><div style="font-size:2rem;">📋</div><div style="font-weight:600;color:#1a3a5c;">Routine Variables</div><div style="font-size:0.85rem;color:#7a8fa3;">Age, BMI, Sex, NPAR, CBC</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style="text-align:center;padding:1rem;"><div style="font-size:2rem;">🔍</div><div style="font-weight:600;color:#1a3a5c;">Explainable</div><div style="font-size:0.85rem;color:#7a8fa3;">SHAP-based contributions</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div style="text-align:center;padding:1rem;"><div style="font-size:2rem;">📊</div><div style="font-weight:600;color:#1a3a5c;">Clinical Decision Support</div><div style="font-size:0.85rem;color:#7a8fa3;">DXA prioritization</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
    with col_btn2:
        if st.button("🚀 Start Assessment", use_container_width=True, type="primary"):
            navigate_to('input')
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Use Notice</strong><br>
        This system is intended for <strong>clinical decision-support evaluation</strong> and 
        does not replace DXA or physician judgment. Always consider the patient's complete 
        clinical picture and local guidelines.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="eo-footer">EO-CDSS v2.1 · For research and clinical evaluation purposes only</div>', unsafe_allow_html=True)

# ============================================================
# 输入页（省略，内容与之前相同，保证完整）
# ============================================================
def show_input():
    st.markdown("""
    <div class="step-indicator">
        <div class="step-dot done"><span class="dot"></span> Home</div>
        <div class="step-dot active"><span class="dot"></span> Patient Info</div>
        <div class="step-dot"><span class="dot"></span> Results</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## 📋 Patient Information")
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Demographics**")
            age = st.slider("Age (years)", 50, 90, 65)
            sex = st.radio("Sex", ["Female", "Male"], index=0, horizontal=True)
            bmi = st.number_input("BMI (kg/m²)", 15.0, 45.0, 25.0, step=0.1)
            race = st.selectbox("Race/Ethnicity", ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American", "Other Hispanic", "Other Race"], index=0)
            education = st.selectbox("Education", ["≥ College", "High School/GED", "< High School"], index=0)
            income = st.number_input("Income-to-Poverty Ratio", 0.0, 5.0, 2.5, step=0.1)
        with col2:
            st.markdown("**Laboratory Markers**")
            npar = st.number_input("NPAR", 5.0, 30.0, 14.0, step=0.1)
            nlr = st.number_input("NLR", 0.5, 10.0, 2.5, step=0.1)
            plr = st.number_input("PLR", 50, 400, 150, step=5)
            sii = st.number_input("SII", 100, 2000, 500, step=50)
            st.markdown("**Lifestyle & Comorbidities**")
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            drinking = st.selectbox("Drinking", ["No", "Yes"])
            col_h = st.columns(3)
            with col_h[0]: hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            with col_h[1]: diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            with col_h[2]: arthritis = st.selectbox("Arthritis", ["No", "Yes"])
        st.markdown("---")
        st.markdown("### 🦴 Osteopenia Status")
        st.markdown("*If the patient has known osteopenia (T-score between -1.0 and -2.5)*")
        known_osteopenia = st.radio("Known Osteopenia?", ["No", "Yes"], index=0, horizontal=True)
        if known_osteopenia == "Yes":
            st.info("📌 **Clinical Note**: Patients with known osteopenia may benefit from earlier DXA reassessment and individualized preventive management.")
        st.markdown("---")
        col_btn = st.columns([1, 1.5, 1])
        with col_btn[1]:
            submitted = st.form_submit_button("📊 Generate Assessment", use_container_width=True, type="primary")
        if submitted:
            sex_enc = 1 if sex == "Female" else 0
            smk = 1 if smoking == "Yes" else 0
            drk = 1 if drinking == "Yes" else 0
            hyp = 1 if hypertension == "Yes" else 0
            dm = 1 if diabetes == "Yes" else 0
            arth = 1 if arthritis == "Yes" else 0
            race_map = {"Non-Hispanic White":3, "Non-Hispanic Black":4, "Mexican American":1, "Other Hispanic":2, "Other Race":5}
            edu_map = {"≥ College":3, "High School/GED":2, "< High School":1}
            st.session_state.patient_data = {
                'age':age, 'sex':sex_enc, 'sex_display':sex, 'bmi':bmi,
                'race':race_map[race], 'race_display':race,
                'education':edu_map[education], 'education_display':education,
                'income':income, 'npar':npar, 'nlr':nlr, 'plr':plr, 'sii':sii,
                'smoking':smk, 'smoking_display':smoking,
                'drinking':drk, 'drinking_display':drinking,
                'hypertension':hyp, 'hypertension_display':hypertension,
                'diabetes':dm, 'diabetes_display':diabetes,
                'arthritis':arth, 'arthritis_display':arthritis,
                'known_osteopenia':known_osteopenia
            }
            prob = calculate_risk(st.session_state.patient_data)
            st.session_state.result = prob
            navigate_to('result')

# ============================================================
# 结果页（省略，保留完整功能）
# ============================================================
def show_result():
    st.markdown("""
    <div class="step-indicator">
        <div class="step-dot done"><span class="dot"></span> Home</div>
        <div class="step-dot done"><span class="dot"></span> Patient Info</div>
        <div class="step-dot active"><span class="dot"></span> Results</div>
    </div>
    """, unsafe_allow_html=True)
    data = st.session_state.patient_data
    prob = st.session_state.result
    if prob is None:
        st.warning("No assessment found.")
        if st.button("Start New Assessment"):
            navigate_to('home')
        return
    risk_level, risk_icon, rec = get_risk_level(prob)
    risk_pct = prob*100
    st.markdown("## 📊 AI Osteoporosis Assessment Report")
    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div><div class="result-label">Estimated Osteoporosis Probability</div>
            <div class="result-number risk-{risk_level.lower()}">{risk_pct:.1f}%</div></div>
            <div style="text-align:center;"><div style="font-size:3rem;">{risk_icon}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{'#c0392b' if risk_level=='High' else '#e67e22' if risk_level=='Moderate' else '#27ae60'};">{risk_level} Risk</div></div>
            <div style="max-width:300px;"><div style="font-size:0.85rem;color:#7a8fa3;">Clinical Recommendation</div>
            <div style="font-weight:500;color:#1a3a5c;">{rec}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🔍 Key Contributors (SHAP)")
    contribs = get_shap_contributions(data)
    if contribs:
        max_abs = max(c['abs_value'] for c in contribs)
        for c in contribs:
            pct = (c['abs_value']/max_abs*100) if max_abs>0 else 0
            color = '#c0392b' if c['direction']=='↑' else '#27ae60'
            st.markdown(f"""
            <div class="contrib-container">
                <div class="contrib-label"><span><strong>{c['feature']}</strong> {c['direction']}</span><span>{c['abs_value']:.3f}</span></div>
                <div class="contrib-bar-bg"><div class="contrib-bar-fill" style="width:{pct}%;background:{color};"></div></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No major contributors identified.")
    if data['known_osteopenia'] == "Yes":
        st.markdown("### 🦴 Osteopenia-Specific Advice")
        st.info("Patients with known osteopenia may benefit from earlier DXA reassessment and individualized preventive management. This system does not predict progression.")
    st.markdown("### 💡 Clinical Decision Support")
    st.markdown(f"""
    <div class="result-card">
        <ul style="list-style-type:none;padding-left:0;margin:0;">
            <li>📌 <strong>Risk Level:</strong> {risk_icon} {risk_level} ({risk_pct:.1f}%)</li>
            <li>📋 <strong>Recommendation:</strong> {rec}</li>
            <li>📅 <strong>Suggested Follow-up:</strong> {'3-6 months' if risk_level=='High' else '6-12 months' if risk_level=='Moderate' else '12-24 months'}</li>
            <li style="margin-top:8px;color:#7a8fa3;font-size:0.9rem;">⚠️ This system is intended for <strong>clinical decision-support evaluation</strong> and does not replace DXA or physician judgment.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 New Assessment", use_container_width=True): navigate_to('input')
    with col2:
        # 简单PDF下载（不依赖reportlab）
        if st.button("📄 Download Report (TXT)", use_container_width=True):
            text = f"EO-CDSS Report\nGenerated: {datetime.now()}\n\nPatient: Age {data['age']}, {data['sex_display']}, BMI {data['bmi']}\nRisk: {risk_pct:.1f}% ({risk_level})\nRecommendation: {rec}\n\nContributors:\n" + "\n".join([f"{c['feature']}: {c['direction']} ({c['abs_value']:.3f})" for c in contribs]) + "\n\nDisclaimer: This system is for clinical decision-support evaluation only."
            st.download_button("📥 Download TXT", text, file_name=f"EO-CDSS_Report_{datetime.now().strftime('%Y%m%d')}.txt")
    with col3:
        if st.button("🏠 Home", use_container_width=True): navigate_to('home')

# ============================================================
# 主路由
# ============================================================
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'input':
    show_input()
elif st.session_state.page == 'result':
    show_result()