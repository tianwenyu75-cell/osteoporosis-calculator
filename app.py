import streamlit as st
import numpy as np

# ============================================================
# 页面设置
# ============================================================
st.set_page_config(
    page_title="Osteoporosis Risk Calculator",
    page_icon="🦴",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a3a5c; text-align: center; }
    .subtitle { font-size: 1.0rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .result-box { background: #f0f4fa; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border-left: 5px solid #2E86AB; }
    .risk-high { color: #c0392b; font-weight: 700; font-size: 2.8rem; }
    .risk-moderate { color: #e67e22; font-weight: 700; font-size: 2.8rem; }
    .risk-low { color: #27ae60; font-weight: 700; font-size: 2.8rem; }
    .contrib-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin: 4px 0; }
    .disclaimer { background: #fff3cd; padding: 0.8rem 1.2rem; border-radius: 8px; border-left: 4px solid #ffc107; font-size: 0.85rem; margin: 1rem 0; }
    .footer { text-align: center; color: #999; font-size: 0.75rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🦴 Explainable Osteoporosis<br>Clinical Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An Explainable Osteoimmune AI Framework for Opportunistic Risk Stratification</div>', unsafe_allow_html=True)

# ============================================================
# 模型系数（来自你的逻辑回归）
# ============================================================
COEFS = {
    'intercept': -2.135957,
    'BMXBMI': -0.896414,
    'RIAGENDR': 0.769691,
    'RIDAGEYR': 0.682537,
    'RIDRETH1': -0.081783,
    'DMDEDUC2': 0.064456,
    'INDFMPIR': -0.104677,
    'NPAR': 0.259990,
    'SII': 0.018336,
    'PLR': -0.047041,
    'NLR': -0.075340,
    '吸烟_new': 0.029675,
    '饮酒_new': -0.129385,
    '高血压_new': -0.046589,
    '糖尿病_new': 0.053170,
    '关节炎_new': -0.024440
}

FEATURE_NAMES = [
    'BMXBMI', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2',
    'INDFMPIR', 'NPAR', 'SII', 'PLR', 'NLR',
    '吸烟_new', '饮酒_new', '高血压_new', '糖尿病_new', '关节炎_new'
]

# 标准化参数（来自你的 scaler）
TRAIN_MEANS = {
    'BMXBMI': 28.696, 'RIAGENDR': 1.4788, 'RIDAGEYR': 63.825,
    'RIDRETH1': 3.0356, 'DMDEDUC2': 3.2473, 'INDFMPIR': 2.6343,
    'NPAR': 13.938, 'SII': 533.55, 'PLR': 127.88, 'NLR': 2.1958,
    '吸烟_new': 0.4990, '饮酒_new': 0.4693, '高血压_new': 0.5182,
    '糖尿病_new': 0.1913, '关节炎_new': 0.4094
}

TRAIN_STDS = {
    'BMXBMI': 5.6667, 'RIAGENDR': 0.4996, 'RIDAGEYR': 9.2401,
    'RIDRETH1': 1.1252, 'DMDEDUC2': 1.3149, 'INDFMPIR': 1.5375,
    'NPAR': 2.5951, 'SII': 483.12, 'PLR': 54.183, 'NLR': 1.2798,
    '吸烟_new': 0.5000, '饮酒_new': 0.4991, '高血压_new': 0.4997,
    '糖尿病_new': 0.3933, '关节炎_new': 0.4917
}

def standardize_input(raw_values):
    std_values = []
    for i, name in enumerate(FEATURE_NAMES):
        std_values.append((raw_values[i] - TRAIN_MEANS[name]) / TRAIN_STDS[name])
    return np.array(std_values)

def calculate_risk(raw_values):
    std_values = standardize_input(raw_values)
    logit = COEFS['intercept']
    for i, name in enumerate(FEATURE_NAMES):
        logit += COEFS[name] * std_values[i]
    prob = 1 / (1 + np.exp(-logit))
    return prob

# ============================================================
# 输入区域
# ============================================================
st.markdown("### 📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 50, 90, 65)
    sex = st.radio("Sex", ["Female", "Male"], index=0)
    bmi = st.number_input("BMI (kg/m²)", 15.0, 45.0, 25.0, step=0.1)
    npar = st.number_input("NPAR", 5.0, 30.0, 14.0, step=0.1)
    nlr = st.number_input("NLR", 0.5, 10.0, 2.5, step=0.1)
    plr = st.number_input("PLR", 50, 400, 150, step=5)
    sii = st.number_input("SII", 100, 2000, 500, step=50)

with col2:
    smoking = st.selectbox("Smoking (≥100 cigarettes)", ["No", "Yes"])
    drinking = st.selectbox("Drinking (≥12 drinks/year)", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    arthritis = st.selectbox("Arthritis", ["No", "Yes"])
    race = st.selectbox("Race/Ethnicity", 
                        ["Mexican American", "Other Hispanic", "Non-Hispanic White", 
                         "Non-Hispanic Black", "Other Race"], index=2)
    education = st.selectbox("Education", ["< High School", "High School/GED", "≥ College"], index=1)
    income = st.number_input("Income-to-Poverty Ratio", 0.0, 5.0, 2.5, step=0.1)

# ============================================================
# 编码
# ============================================================
race_map = {"Mexican American": 1, "Other Hispanic": 2, "Non-Hispanic White": 3, 
            "Non-Hispanic Black": 4, "Other Race": 5}
education_map = {"< High School": 1, "High School/GED": 2, "≥ College": 3}

sex_encoded = 1 if sex == "Female" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
drinking_encoded = 1 if drinking == "Yes" else 0
hypertension_encoded = 1 if hypertension == "Yes" else 0
diabetes_encoded = 1 if diabetes == "Yes" else 0
arthritis_encoded = 1 if arthritis == "Yes" else 0
race_encoded = race_map[race]
education_encoded = education_map[education]

# ============================================================
# 计算
# ============================================================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calculate = st.button("🔄 Calculate Risk", use_container_width=True)

if calculate:
    raw_values = [
        bmi, sex_encoded, age, race_encoded, education_encoded,
        income, npar, sii, plr, nlr,
        smoking_encoded, drinking_encoded, hypertension_encoded,
        diabetes_encoded, arthritis_encoded
    ]
    
    prob = calculate_risk(raw_values) * 100
    
    if prob >= 30:
        risk_class, risk_color, recommendation, follow_up = "High", "risk-high", "⚠️ Consider DXA and comprehensive assessment", "3-6 months"
    elif prob >= 15:
        risk_class, risk_color, recommendation, follow_up = "Moderate", "risk-moderate", "📋 Consider DXA based on additional risk factors", "6-12 months"
    else:
        risk_class, risk_color, recommendation, follow_up = "Low", "risk-low", "✅ Routine follow-up. Lifestyle optimization recommended.", "12-24 months"
    
    st.markdown("---")
    st.markdown("### 📊 Risk Assessment Report")
    
    col_res1, col_res2, col_res3 = st.columns([1, 1.5, 1])
    
    with col_res1:
        st.markdown(f"""
        <div class="result-box" style="text-align:center;">
            <div style="font-size:0.9rem;color:#666;">Estimated Risk</div>
            <div class="{risk_color}">{prob:.1f}%</div>
            <div style="font-size:1.2rem;font-weight:600;">{risk_class} Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_res2:
        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:0.9rem;color:#666;">Recommendation</div>
            <div style="font-size:1.0rem;font-weight:500;">{recommendation}</div>
            <div style="font-size:0.9rem;color:#555;margin-top:0.3rem;">📅 Follow-up: {follow_up}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_res3:
        st.markdown(f"""
        <div class="result-box" style="background:#e8f4f8;">
            <div style="font-size:0.9rem;color:#666;">Interpretation</div>
            <div style="font-size:1.0rem;">{"🔴 High risk" if risk_class == "High" else "🟡 Moderate risk" if risk_class == "Moderate" else "🟢 Low risk"}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # SHAP 贡献
    st.markdown("### 🔍 Major Contributors")
    std_values = standardize_input(raw_values)
    contributions = []
    for i, name in enumerate(FEATURE_NAMES):
        contrib = COEFS[name] * std_values[i]
        if abs(contrib) > 0.01:
            contributions.append((name, contrib, "↑" if contrib > 0 else "↓"))
    
    contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
    max_abs = max([abs(c[1]) for c in contributions_sorted]) if contributions_sorted else 1
    
    for name, contrib, direction in contributions_sorted:
        pct = min(abs(contrib) / max_abs * 100, 100)
        color = '#c0392b' if direction == '↑' else '#27ae60'
        display_name = name.replace('_new', '').replace('BMX', '').replace('RIAGENDR', 'Sex (Female)').replace('RIDAGEYR', 'Age')
        st.markdown(f"""
        <div style="margin:6px 0;">
            <div class="contrib-label">
                <span>{display_name} {direction}</span>
                <span>{abs(contrib):.3f}</span>
            </div>
            <div style="background:#e9ecef;border-radius:10px;overflow:hidden;">
                <div style="width:{pct}%;background:{color};height:18px;border-radius:10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer">
        <strong>📌 Suggested Interpretation:</strong><br>
        • This tool is intended for <strong>research and clinical decision-support evaluation only</strong>.<br>
        • <strong>Not intended for standalone diagnosis or treatment decisions.</strong><br>
        • If osteopenia is already known, consider individualized fracture-risk assessment and follow-up.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <strong>Explainable Osteoimmune AI Framework</strong> • Based on NHANES 2005-2020 (n=10,189)<br>
    Logistic regression model (AUC=0.7891) • For research use only
</div>
""", unsafe_allow_html=True)