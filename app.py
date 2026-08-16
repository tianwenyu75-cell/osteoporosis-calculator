# ============================================================
# EO-CDSS: Explainable Osteoimmune Clinical Decision Support System
# Version 2.0
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

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
# 自定义CSS
# ============================================================
st.markdown("""
<style>
    /* 全局 */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* 首页标题 */
    .eo-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #0a1628;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .eo-subtitle {
        font-size: 1.2rem;
        color: #4a6a8b;
        text-align: center;
        font-weight: 400;
        letter-spacing: 4px;
        margin-bottom: 0.5rem;
    }
    .eo-tagline {
        font-size: 1rem;
        color: #7a8fa3;
        text-align: center;
        margin-bottom: 2.5rem;
        font-style: italic;
    }
    
    /* 流程图 */
    .flow-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    .flow-box {
        background: white;
        padding: 12px 24px;
        border-radius: 10px;
        border: 1.5px solid #dde7f0;
        font-weight: 500;
        color: #1a3a5c;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .flow-arrow {
        color: #8aa3bc;
        font-size: 1.5rem;
        font-weight: 300;
    }
    
    /* 按钮 */
    .stButton > button {
        border-radius: 30px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2.5rem !important;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 134, 171, 0.3);
    }
    
    /* 结果卡片 */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #eef3f8;
        margin: 1rem 0;
    }
    .result-number {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
    }
    .result-label {
        font-size: 0.9rem;
        color: #7a8fa3;
        text-align: center;
        letter-spacing: 1px;
    }
    
    /* 风险标签 */
    .risk-high {
        color: #c0392b;
    }
    .risk-moderate {
        color: #e67e22;
    }
    .risk-low {
        color: #27ae60;
    }
    
    /* 贡献条 */
    .contrib-container {
        margin: 8px 0;
    }
    .contrib-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #2c3e50;
        margin-bottom: 2px;
    }
    .contrib-bar-bg {
        background: #e9ecef;
        border-radius: 8px;
        overflow: hidden;
        height: 20px;
    }
    .contrib-bar-fill {
        height: 20px;
        border-radius: 8px;
        transition: width 0.6s ease;
    }
    
    /* 免责声明 */
    .disclaimer-box {
        background: #fff8e7;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #f1c40f;
        font-size: 0.85rem;
        color: #5d4e37;
        margin: 1.5rem 0;
    }
    .disclaimer-box strong {
        color: #8a6d3b;
    }
    
    /* 页脚 */
    .eo-footer {
        text-align: center;
        color: #9aafc4;
        font-size: 0.75rem;
        padding-top: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e8edf3;
    }
    
    /* 导航指示器 */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .step-dot {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #9aafc4;
        font-size: 0.85rem;
    }
    .step-dot.active {
        color: #1a3a5c;
        font-weight: 600;
    }
    .step-dot .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #dde7f0;
    }
    .step-dot.active .dot {
        background: #2E86AB;
    }
    .step-dot.done .dot {
        background: #27ae60;
    }
    
    /* PDF下载按钮 */
    .pdf-btn {
        background: #1a3a5c;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 30px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    .pdf-btn:hover {
        background: #0a1628;
    }
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
# 加载模型和标准化器
# ============================================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('logistic_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        # 如果文件不存在，使用内置系数（备用方案）
        return None, None

model, scaler = load_models()

# 特征名称（必须与训练时一致）
FEATURE_NAMES = [
    'BMXBMI', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2',
    'INDFMPIR', 'NPAR', 'SII', 'PLR', 'NLR',
    '吸烟_new', '饮酒_new', '高血压_new', '糖尿病_new', '关节炎_new'
]

# ============================================================
# 工具函数
# ============================================================
def calculate_risk(input_dict):
    """使用真实模型计算风险概率"""
    # 构建特征向量（顺序必须与训练时一致）
    raw_values = [
        input_dict['bmi'],
        input_dict['sex'],
        input_dict['age'],
        input_dict['race'],
        input_dict['education'],
        input_dict['income'],
        input_dict['npar'],
        input_dict['sii'],
        input_dict['plr'],
        input_dict['nlr'],
        input_dict['smoking'],
        input_dict['drinking'],
        input_dict['hypertension'],
        input_dict['diabetes'],
        input_dict['arthritis']
    ]
    
    X_raw = np.array([raw_values])
    
    if scaler is not None:
        X_scaled = scaler.transform(X_raw)
    else:
        # 备用：手动标准化
        means = [28.696, 1.4788, 63.825, 3.0356, 3.2473, 2.6343,
                 13.938, 533.55, 127.88, 2.1958, 0.4990, 0.4693,
                 0.5182, 0.1913, 0.4094]
        stds = [5.6667, 0.4996, 9.2401, 1.1252, 1.3149, 1.5375,
                2.5951, 483.12, 54.183, 1.2798, 0.5000, 0.4991,
                0.4997, 0.3933, 0.4917]
        X_scaled = (X_raw - means) / stds
    
    if model is not None:
        prob = model.predict_proba(X_scaled)[0][1]
    else:
        # 备用：使用之前提取的系数
        coefs = [-2.135957, -0.896414, 0.769691, 0.682537, -0.081783,
                 0.064456, -0.104677, 0.259990, 0.018336, -0.047041,
                 -0.075340, 0.029675, -0.129385, -0.046589, 0.053170, -0.024440]
        logit = coefs[0] + np.sum(coefs[i+1] * X_scaled[0][i] for i in range(len(FEATURE_NAMES)))
        prob = 1 / (1 + np.exp(-logit))
    
    return prob

def get_risk_level(prob):
    if prob >= 0.30:
        return 'High', '🔴', 'Consider DXA and comprehensive assessment'
    elif prob >= 0.15:
        return 'Moderate', '🟡', 'Consider DXA based on additional risk factors'
    else:
        return 'Low', '🟢', 'Routine follow-up. Lifestyle optimization recommended.'

def get_shap_contributions(input_dict, prob):
    """计算SHAP风格的贡献（基于系数×标准化特征值）"""
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
        means = [28.696, 1.4788, 63.825, 3.0356, 3.2473, 2.6343,
                 13.938, 533.55, 127.88, 2.1958, 0.4990, 0.4693,
                 0.5182, 0.1913, 0.4094]
        stds = [5.6667, 0.4996, 9.2401, 1.1252, 1.3149, 1.5375,
                2.5951, 483.12, 54.183, 1.2798, 0.5000, 0.4991,
                0.4997, 0.3933, 0.4917]
        X_scaled = (X_raw - means) / stds
        X_scaled = X_scaled[0]
    
    # 使用系数
    coefs = [-0.896414, 0.769691, 0.682537, -0.081783, 0.064456,
             -0.104677, 0.259990, 0.018336, -0.047041, -0.075340,
             0.029675, -0.129385, -0.046589, 0.053170, -0.024440]
    
    display_names = ['BMI', 'Sex (Female)', 'Age', 'Race', 'Education',
                     'Income', 'NPAR', 'SII', 'PLR', 'NLR',
                     'Smoking', 'Drinking', 'Hypertension', 'Diabetes', 'Arthritis']
    
    contributions = []
    for i, (name, display, coef) in enumerate(zip(FEATURE_NAMES, display_names, coefs)):
        contrib = coef * X_scaled[i]
        if abs(contrib) > 0.005:
            direction = '↑' if contrib > 0 else '↓'
            contributions.append({
                'feature': display,
                'value': contrib,
                'direction': direction,
                'abs_value': abs(contrib)
            })
    
    contributions.sort(key=lambda x: x['abs_value'], reverse=True)
    return contributions[:6]

# ============================================================
# 页面路由
# ============================================================
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# ============================================================
# 页面1: Home
# ============================================================
def show_home():
    # 标题
    st.markdown('<div class="eo-title">🏥 EO-CDSS</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-subtitle">EXPLAINABLE OSTEOIMMUNE</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-subtitle" style="font-size:1.4rem;margin-bottom:0;">Clinical Decision Support System</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-tagline">Based on NHANES 2005-2020 · n=10,189 · AUC=0.7891</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 流程图
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
    
    # 系统特点
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="text-align:center;padding:1rem;">
            <div style="font-size:2rem;">📋</div>
            <div style="font-weight:600;color:#1a3a5c;">Routine Variables</div>
            <div style="font-size:0.85rem;color:#7a8fa3;">Age, BMI, Sex, NPAR, CBC</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:1rem;">
            <div style="font-size:2rem;">🔍</div>
            <div style="font-weight:600;color:#1a3a5c;">Explainable</div>
            <div style="font-size:0.85rem;color:#7a8fa3;">SHAP-based contributions</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="text-align:center;padding:1rem;">
            <div style="font-size:2rem;">📊</div>
            <div style="font-weight:600;color:#1a3a5c;">Clinical Decision Support</div>
            <div style="font-size:0.85rem;color:#7a8fa3;">DXA prioritization</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 开始按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
    with col_btn2:
        if st.button("🚀 Start Assessment", use_container_width=True, type="primary"):
            navigate_to('input')
    
    # 免责声明
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Use Notice</strong><br>
        This system is intended for <strong>clinical decision-support evaluation</strong> and 
        does not replace DXA or physician judgment. Always consider the patient's complete 
        clinical picture and local guidelines.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eo-footer">
        EO-CDSS v2.0 · Explainable Osteoimmune Clinical Decision Support System<br>
        For research and clinical evaluation purposes only
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 页面2: Input
# ============================================================
def show_input():
    # 步骤指示器
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
            age = st.slider("Age (years)", 50, 90, 65, help="50-90 years")
            sex = st.radio("Sex", ["Female", "Male"], index=0, horizontal=True)
            bmi = st.number_input("BMI (kg/m²)", 15.0, 45.0, 25.0, step=0.1)
            race = st.selectbox("Race/Ethnicity", 
                               ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American", 
                                "Other Hispanic", "Other Race"], index=0)
            education = st.selectbox("Education", ["≥ College", "High School/GED", "< High School"], index=0)
            income = st.number_input("Income-to-Poverty Ratio", 0.0, 5.0, 2.5, step=0.1)
        
        with col2:
            st.markdown("**Laboratory Markers**")
            npar = st.number_input("NPAR", 5.0, 30.0, 14.0, step=0.1, 
                                   help="Neutrophil percentage-to-albumin ratio")
            nlr = st.number_input("NLR", 0.5, 10.0, 2.5, step=0.1,
                                 help="Neutrophil-to-lymphocyte ratio")
            plr = st.number_input("PLR", 50, 400, 150, step=5,
                                 help="Platelet-to-lymphocyte ratio")
            sii = st.number_input("SII", 100, 2000, 500, step=50,
                                 help="Systemic immune-inflammation index")
            
            st.markdown("**Lifestyle & Comorbidities**")
            col_sm, col_dr = st.columns(2)
            with col_sm:
                smoking = st.selectbox("Smoking", ["No", "Yes"])
            with col_dr:
                drinking = st.selectbox("Drinking", ["No", "Yes"])
            
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            with col_h2:
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            with col_h3:
                arthritis = st.selectbox("Arthritis", ["No", "Yes"])
        
        st.markdown("---")
        
        # ============================================================
        # Osteopenia（核心新增功能）
        # ============================================================
        st.markdown("### 🦴 Osteopenia Status")
        st.markdown("*If the patient has known osteopenia (T-score between -1.0 and -2.5)*")
        known_osteopenia = st.radio("Known Osteopenia?", ["No", "Yes"], index=0, horizontal=True)
        
        if known_osteopenia == "Yes":
            st.info("📌 **Clinical Note**: Patients with known osteopenia may benefit from earlier DXA reassessment and individualized preventive management. This system does not predict progression from osteopenia to osteoporosis.")
        
        st.markdown("---")
        
        # 提交按钮
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
        with col_btn2:
            submitted = st.form_submit_button("📊 Generate Assessment", use_container_width=True, type="primary")
        
        if submitted:
            # 编码分类变量
            sex_encoded = 1 if sex == "Female" else 0
            smoking_encoded = 1 if smoking == "Yes" else 0
            drinking_encoded = 1 if drinking == "Yes" else 0
            hypertension_encoded = 1 if hypertension == "Yes" else 0
            diabetes_encoded = 1 if diabetes == "Yes" else 0
            arthritis_encoded = 1 if arthritis == "Yes" else 0
            
            race_map = {"Non-Hispanic White": 3, "Non-Hispanic Black": 4, 
                       "Mexican American": 1, "Other Hispanic": 2, "Other Race": 5}
            education_map = {"≥ College": 3, "High School/GED": 2, "< High School": 1}
            
            race_encoded = race_map[race]
            education_encoded = education_map[education]
            
            # 保存到session_state
            st.session_state.patient_data = {
                'age': age,
                'sex': sex_encoded,
                'sex_display': sex,
                'bmi': bmi,
                'race': race_encoded,
                'race_display': race,
                'education': education_encoded,
                'education_display': education,
                'income': income,
                'npar': npar,
                'nlr': nlr,
                'plr': plr,
                'sii': sii,
                'smoking': smoking_encoded,
                'smoking_display': smoking,
                'drinking': drinking_encoded,
                'drinking_display': drinking,
                'hypertension': hypertension_encoded,
                'hypertension_display': hypertension,
                'diabetes': diabetes_encoded,
                'diabetes_display': diabetes,
                'arthritis': arthritis_encoded,
                'arthritis_display': arthritis,
                'known_osteopenia': known_osteopenia
            }
            
            # 计算风险
            prob = calculate_risk(st.session_state.patient_data)
            st.session_state.result = prob
            
            # 跳转到结果页
            navigate_to('result')

# ============================================================
# 页面3: Result
# ============================================================
def show_result():
    # 步骤指示器
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
        st.warning("No assessment found. Please start a new assessment.")
        if st.button("Start New Assessment"):
            navigate_to('home')
        return
    
    risk_level, risk_icon, recommendation = get_risk_level(prob)
    risk_pct = prob * 100
    risk_color_class = f"risk-{risk_level.lower()}"
    
    # ============================================================
    # 结果卡片
    # ============================================================
    st.markdown("## 📊 AI Osteoporosis Assessment Report")
    
    # 主卡片
    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div>
                <div class="result-label">Estimated Osteoporosis Probability</div>
                <div class="result-number {risk_color_class}">{risk_pct:.1f}%</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:3rem;">{risk_icon}</div>
                <div style="font-size:1.5rem;font-weight:700;color:{'#c0392b' if risk_level=='High' else '#e67e22' if risk_level=='Moderate' else '#27ae60'};">{risk_level} Risk</div>
            </div>
            <div style="max-width:300px;">
                <div style="font-size:0.85rem;color:#7a8fa3;">Clinical Recommendation</div>
                <div style="font-weight:500;color:#1a3a5c;">{recommendation}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SHAP贡献条
    # ============================================================
    st.markdown("### 🔍 Key Contributors (SHAP)")
    contributions = get_shap_contributions(data, prob)
    
    if contributions:
        max_abs = max([c['abs_value'] for c in contributions])
        
        for c in contributions:
            pct = (c['abs_value'] / max_abs * 100) if max_abs > 0 else 0
            color = '#c0392b' if c['direction'] == '↑' else '#27ae60'
            st.markdown(f"""
            <div class="contrib-container">
                <div class="contrib-label">
                    <span><strong>{c['feature']}</strong> {c['direction']}</span>
                    <span>{c['abs_value']:.3f}</span>
                </div>
                <div class="contrib-bar-bg">
                    <div class="contrib-bar-fill" style="width:{pct}%;background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No major contributors identified.")
    
    # ============================================================
    # Osteopenia 临床建议
    # ============================================================
    if data['known_osteopenia'] == "Yes":
        st.markdown("### 🦴 Osteopenia-Specific Advice")
        st.info("""
        **Clinical Recommendation:**\n
        Patients with known osteopenia (T-score between -1.0 and -2.5) may benefit from:\n
        • Earlier DXA reassessment\n
        • Individualized fracture-risk assessment\n
        • Consideration of lifestyle, falls-prevention, or pharmacological evaluation\n
        *This system does not predict progression from osteopenia to osteoporosis; this advice is for clinical consideration only.*
        """)
    
    # ============================================================
    # 临床决策支持
    # ============================================================
    st.markdown("### 💡 Clinical Decision Support")
    
    st.markdown(f"""
    <div class="result-card">
        <ul style="list-style-type:none;padding-left:0;margin:0;">
            <li style="padding:4px 0;">📌 <strong>Risk Level:</strong> {risk_icon} {risk_level} Risk ({risk_pct:.1f}%)</li>
            <li style="padding:4px 0;">📋 <strong>Recommendation:</strong> {recommendation}</li>
            <li style="padding:4px 0;">📅 <strong>Suggested Follow-up:</strong> {'3-6 months' if risk_level == 'High' else '6-12 months' if risk_level == 'Moderate' else '12-24 months'}</li>
            <li style="padding:4px 0;">🔍 <strong>Interpretation:</strong> This patient has a {risk_level.lower()} estimated osteoporosis risk.</li>
            <li style="padding:4px 0;margin-top:8px;color:#7a8fa3;font-size:0.9rem;">
                ⚠️ This system is intended for <strong>clinical decision-support evaluation</strong> and 
                does not replace DXA or physician judgment.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # 按钮区：新评估 + PDF下载
    # ============================================================
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
    
    with col_btn1:
        if st.button("🔄 New Assessment", use_container_width=True):
            navigate_to('input')
    
    with col_btn2:
        # PDF下载
        if st.button("📄 Download PDF Report", use_container_width=True):
            generate_pdf_report(data, prob, risk_level, risk_pct, recommendation, contributions)
    
    with col_btn3:
        if st.button("🏠 Home", use_container_width=True):
            navigate_to('home')

# ============================================================
# PDF生成功能
# ============================================================
def generate_pdf_report(data, prob, risk_level, risk_pct, recommendation, contributions):
    """生成PDF报告并下载"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,  # 居中
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6,
        textColor=colors.HexColor('#1a3a5c')
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    story = []
    
    # 标题
    story.append(Paragraph("EO-CDSS Clinical Assessment Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # 患者信息
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ["Age", str(data['age'])],
        ["Sex", data['sex_display']],
        ["BMI", f"{data['bmi']:.1f}"],
        ["NPAR", f"{data['npar']:.1f}"],
        ["NLR", f"{data['nlr']:.1f}"],
        ["PLR", str(data['plr'])],
        ["SII", str(data['sii'])],
        ["Smoking", data['smoking_display']],
        ["Drinking", data['drinking_display']],
        ["Hypertension", data['hypertension_display']],
        ["Diabetes", data['diabetes_display']],
        ["Arthritis", data['arthritis_display']],
        ["Known Osteopenia", data['known_osteopenia']]
    ]
    
    t = Table(patient_data, colWidths=[1.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    
    # 风险结果
    story.append(Paragraph("Risk Assessment", heading_style))
    risk_text = f"<b>Osteoporosis Probability:</b> {risk_pct:.1f}%<br/>"
    risk_text += f"<b>Risk Level:</b> {risk_level}<br/>"
    risk_text += f"<b>Recommendation:</b> {recommendation}"
    story.append(Paragraph(risk_text, normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    # 主要贡献
    if contributions:
        story.append(Paragraph("Major Contributors (SHAP)", heading_style))
        for c in contributions:
            story.append(Paragraph(f"{c['feature']}: {c['direction']} ({c['abs_value']:.3f})", normal_style))
        story.append(Spacer(1, 0.1*inch))
    
    # 临床建议
    story.append(Paragraph("Clinical Decision Support", heading_style))
    story.append(Paragraph(f"- {recommendation}", normal_style))
    if data['known_osteopenia'] == "Yes":
        story.append(Paragraph("- Patients with known osteopenia may benefit from earlier DXA reassessment.", normal_style))
    story.append(Paragraph("- This system assists clinicians and does not replace DXA or physician judgment.", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    # 免责声明
    story.append(Paragraph("Disclaimer", heading_style))
    story.append(Paragraph(
        "This report is generated by an explainable AI-based clinical decision support system "
        "for research and clinical evaluation purposes. The results should be interpreted in "
        "the context of the patient's complete clinical picture and local guidelines.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"EO-CDSS v2.0 • {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    
    doc.build(story)
    
    # 下载
    pdf_data = buffer.getvalue()
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="EO-CDSS_Report_{datetime.now().strftime("%Y%m%d")}.pdf">📄 Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# ============================================================
# 主路由
# ============================================================
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'input':
    show_input()
elif st.session_state.page == 'result':
    show_result()