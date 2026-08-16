import streamlit as st
import math
from datetime import datetime

st.set_page_config(page_title="EO-CDSS", page_icon="🏥", layout="wide")

# 样式
st.markdown("""
<style>
    .eo-title { font-size: 3rem; font-weight: 800; color: #0a1628; text-align: center; }
    .eo-subtitle { font-size: 1.1rem; color: #4a6a8b; text-align: center; }
    .result-card { background: #f0f4fa; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 5px solid #2E86AB; }
    .risk-high { color: #c0392b; font-weight: 700; font-size: 2.5rem; }
    .risk-moderate { color: #e67e22; font-weight: 700; font-size: 2.5rem; }
    .risk-low { color: #27ae60; font-weight: 700; font-size: 2.5rem; }
    .disclaimer-box { background: #fff8e7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f1c40f; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------- 模型系数（直接从你论文中提取） ----------
# 截距和15个特征的系数（标准化后的）
INTERCEPT = -2.135957
COEFS = {
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

# 训练集的均值和标准差（标准化参数）
MEANS = {
    'BMXBMI': 28.696, 'RIAGENDR': 1.4788, 'RIDAGEYR': 63.825,
    'RIDRETH1': 3.0356, 'DMDEDUC2': 3.2473, 'INDFMPIR': 2.6343,
    'NPAR': 13.938, 'SII': 533.55, 'PLR': 127.88, 'NLR': 2.1958,
    '吸烟_new': 0.4990, '饮酒_new': 0.4693, '高血压_new': 0.5182,
    '糖尿病_new': 0.1913, '关节炎_new': 0.4094
}

STDS = {
    'BMXBMI': 5.6667, 'RIAGENDR': 0.4996, 'RIDAGEYR': 9.2401,
    'RIDRETH1': 1.1252, 'DMDEDUC2': 1.3149, 'INDFMPIR': 1.5375,
    'NPAR': 2.5951, 'SII': 483.12, 'PLR': 54.183, 'NLR': 1.2798,
    '吸烟_new': 0.5000, '饮酒_new': 0.4991, '高血压_new': 0.4997,
    '糖尿病_new': 0.3933, '关节炎_new': 0.4917
}

FEATURE_NAMES = ['BMXBMI', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2',
                 'INDFMPIR', 'NPAR', 'SII', 'PLR', 'NLR',
                 '吸烟_new', '饮酒_new', '高血压_new', '糖尿病_new', '关节炎_new']

# ---------- 计算函数 ----------
def calculate_risk(values):
    # 标准化
    std = []
    for name, val in zip(FEATURE_NAMES, values):
        std.append((val - MEANS[name]) / STDS[name])
    # 计算logit
    logit = INTERCEPT
    for name, s in zip(FEATURE_NAMES, std):
        logit += COEFS[name] * s
    # sigmoid
    prob = 1.0 / (1.0 + math.exp(-logit))
    return prob

def get_risk_level(prob):
    if prob >= 0.30:
        return "High", "🔴", "Consider DXA and comprehensive assessment"
    elif prob >= 0.15:
        return "Moderate", "🟡", "Consider DXA based on additional risk factors"
    else:
        return "Low", "🟢", "Routine follow-up. Lifestyle optimization recommended."

def get_contributions(values):
    std = []
    for name, val in zip(FEATURE_NAMES, values):
        std.append((val - MEANS[name]) / STDS[name])
    contribs = []
    for name, s, coef in zip(FEATURE_NAMES, std, COEFS.values()):
        c = coef * s
        if abs(c) > 0.005:
            display = name.replace('_new', '').replace('BMX', '').replace('RIAGENDR', 'Sex')
            display = display.replace('RIDAGEYR', 'Age')
            contribs.append((display, c, "↑" if c > 0 else "↓"))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:5]

# ---------- 会话状态 ----------
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'result' not in st.session_state:
    st.session_state.result = None
if 'data' not in st.session_state:
    st.session_state.data = {}

def navigate(page):
    st.session_state.page = page
    st.rerun()

# ---------- 首页 ----------
def home():
    st.markdown('<div class="eo-title">🏥 EO-CDSS</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-subtitle">Explainable Osteoimmune Clinical Decision Support System</div>', unsafe_allow_html=True)
    st.markdown('<div class="eo-subtitle" style="font-size:0.9rem;color:#7a8fa3;">Based on NHANES 2005-2020 · n=10,189 · AUC=0.7891</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:20px;margin:2rem 0;flex-wrap:wrap;">
        <div style="background:white;padding:12px 24px;border-radius:10px;border:1.5px solid #dde7f0;">📋 Routine Variables</div>
        <span style="color:#8aa3bc;font-size:1.5rem;">→</span>
        <div style="background:white;padding:12px 24px;border-radius:10px;border:1.5px solid #dde7f0;">🤖 Explainable AI</div>
        <span style="color:#8aa3bc;font-size:1.5rem;">→</span>
        <div style="background:white;padding:12px 24px;border-radius:10px;border:1.5px solid #dde7f0;">📊 Clinical Decision</div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**📋 Routine Variables**\n\nAge, BMI, Sex, NPAR, CBC")
    with col2: st.markdown("**🔍 Explainable**\n\nSHAP-based contributions")
    with col3: st.markdown("**📊 Clinical Decision**\n\nDXA prioritization")
    st.markdown("---")
    if st.button("🚀 Start Assessment", use_container_width=True):
        navigate('input')
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ This system is for <strong>clinical decision-support evaluation</strong> only.
        Does not replace DXA or physician judgment.
    </div>
    """, unsafe_allow_html=True)

# ---------- 输入页 ----------
def input_page():
    st.markdown("## 📋 Patient Information")
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age (years)", 50, 90, 65)
            sex = st.radio("Sex", ["Female", "Male"], index=0, horizontal=True)
            bmi = st.number_input("BMI (kg/m²)", 15.0, 45.0, 25.0, step=0.1)
            race = st.selectbox("Race", ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American", "Other Hispanic", "Other Race"], index=0)
            edu = st.selectbox("Education", ["≥ College", "High School/GED", "< High School"], index=0)
            income = st.number_input("Income-to-Poverty Ratio", 0.0, 5.0, 2.5, step=0.1)
        with col2:
            npar = st.number_input("NPAR", 5.0, 30.0, 14.0, step=0.1)
            nlr = st.number_input("NLR", 0.5, 10.0, 2.5, step=0.1)
            plr = st.number_input("PLR", 50, 400, 150, step=5)
            sii = st.number_input("SII", 100, 2000, 500, step=50)
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            drinking = st.selectbox("Drinking", ["No", "Yes"])
            hyp = st.selectbox("Hypertension", ["No", "Yes"])
            dm = st.selectbox("Diabetes", ["No", "Yes"])
            arth = st.selectbox("Arthritis", ["No", "Yes"])
        st.markdown("---")
        osteo = st.radio("Known Osteopenia?", ["No", "Yes"], index=0, horizontal=True)
        if osteo == "Yes":
            st.info("Patients with known osteopenia may benefit from earlier DXA reassessment.")
        if st.form_submit_button("📊 Generate Assessment", use_container_width=True, type="primary"):
            race_map = {"Non-Hispanic White":3, "Non-Hispanic Black":4, "Mexican American":1, "Other Hispanic":2, "Other Race":5}
            edu_map = {"≥ College":3, "High School/GED":2, "< High School":1}
            vals = [
                bmi, 1 if sex=="Female" else 0, age, race_map[race], edu_map[edu], income,
                npar, sii, plr, nlr,
                1 if smoking=="Yes" else 0, 1 if drinking=="Yes" else 0,
                1 if hyp=="Yes" else 0, 1 if dm=="Yes" else 0, 1 if arth=="Yes" else 0
            ]
            prob = calculate_risk(vals)
            st.session_state.result = prob
            st.session_state.data = {
                'age': age, 'sex': sex, 'bmi': bmi, 'npar': npar,
                'nlr': nlr, 'plr': plr, 'sii': sii,
                'smoking': smoking, 'drinking': drinking,
                'hypertension': hyp, 'diabetes': dm, 'arthritis': arth,
                'osteo': osteo, 'prob': prob
            }
            navigate('result')

# ---------- 结果页 ----------
def result_page():
    data = st.session_state.data
    prob = st.session_state.result
    if prob is None:
        st.warning("No assessment found.")
        if st.button("Start New"): navigate('home')
        return
    risk, icon, rec = get_risk_level(prob)
    pct = prob * 100
    color = '#c0392b' if risk=='High' else '#e67e22' if risk=='Moderate' else '#27ae60'
    st.markdown("## 📊 AI Osteoporosis Assessment Report")
    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div><div style="font-size:0.9rem;color:#7a8fa3;">Estimated Osteoporosis Probability</div>
            <div style="font-size:2.5rem;font-weight:700;color:{color};">{pct:.1f}%</div></div>
            <div style="text-align:center;font-size:2rem;">{icon}<br><span style="font-size:1.2rem;font-weight:700;color:{color};">{risk} Risk</span></div>
            <div><div style="font-size:0.85rem;color:#7a8fa3;">Recommendation</div>
            <div style="font-weight:500;">{rec}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # 主要贡献
    vals = [
        data['bmi'], 1 if data['sex']=="Female" else 0, data['age'],
        3, 2, 2.5, data['npar'], data['sii'], data['plr'], data['nlr'],
        1 if data['smoking']=="Yes" else 0, 1 if data['drinking']=="Yes" else 0,
        1 if data['hypertension']=="Yes" else 0, 1 if data['diabetes']=="Yes" else 0,
        1 if data['arthritis']=="Yes" else 0
    ]
    contribs = get_contributions(vals)
    st.markdown("### 🔍 Key Contributors")
    if contribs:
        max_abs = max(abs(c[1]) for c in contribs)
        for name, c, d in contribs:
            pct_bar = abs(c)/max_abs*100 if max_abs>0 else 0
            color_bar = '#c0392b' if d=='↑' else '#27ae60'
            st.markdown(f"""
            <div style="margin:6px 0;">
                <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
                    <span><strong>{name}</strong> {d}</span>
                    <span>{abs(c):.3f}</span>
                </div>
                <div style="background:#e9ecef;border-radius:8px;overflow:hidden;height:18px;">
                    <div style="width:{pct_bar}%;background:{color_bar};height:18px;border-radius:8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No major contributors identified.")
    # 骨量减少建议
    if data['osteo'] == "Yes":
        st.info("🦴 **Osteopenia:** Earlier DXA reassessment and individualized preventive management may be considered.")
    # 临床建议
    st.markdown("### 💡 Clinical Decision Support")
    st.markdown(f"""
    <div class="disclaimer-box">
        📌 {icon} {risk} Risk ({pct:.1f}%)<br>
        📋 {rec}<br>
        📅 Follow-up: {'3-6 months' if risk=='High' else '6-12 months' if risk=='Moderate' else '12-24 months'}<br><br>
        ⚠️ This system is for <strong>clinical decision-support evaluation</strong> only.
        Does not replace DXA or physician judgment.
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Assessment", use_container_width=True): navigate('input')
    with col2:
        # 简单文本报告下载
        txt = f"""EO-CDSS Assessment Report
Generated: {datetime.now()}

Patient: Age {data['age']}, {data['sex']}, BMI {data['bmi']}
NPAR: {data['npar']} | NLR: {data['nlr']} | PLR: {data['plr']} | SII: {data['sii']}

Risk: {pct:.1f}% ({risk})
Recommendation: {rec}
Known Osteopenia: {data['osteo']}

Contributors:
{chr(10).join([f'  {name}: {d} ({abs(c):.3f})' for name,c,d in contribs])}

Disclaimer: This system is for clinical decision-support evaluation only.
Does not replace DXA or physician judgment.
"""
        st.download_button("📄 Download Report", txt, file_name=f"EO-CDSS_Report_{datetime.now().strftime('%Y%m%d')}.txt")

# ---------- 路由 ----------
if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'input':
    input_page()
else:
    result_page()