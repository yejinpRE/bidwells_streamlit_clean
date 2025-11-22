import streamlit as st
from io import StringIO
from typing import Dict, Any, Tuple, List
from math import exp

# =========================
# PDF/text loader
# =========================
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

def extract_text_from_pdf(file) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)

def get_plain_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
    return stringio.read()


# =========================
# Variable extraction
# =========================
def extract_variables_from_text(text: str) -> Dict[str, float]:
    t = text.lower()
    vars_dict: Dict[str, float] = {}

    # 기존 변수들 ----------------------
    vars_dict["X1_units_intensity"] = 1.0 if ("dwellings" in t or "units" in t) else 0.0
    vars_dict["X2_heritage_issue"] = 1.0 if ("heritage" in t or "listed building" in t) else 0.0
    vars_dict["X3_greenbelt_issue"] = 1.0 if ("green belt" in t or "open space" in t) else 0.0
    vars_dict["X4_transport_issue"] = 1.0 if ("highway" in t or "transport" in t or "traffic" in t) else 0.0

    benefits_keywords = ["significant benefit", "substantial benefit", "major benefit", "transformational"]
    vars_dict["X5_benefit_language"] = 1.0 if any(k in t for k in benefits_keywords) else 0.0

    no_harm_keywords = ["no harm", "negligible impact", "minimal impact"]
    vars_dict["X6_no_harm_claim"] = 1.0 if any(k in t for k in no_harm_keywords) else 0.0

    vars_dict["X7_sustainability"] = 1.0 if ("sustainable" in t or "net zero" in t or "biodiversity net gain" in t) else 0.0

    community_kw = ["community benefit", "social value", "public realm", "local jobs"]
    vars_dict["X8_community_benefit"] = 1.0 if any(k in t for k in community_kw) else 0.0

    # 새 변수들 ------------------------

    # X9: 디자인 / 장소성 긍정 언급
    design_kw = ["high quality design", "good design", "well-designed", "design code", "design-led"]
    vars_dict["X9_design_quality"] = 1.0 if any(k in t for k in design_kw) or "design" in t else 0.0

    # X10: 저렴 주택 / affordable housing
    ah_kw = ["affordable housing", "social rent", "affordable homes", "intermediate housing"]
    vars_dict["X10_affordable_housing"] = 1.0 if any(k in t for k in ah_kw) else 0.0

    # X11: 일자리 / 경제효과
    jobs_kw = ["jobs", "employment", "workspace", "office floorspace", "economic benefit", "economic benefits"]
    vars_dict["X11_jobs_economic_benefit"] = 1.0 if any(k in t for k in jobs_kw) else 0.0

    # X12: 홍수위험 / Sequential test
    flood_kw = ["flood risk", "flooding", "flood zone 2", "flood zone 3", "sequential test"]
    vars_dict["X12_flood_risk_issue"] = 1.0 if any(k in t for k in flood_kw) else 0.0

    # X13: 인접 주거편익 / 소음·채광
    amenity_kw = [
        "residential amenity", "amenity of neighbouring", "overlooking", "overbearing",
        "loss of privacy", "noise", "daylight", "sunlight"
    ]
    vars_dict["X13_neighbour_amenity_issue"] = 1.0 if any(k in t for k in amenity_kw) else 0.0

    # X14: 자연 / 생태 / 보호구역
    nature_kw = [
        "biodiversity", "ecology", "habitat", "sssi", "site of special scientific interest",
        "ancient woodland", "protected species", "bat survey", "ecological impact"
    ]
    vars_dict["X14_nature_biodiversity_issue"] = 1.0 if any(k in t for k in nature_kw) else 0.0

    return vars_dict

# =========================
# Spin Engine
# =========================
SPIN_PHRASES = {
    "no harm": "there may be some degree of harm",
    "negligible impact": "a limited impact",
    "minimal impact": "a limited impact",
    "significant benefit": "a modest benefit",
    "substantial benefit": "a notable benefit",
    "major benefit": "a notable benefit",
    "transformational": "material but not exceptional"
}

def detect_spin(text: str) -> Dict[str, Any]:
    t = text.lower()
    hits = [p for p in SPIN_PHRASES if p in t]
    return {"spin_phrases_found": hits, "spin_score": len(hits)}


# =========================
# Despin Engine
# =========================
def despin_text(text: str) -> str:
    new_text = text
    for phrase, replacement in SPIN_PHRASES.items():
        new_text = new_text.replace(phrase, replacement)
        new_text = new_text.replace(phrase.capitalize(), replacement.capitalize())
    return new_text


# =========================
# Mock logistic regression
# =========================
LOGIT_COEFFS = {
    "intercept": -0.5,

    # X-variables from text
    "X1_units_intensity": -0.3,
    "X2_heritage_issue": -0.8,
    "X3_greenbelt_issue": -1.0,
    "X4_transport_issue": -0.4,
    "X5_benefit_language": 0.5,
    "X6_no_harm_claim": -0.4,
    "X7_sustainability": 0.6,
    "X8_community_benefit": 0.5,
    "X9_design_quality": 0.4,
    "X10_affordable_housing": 0.8,
    "X11_jobs_economic_benefit": 0.6,
    "X12_flood_risk_issue": -0.9,
    "X13_neighbour_amenity_issue": -0.7,
    "X14_nature_biodiversity_issue": -0.8,

    # Macro context (numeric)
    "M1_tilted_balance": 2.0,        # NPPF tilted balance 적용되면 강한 플러스
    "M2_5yhls_pressure": 0.8,       # 5YHLS 부족 압력
    "M3_lpa_approval_rate": 1.2     # LPA 승인률 (0~1)
}

def logistic(x: float) -> float:
    return 1 / (1 + exp(-x))

def calculate_probability(vars_dict: Dict[str, float],
                          macro_context: Dict[str, float]) -> Tuple[float, Dict[str, float]]:

    full_vars = {**vars_dict, **macro_context}
    linear_sum = LOGIT_COEFFS["intercept"]
    contributions = {}

    for name, value in full_vars.items():
        coef = LOGIT_COEFFS.get(name, 0.0)
        contrib = coef * value
        contributions[name] = contrib
        linear_sum += contrib

    return logistic(linear_sum), contributions

def classify_strengths_weaknesses(contribs: Dict[str, float]):
    strengths = [f"{k}: {v:+.2f}" for k, v in contribs.items() if v > 0]
    weaknesses = [f"{k}: {v:+.2f}" for k, v in contribs.items() if v < 0]
    strengths.sort(reverse=True)
    weaknesses.sort()
    return strengths, weaknesses


# =========================
# Report Engine
# =========================
def generate_report(prob, strengths, weaknesses, spin_info):
    lines = []
    lines.append(f"### Estimated approval probability: **{prob*100:.1f}%**")
    lines.append("")

    if strengths:
        lines.append("**Strengths:**")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    if weaknesses:
        lines.append("**Weaknesses:**")
        for w in weaknesses:
            lines.append(f"- {w}")
        lines.append("")

    if spin_info["spin_score"] > 0:
        lines.append(f"**Spin detected:** {spin_info['spin_score']} phrase(s)")
        lines.append(f"- {', '.join(spin_info['spin_phrases_found'])}")
        lines.append("")

    lines.append("**Suggested improvements:**")
    lines.append("- Clarify evidence behind claimed benefits.")
    lines.append("- Provide clearer mitigation for risks.")
    lines.append("- Align narrative more directly with NPPF and local policy.")
    lines.append("- Avoid overly promotional phrasing.")
    return "\n".join(lines)


# =========================
# Main UI
# =========================
def main():

    # -------- Sidebar: inputs --------
    with st.sidebar:
        st.title("Plan Checker – Inputs")

        st.markdown("### 1. Upload documents")
        ps_file = st.file_uploader(
            "Planning Statement",
            type=["pdf", "txt"],
            key="ps_lean"
        )
        rec_file = st.file_uploader(
            "Recommendation / Committee Report (optional)",
            type=["pdf", "txt"],
            key="rec_lean"
        )

        st.markdown("### 2. Macro context")
        macro_5yhls = st.selectbox(
            "5YHLS position",
            ["Unknown / Neutral", "Above 5 years", "Below 5 years"],
            key="macro_5yhls"
        )
        macro_political = st.selectbox(
            "Committee attitude",
            ["Unknown / Mixed", "Generally pro-development", "Generally risk-averse"],
            key="macro_political"
        )

        macro_context = {
            "M1_5yhls_pressure": (
                1.0 if macro_5yhls == "Below 5 years"
                else -0.2 if macro_5yhls == "Above 5 years"
                else 0.0
            ),
            "M2_political_support": (
                0.5 if macro_political == "Generally pro-development"
                else -0.5 if macro_political == "Generally risk-averse"
                else 0.0
            )
        }

        st.markdown("### 3. Run")
        run_clicked = st.button("Run Plan Checker", use_container_width=True)

    # -------- Main area: outputs --------
    st.title("Bayes ‘Plan Checker’ – Lean Prototype")
    st.markdown(
        """
        This prototype:
        1. Reads the Planning Statement  
        2. Detects and removes **spin**  
        3. Re-scores the de-spun statement  
        4. Estimates the **probability of approval** and explains strengths / weaknesses  
        """
    )

    if not run_clicked:
        st.info("Upload a Planning Statement and click **Run Plan Checker** in the sidebar.")
        return

    if ps_file is None:
        st.error("Please upload at least a Planning Statement file in the sidebar.")
        return

    # -------- Core processing --------
    ps_text = get_plain_text(ps_file)
    rec_text = get_plain_text(rec_file) if rec_file else ""

    vars_initial = extract_variables_from_text(ps_text)
    spin_info = detect_spin(ps_text)

    despinned = despin_text(ps_text)
    vars_despin = extract_variables_from_text(despinned)

    prob, contribs = calculate_probability(vars_despin, macro_context)
    strengths, weaknesses = classify_strengths_weaknesses(contribs)

    # -------- Tabs for results --------
    tab1, tab2, tab3 = st.tabs(
        ["🔎 Headline result", "📊 Variables & Spin", "📄 Text view"]
    )

    # --- Tab 1: Headline result ---
    with tab1:
        st.subheader("Headline approval probability")
        st.metric("Estimated probability of approval", f"{prob*100:.1f}%")

        st.markdown("### Key strengths")
        if strengths:
            st.markdown("\n".join(f"- {s}" for s in strengths))
        else:
            st.write("No clear strengths detected.")

        st.markdown("### Key weaknesses / risks")
        if weaknesses:
            st.markdown("\n".join(f"- {w}" for w in weaknesses))
        else:
            st.write("No clear weaknesses detected.")

        st.markdown("---")
        st.markdown("### Summary report")
        st.markdown(generate_report(prob, strengths, weaknesses, spin_info))

    # --- Tab 2: Variables & Spin ---
    with tab2:
        st.subheader("Variable scores (before vs after Despin)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Initial variables (original text)**")
            st.json(vars_initial)
        with col2:
            st.markdown("**Variables after Despin**")
            st.json(vars_despin)

        st.markdown("---")
        st.subheader("Contributions (coef × value)")
        st.json(contribs)

        st.markdown("---")
        st.subheader("Spin detection")
        st.json(spin_info)

    # --- Tab 3: Text view ---
    with tab3:
        st.subheader("Original vs Despinned Planning Statement")

        st.markdown("**Original Planning Statement (sample)**")
        st.text(ps_text[:2000])

        st.markdown("---")
        st.markdown("**Despinned Planning Statement (sample)**")
        st.text(despinned[:2000])

if __name__ == "__main__":
    main()
