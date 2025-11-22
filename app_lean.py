import os
from io import StringIO
from math import exp
from typing import Dict, Any, Tuple, List

import pandas as pd
import streamlit as st

# PDF loader
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# =========================
# 1. Text extraction
# =========================
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
# 2. Variable extraction (X1~X14)
# =========================
def extract_variables_from_text(text: str) -> Dict[str, float]:
    t = text.lower()
    vars_dict: Dict[str, float] = {}

    # X1–X4: basic risk themes
    vars_dict["X1_units_intensity"] = 1.0 if ("dwellings" in t or "units" in t) else 0.0
    vars_dict["X2_heritage_issue"] = 1.0 if ("heritage" in t or "listed building" in t) else 0.0
    vars_dict["X3_greenbelt_issue"] = 1.0 if ("green belt" in t or "open space" in t) else 0.0
    vars_dict["X4_transport_issue"] = 1.0 if ("highway" in t or "transport" in t or "traffic" in t) else 0.0

    # X5: positive benefit language
    benefits_keywords = ["significant benefit", "substantial benefit", "major benefit", "transformational"]
    vars_dict["X5_benefit_language"] = 1.0 if any(k in t for k in benefits_keywords) else 0.0

    # X6: "no harm" style spin
    no_harm_keywords = ["no harm", "negligible impact", "minimal impact"]
    vars_dict["X6_no_harm_claim"] = 1.0 if any(k in t for k in no_harm_keywords) else 0.0

    # X7: sustainability / climate
    vars_dict["X7_sustainability"] = (
        1.0 if ("sustainable" in t or "net zero" in t or "biodiversity net gain" in t) else 0.0
    )

    # X8: community / social benefits
    community_kw = ["community benefit", "social value", "public realm", "local jobs"]
    vars_dict["X8_community_benefit"] = 1.0 if any(k in t for k in community_kw) else 0.0

    # X9: design quality
    design_kw = ["high quality design", "good design", "well-designed", "design code", "design-led"]
    vars_dict["X9_design_quality"] = 1.0 if any(k in t for k in design_kw) or "design" in t else 0.0

    # X10: affordable housing
    ah_kw = ["affordable housing", "social rent", "affordable homes", "intermediate housing"]
    vars_dict["X10_affordable_housing"] = 1.0 if any(k in t for k in ah_kw) else 0.0

    # X11: jobs / economic benefit
    jobs_kw = ["jobs", "employment", "workspace", "office floorspace", "economic benefit"]
    vars_dict["X11_jobs_economic_benefit"] = 1.0 if any(k in t for k in jobs_kw) else 0.0

    # X12: flood risk
    flood_kw = ["flood risk", "flooding", "flood zone", "sequential test"]
    vars_dict["X12_flood_risk_issue"] = 1.0 if any(k in t for k in flood_kw) else 0.0

    # X13: neighbour amenity
    amenity_kw = ["residential amenity", "overlooking", "overbearing", "noise", "daylight", "sunlight"]
    vars_dict["X13_neighbour_amenity_issue"] = 1.0 if any(k in t for k in amenity_kw) else 0.0

    # X14: ecology / biodiversity
    nature_kw = ["biodiversity", "ecology", "habitat", "sssi", "ancient woodland", "protected species"]
    vars_dict["X14_nature_biodiversity_issue"] = 1.0 if any(k in t for k in nature_kw) else 0.0

    return vars_dict


# =========================
# 3. Spin & Despin engine
# =========================
SPIN_PHRASES = {
    "no harm": "there may be some degree of harm",
    "negligible impact": "a limited impact",
    "minimal impact": "a limited impact",
    "significant benefit": "a modest benefit",
    "substantial benefit": "a notable benefit",
    "major benefit": "a notable benefit",
    "transformational": "material but not exceptional",
}


def detect_spin(text: str) -> Dict[str, Any]:
    t = text.lower()
    hits = [p for p in SPIN_PHRASES if p in t]
    return {"spin_phrases_found": hits, "spin_score": len(hits)}


def despin_text(text: str) -> str:
    new_text = text
    for phrase, replacement in SPIN_PHRASES.items():
        new_text = new_text.replace(phrase, replacement)
        new_text = new_text.replace(phrase.capitalize(), replacement.capitalize())
    return new_text


# =========================
# 4. Mock logistic regression
# =========================
LOGIT_COEFFS: Dict[str, float] = {
    "intercept": -0.5,

    # X variables
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

    # Macro numeric context
    "M1_tilted_balance": 2.0,
    "M2_5yhls_pressure": 0.8,
    "M3_lpa_approval_rate": 1.2,
}


def logistic(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def calculate_probability(vars_dict: Dict[str, float],
                          macro_context: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    full_vars = {**vars_dict, **macro_context}
    linear_sum = LOGIT_COEFFS["intercept"]
    contributions: Dict[str, float] = {}

    for name, value in full_vars.items():
        coef = LOGIT_COEFFS.get(name, 0.0)
        contrib = coef * value
        contributions[name] = contrib
        linear_sum += contrib

    prob = logistic(linear_sum)
    return prob, contributions


def classify_strengths_weaknesses(contribs: Dict[str, float]) -> Tuple[List[str], List[str]]:
    strengths = [f"{k}: {v:+.2f}" for k, v in contribs.items() if v > 0]
    weaknesses = [f"{k}: {v:+.2f}" for k, v in contribs.items() if v < 0]
    strengths.sort(key=lambda x: float(x.split(":")[1]), reverse=True)
    weaknesses.sort(key=lambda x: float(x.split(":")[1]))
    return strengths, weaknesses

def generate_report(prob: float,
                    strengths: List[str],
                    weaknesses: List[str],
                    spin_info: Dict[str, Any]) -> str:
    """
    Build a simple AI-style planning risk report string
    based on the calculated probability, strengths, weaknesses,
    and spin information.
    """
    lines: List[str] = []
    # 1) Headline probability
    lines.append(f"Estimated approval probability: **{prob*100:.1f}%**")
    lines.append("")

    # 2) Strengths
    if strengths:
        lines.append("**Strengths:**")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    # 3) Weaknesses
    if weaknesses:
        lines.append("**Weaknesses / risks:**")
        for w in weaknesses:
            lines.append(f"- {w}")
        lines.append("")

    # 4) Spin info
    if spin_info.get("spin_score", 0) > 0:
        lines.append(f"**Spin detected:** {spin_info['spin_score']} phrase(s)")
        phrases = spin_info.get("spin_phrases_found") or []
        if phrases:
            lines.append(f"- Phrases: {', '.join(phrases)}")
        lines.append("")

    # 5) Generic improvement advice
    lines.append("**Suggested improvements:**")
    lines.append("- Clarify evidence behind key claimed benefits.")
    lines.append("- Add specific mitigation for identified harms (heritage, transport, flood, amenity, ecology).")
    lines.append("- Align the narrative explicitly with NPPF tests and local plan policies.")
    lines.append("- Reduce overly promotional or 'no harm' language; adopt balanced officer-style wording.")

    return "\n".join(lines)

# =========================
# 5. Case database saver
# =========================
def save_case_to_csv(
    case_id: str,
    outcome: str,
    prob: float,
    vars_dict: Dict[str, float],
    macro_context: Dict[str, float],
    spin_info: Dict[str, Any],
    path: str = "cases.csv",
) -> None:

    row = {
        "case_id": case_id,
        "outcome": outcome,
        "predicted_prob": prob,
        "spin_score": spin_info.get("spin_score", 0),
    }

    for k, v in vars_dict.items():
        row[k] = v

    for k, v in macro_context.items():
        row[k] = v

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)


# =========================
# 6. UI – sidebar + tabs (with session state)
# =========================
def main():
    st.title("Bayes ‘Plan Checker’ – Lean Prototype")

    # Maintain "has_run" state so Save works even after rerun
    if "has_run" not in st.session_state:
        st.session_state["has_run"] = False

    # ---------- Sidebar: inputs ----------
    with st.sidebar:
        st.title("Inputs")

        st.markdown("### 1. Upload documents")
        ps_file = st.file_uploader(
            "Planning Statement",
            type=["pdf", "txt"],
            key="ps_lean",
        )
        rec_file = st.file_uploader(
            "Officer / Committee Report (optional)",
            type=["pdf", "txt"],
            key="rec_lean",
        )

        st.markdown("### 2. Macro context (numeric)")

        tilted_on = st.checkbox("Tilted balance applies (NPPF 11d)")
        M1_tilted_balance = 1.0 if tilted_on else 0.0

        five_years = st.number_input(
            "5YHLS (years of supply)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
        )
        M2_5yhls_pressure = 1.0 if five_years < 5.0 else -0.5

        approval_pct = st.number_input(
            "Local planning approval rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0,
        )
        M3_lpa_approval_rate = approval_pct / 100.0

        macro_context = {
            "M1_tilted_balance": M1_tilted_balance,
            "M2_5yhls_pressure": M2_5yhls_pressure,
            "M3_lpa_approval_rate": M3_lpa_approval_rate,
        }

        st.markdown("### 3. Run")
        run_clicked = st.button("Run Plan Checker", use_container_width=True)
        if run_clicked:
            st.session_state["has_run"] = True

    # ---------- Guard conditions ----------
    if not st.session_state["has_run"]:
        st.info("Upload documents and set macro context, then click **Run Plan Checker**.")
        return

    if ps_file is None:
        st.error("Please upload a Planning Statement (PDF or TXT).")
        return

    # ---------- Core processing ----------
    ps_text = get_plain_text(ps_file)
    vars_initial = extract_variables_from_text(ps_text)
    spin_info = detect_spin(ps_text)
    despinned = despin_text(ps_text)
    vars_despin = extract_variables_from_text(despinned)

    prob, contribs = calculate_probability(vars_despin, macro_context)
    strengths, weaknesses = classify_strengths_weaknesses(contribs)

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔎 Headline result",
            "📊 Variables & Spin",
            "🧹 De-spun Planning Statement",
            "📁 Case database",
        ]
    )

    # --- Tab 1: Headline + Save case ---
    with tab1:
        st.subheader("Approval probability")
        st.metric("Estimated probability of approval", f"{prob*100:.1f}%")

        st.markdown("### Strengths")
        st.write(strengths or "None")

        st.markdown("### Weaknesses / risks")
        st.write(weaknesses or "None")

        st.markdown("---")
        st.markdown("### AI-style planning risk report")
        st.markdown(generate_report(prob, strengths, weaknesses, spin_info))

        # Save case (form to avoid losing input on rerun)
        st.markdown("---")
        st.subheader("Save this case to database")

        with st.form("save_case_form"):
            case_id = st.text_input(
                "Case ID (e.g. 'Case 16 – Land at Gough Street')",
                value="",
            )
            outcome = st.selectbox(
                "Decision outcome (from Decision Document)",
                ["Unknown", "Approved", "Refused"],
            )
            submitted = st.form_submit_button("Save case")

            if submitted:
                if not case_id.strip():
                    st.warning("Please enter a Case ID before saving.")
                elif outcome == "Unknown":
                    st.warning("Please select an outcome (Approved / Refused).")
                else:
                    save_case_to_csv(
                        case_id=case_id.strip(),
                        outcome=outcome,
                        prob=prob,
                        vars_dict=vars_despin,
                        macro_context=macro_context,
                        spin_info=spin_info,
                        path="cases.csv",
                    )
                    st.success("Case saved to cases.csv ✅")

    # --- Tab 2: Variables & Spin (slightly prettier) ---
    with tab2:
        st.subheader("Variables (before vs after de-spinning)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original variables**")
            st.json(vars_initial)
        with col2:
            st.markdown("**De-spun variables**")
            st.json(vars_despin)

        st.markdown("---")
        st.subheader("Contribution breakdown (coef × value)")

        contrib_df = (
            pd.DataFrame(
                [{"variable": k, "contribution": v} for k, v in contribs.items()]
            )
            .sort_values("contribution")
        )
        st.dataframe(contrib_df, use_container_width=True)

        st.bar_chart(
            contrib_df.set_index("variable"),
            height=300,
        )

        st.markdown("---")
        st.subheader("Spin detection")
        st.write(f"Spin score: **{spin_info['spin_score']}**")
        st.write(spin_info["spin_phrases_found"])

    # --- Tab 3: De-spun PS ---
    with tab3:
        st.subheader("De-spun Planning Statement (bias removed)")
        st.text(despinned[:3000])

        st.markdown("---")
        st.subheader("Original Planning Statement (for comparison)")
        st.text(ps_text[:3000])

    # --- Tab 4: Case database viewer ---
    with tab4:
        st.subheader("Saved cases database")

        if os.path.exists("cases.csv"):
            df = pd.read_csv("cases.csv")
            st.dataframe(df, use_container_width=True)

            st.markdown("---")
            st.subheader("Summary")

            total = len(df)
            approved = (df["outcome"] == "Approved").sum()
            refused = (df["outcome"] == "Refused").sum()
            avg_prob = df["predicted_prob"].mean()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total cases", total)
            c2.metric("Approved", approved)
            c3.metric("Refused", refused)
            if avg_prob is not None:
                st.metric("Average predicted probability", f"{avg_prob*100:.1f}%")
        else:
            st.info("No cases saved yet. Run the model and use **Save case** in Tab 1 to build the database.")


if __name__ == "__main__":
    main()

