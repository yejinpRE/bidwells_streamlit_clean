import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from engine1_text import (
    extract_text_from_pdf,
    engine1_run,
    base_scores_from_text,  # used in repository tab if needed
)
from engine2_context import build_context_features
from engine3_model import (
    predict_approval_probability,
    _sigmoid,
    logistic_from_coeffs,
)

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bayes Plan Checker – Prototype",
    layout="wide",
)

st.title("Bayes ‘Plan Checker’ System – Prototype")
st.caption("Engine 0–3 | Multi-document, context-aware planning risk engine")

# ---------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------
for key in [
    "ps_text",
    "cr_text",
    "ap_text",
    "ps_scores",
    "cr_scores",
    "doc_features",
    "ctx_features",
    "prediction",
    "repo_df",
    "trained_model",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_engine01, tab_engine2, tab_engine3, tab_repo = st.tabs(
    [
        "Engine 0 & 1 – Documents",
        "Engine 2 – Context",
        "Engine 3 – Output",
        "Repository / Batch",
    ]
)

# =========================================================
# TAB 1 – ENGINE 0 & 1: Documents
# =========================================================
with tab_engine01:
    st.header("Engine 0 & 1 – Document analysis")

    st.markdown(
        """
        **Purpose**  
        This tab runs Engine 0 (rulebook scoring) and Engine 1 (multi-document analysis).  
        Upload the documents for the case you want to analyse:
        - Planning Statement (PS)  
        - Committee / Officer Report (CR)  
        - Appeal Decision (optional)
        """
    )

    col_ps, col_cr, col_ap = st.columns(3)

    with col_ps:
        ps_file = st.file_uploader(
            "Planning Statement (PS)",
            type=["pdf"],
            key="ps_uploader",
        )
    with col_cr:
        cr_file = st.file_uploader(
            "Committee / Officer Report (CR)",
            type=["pdf"],
            key="cr_uploader",
        )
    with col_ap:
        ap_file = st.file_uploader(
            "Appeal Decision (optional)",
            type=["pdf"],
            key="ap_uploader",
        )

    run_engine01 = st.button("Run Engine 0 & 1 on uploaded documents")

    if run_engine01:
        if not (ps_file or cr_file):
            st.error("Please upload at least a Planning Statement or a Committee/Officer Report.")
        else:
            with st.spinner("Reading PDFs and running Engine 0 & 1..."):
                ps_text = extract_text_from_pdf(ps_file) if ps_file else None
                cr_text = extract_text_from_pdf(cr_file) if cr_file else None
                ap_text = extract_text_from_pdf(ap_file) if ap_file else None

                st.session_state["ps_text"] = ps_text
                st.session_state["cr_text"] = cr_text
                st.session_state["ap_text"] = ap_text

                ps_scores, cr_scores, doc_features = engine1_run(ps_text, cr_text, ap_text)

                st.session_state["ps_scores"] = ps_scores
                st.session_state["cr_scores"] = cr_scores
                st.session_state["doc_features"] = doc_features

            st.success("Engine 0 & 1 completed for this case.")

    if st.session_state["doc_features"] is not None:
        st.subheader("Engine 0 & 1 – Outputs for this case")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Planning Statement – Rulebook scores**")
            if st.session_state["ps_scores"]:
                st.json(st.session_state["ps_scores"])
            else:
                st.info("No Planning Statement uploaded.")

            st.markdown("**Appeal Decision – Scores (if any)**")
            ap_scores = st.session_state["doc_features"].get("Appeal_Scores", {})
            if ap_scores:
                st.json(ap_scores)
            else:
                st.info("No Appeal Decision uploaded.")

        with col2:
            st.markdown("**Committee / Officer Report – Rulebook scores**")
            if st.session_state["cr_scores"]:
                st.json(st.session_state["cr_scores"])
            else:
                st.info("No Committee/Officer Report uploaded.")

            st.markdown("**Aggregated X-variables used by later engines**")
            st.json(st.session_state["doc_features"])

        st.markdown("**Spin Index (difference between PS and CR)**")
        spin_index = st.session_state["doc_features"].get("X10_Spin_Index", 0.0)
        st.metric("X10 – Spin Index", f"{spin_index:.2f}")
    else:
        st.info("Upload documents and click the button to run Engine 0 & 1.")

# =========================================================
# TAB 2 – ENGINE 2: Context
# =========================================================
with tab_engine2:
    st.header("Engine 2 – Context inputs")

    st.markdown(
        """
        **Purpose**  
        Engine 2 captures the planning context that is not directly in the documents, such as:
        - Housing pressure  
        - Tilted balance status  
        - Local plan age  
        - Committee attitude  
        - Green Belt flag  
        - Flood zone level  
        - Land type (Brownfield / Greenfield)
        """
    )

    with st.form("context_form"):
        housing_pressure = st.slider(
            "Housing pressure (0 = low, 3 = very high)",
            0.0,
            3.0,
            1.5,
            0.5,
        )

        tb_status = st.select_slider(
            "Tilted balance status",
            options=[0, 1, 2],
            value=1,
            help="0 = Not engaged, 1 = Arguable, 2 = Clearly engaged",
        )

        plan_age = st.select_slider(
            "Local plan age",
            options=[0, 1, 2],
            value=1,
            help="0 = Up-to-date, 1 = Mid, 2 = Old / out-of-date",
        )

        committee_attitude = st.slider(
            "Committee attitude (0 = very restrictive, 3 = very permissive)",
            0.0,
            3.0,
            1.5,
            0.5,
        )

        gb_flag = st.selectbox(
            "Green Belt?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

        floodzone_level = st.select_slider(
            "Flood zone level",
            options=[0, 1, 2, 3],
            value=0,
            help="0 = None / Zone 1, 2 = Zone 2, 3 = Zone 3",
        )

        land_type = st.selectbox(
            "Land type (Brownfield / Greenfield)",
            options=[0, 1],
            format_func=lambda x: (
                "Greenfield / undeveloped land"
                if x == 0
                else "Brownfield / previously developed land (PDL)"
            ),
            help="0 = Greenfield / open land, 1 = Brownfield / previously developed land (PDL)",
        )

        submit_ctx = st.form_submit_button("Save Engine 2 context inputs")

    if submit_ctx:
        ctx_features = build_context_features(
            housing_pressure=housing_pressure,
            tb_status=tb_status,
            plan_age=plan_age,
            committee_attitude=committee_attitude,
            gb_flag=gb_flag,
            floodzone_level=floodzone_level,
            land_type=land_type,
        )
        st.session_state["ctx_features"] = ctx_features
        st.success("Context inputs saved for this case.")

    if st.session_state["ctx_features"] is not None:
        st.subheader("Current Engine 2 context features")
        st.json(st.session_state["ctx_features"])
    else:
        st.info("Set and save context inputs to use Engine 2.")

# =========================================================
# TAB 3 – ENGINE 3: Output (Rule-based + Trained)
# =========================================================
with tab_engine3:
    st.header("Engine 3 – Predictive output")

    st.markdown(
        """
        **Purpose**  
        Engine 3 combines:
        - Engine 0 & 1 document variables (X1–X10)  
        - Engine 2 context variables (X11–X17)  
        - Interaction terms (Z-variables)  
        
        …into a logit-style model to estimate the probability of planning approval and
        to show the contribution of each variable.

        If a trained model exists from the Repository tab, Engine 3 will also show
        the **data-driven probability** using those trained coefficients.
        """
    )

    if st.session_state["doc_features"] is None:
        st.error("Please run Engine 0 & 1 first (upload documents in the first tab).")
    elif st.session_state["ctx_features"] is None:
        st.error("Please set and save context inputs in Engine 2 tab.")
    else:
        run_engine3 = st.button("Run Engine 3 – Compute probability")
        if run_engine3:
            X_all = {}
            X_all.update(st.session_state["doc_features"])
            X_all.update(st.session_state["ctx_features"])

            with st.spinner("Running Engine 3 (rule-based model)..."):
                pred = predict_approval_probability(X_all)

            st.session_state["prediction"] = {
                "rule_based": pred,
                "X_all": X_all,
            }
            st.success("Engine 3 (rule-based) completed for this case.")

        if st.session_state["prediction"] is not None:
            pred = st.session_state["prediction"]["rule_based"]
            X_all = st.session_state["prediction"]["X_all"]
            prob = pred["probability"]
            rating = pred["rating"]

            st.subheader("Rule-based model result")

            col_main, col_side = st.columns([2, 1])
            with col_main:
                st.metric(
                    label="Rule-based approval probability",
                    value=f"{prob*100:.1f}%",
                )
                st.write(f"**Risk rating:** {rating}")
                st.progress(min(max(prob, 0.01), 0.99))

            with col_side:
                st.markdown("**Linear score (Z_total)**")
                st.write(f"{pred['linear_score']:.3f}")

            st.markdown("---")
            st.markdown("### Rule-based coefficients & contributions")

            contrib_rows = pred.get("contributions", [])
            if contrib_rows:
                df_contrib = pd.DataFrame(contrib_rows)
                df_contrib = df_contrib.sort_values(
                    by="abs_contribution", ascending=False
                )

                st.write("Top drivers (rule-based, by |β × X|):")
                st.dataframe(
                    df_contrib.head(10)[
                        [
                            "name",
                            "type",
                            "value",
                            "coefficient",
                            "contribution",
                            "policy_ref",
                        ]
                    ]
                )

                with st.expander("Show all variables and contributions (rule-based)"):
                    st.dataframe(
                        df_contrib[
                            [
                                "name",
                                "type",
                                "value",
                                "coefficient",
                                "contribution",
                                "policy_ref",
                            ]
                        ]
                    )

                st.caption(
                    "‘policy_ref’ shows the key NPPF paragraphs linked to each driver, "
                    "so planners can see exactly which parts of the Framework underpin the model."
                )

            # ---- If a trained regression model exists, show its prediction as well ----
            if st.session_state.get("trained_model") is not None:
                st.markdown("---")
                st.markdown("### Trained regression result (document-only model)")

                tm = st.session_state["trained_model"]
                feature_names = tm.get("feature_names", [])
                coef = np.array(tm.get("coef", []), dtype=float)
                intercept = float(tm.get("intercept", 0.0))

                trained_result = logistic_from_coeffs(
                    {k: float(v) for k, v in X_all.items()},
                    feature_names=feature_names,
                    coef=coef.tolist(),
                    intercept=intercept,
                    policy_links=None,  # 원하면 POLICY_LINKS 넘겨도 됨
                )

                y_prob = trained_result["probability"]
                z_trained = trained_result["linear_score"]
                contrib_trained_rows = trained_result["contributions"]

                st.metric(
                    label="Trained model – approval probability",
                    value=f"{y_prob*100:.1f}%",
                )

                contrib_df = pd.DataFrame(contrib_trained_rows).sort_values(
                    "abs_contribution", ascending=False
                )

                st.markdown("#### Trained model – top drivers")
                st.dataframe(
                    contrib_df.head(10)[
                        ["name", "value", "coefficient", "contribution"]
                    ]
                )

                with st.expander("Show all trained contributions"):
                    st.dataframe(
                        contrib_df[
                            ["name", "value", "coefficient", "contribution"]
                        ]
                    )

                st.caption(
                    "The trained logistic regression model currently uses only document-based "
                    "features (X1–X10 and the Spin Index). As we expand the repository, we can "
                    "extend this to include context variables as well."
                )
            else:
                st.info("No trained model found yet – train one in the Repository tab.")

# =========================================================
# TAB 4 – REPOSITORY / BATCH (Case bundles + Outcomes + Training)
# =========================================================
with tab_repo:
    st.header("Repository / Batch – Case Bundles, Outcomes & Training")

    st.markdown(
        """
        ### Purpose  
        - Upload **multiple PDFs** for multiple cases  
        - Automatically match **PS / CR / AP** by CaseID  
        - Run Engine 0 & 1 to generate case-level features  
        - Edit **Outcomes (Approved / Refused)**  
        - Train a first-pass logistic regression model from the repository  

        ### File naming rule  
        `{CaseID}_PS.pdf`  
        `{CaseID}_CR.pdf`  
        `{CaseID}_AP.pdf`

        Example:  
        `Case001_PS.pdf`, `Case001_CR.pdf`, `Case001_AP.pdf`
        """
    )

    batch_files = st.file_uploader(
        "Upload multiple PDFs (PS / CR / AP bundles)",
        type=["pdf"],
        accept_multiple_files=True,
        key="bundle_files",
    )

    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
    with col_b1:
        run_case_bundle = st.button("Process Case Bundles")
    with col_b2:
        clear_repo2 = st.button("Clear Repository Table")
    with col_b3:
        download_repo2 = st.button("Download Repository CSV")

    # Clear table
    if clear_repo2:
        st.session_state["repo_df"] = None
        st.success("Repository cleared.")

    # A. Case bundle processing
    if run_case_bundle:
        import re

        if not batch_files:
            st.error("Upload files first.")
        else:
            st.info("Processing uploaded files…")

            cases = {}
            for f in batch_files:
                fname = f.name.upper()
                match = re.match(r"(.*)_(PS|CR|AP)\.PDF$", fname)
                if not match:
                    st.warning(f"{f.name} ignored – wrong naming pattern.")
                    continue

                case_id = match.group(1)
                doc_type = match.group(2)

                if case_id not in cases:
                    cases[case_id] = {"PS": None, "CR": None, "AP": None}

                cases[case_id][doc_type] = f

            all_rows = []
            for case_id, bundle in cases.items():
                ps_text = extract_text_from_pdf(bundle["PS"]) if bundle["PS"] else None
                cr_text = extract_text_from_pdf(bundle["CR"]) if bundle["CR"] else None
                ap_text = extract_text_from_pdf(bundle["AP"]) if bundle["AP"] else None

                ps_scores, cr_scores, doc_features = engine1_run(ps_text, cr_text, ap_text)

                X = doc_features.copy()
                X["CaseID"] = case_id
                X["PS_uploaded"] = bundle["PS"].name if bundle["PS"] else None
                X["CR_uploaded"] = bundle["CR"].name if bundle["CR"] else None
                X["AP_uploaded"] = bundle["AP"].name if bundle["AP"] else None

                all_rows.append(X)

            if all_rows:
                df_new = pd.DataFrame(all_rows)
                if st.session_state["repo_df"] is not None:
                    st.session_state["repo_df"] = (
                        pd.concat(
                            [st.session_state["repo_df"], df_new],
                            ignore_index=True,
                        )
                        .drop_duplicates(subset=["CaseID"], keep="last")
                    )
                else:
                    st.session_state["repo_df"] = df_new

                st.success("Case bundles processed into repository table.")

    # Show / edit repository table and outcomes
    if st.session_state["repo_df"] is not None:
        st.markdown("### Repository table")

        df_repo = st.session_state["repo_df"].copy()
        if "Outcome" not in df_repo.columns:
            df_repo["Outcome"] = np.nan

        st.markdown(
            "Use the **Outcome** column to enter 1 for Approved and 0 for Refused "
            "(or leave blank if unknown)."
        )

        edited_df = st.data_editor(
            df_repo,
            num_rows="dynamic",
            key="repo_editor",
        )

        st.session_state["repo_df"] = edited_df

        # Training section
        st.markdown("---")
        st.markdown("### Train logistic regression model from repository")

        train_btn = st.button("Train logistic regression (document features only)")
        if train_btn:
            df_train = edited_df.dropna(subset=["Outcome"]).copy()

            if df_train.empty:
                st.error("Need at least one row with an Outcome (0/1) to train the model.")
            else:
                feature_cols = [
                    "X1_Heritage_Harm",
                    "X2_Design_Quality",
                    "X3_Amenity_Harm",
                    "X4_Ecology_Harm",
                    "X5_GB_Harm",
                    "X6_Flood_Risk",
                    "X7_Economic_Benefit",
                    "X8_Social_Benefit",
                    "X9_Policy_Compliance",
                    "X10_Spin_Index",
                ]

                missing = [c for c in feature_cols if c not in df_train.columns]
                if missing:
                    st.error(f"Missing feature columns in repository: {missing}")
                else:
                    X = df_train[feature_cols].astype(float)
                    y = df_train["Outcome"].astype(int)

                    try:
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X, y)

                        coef = model.coef_[0]
                        intercept = float(model.intercept_[0])

                        st.session_state["trained_model"] = {
                            "feature_names": feature_cols,
                            "coef": coef.tolist(),
                            "intercept": intercept,
                            "model_type": "logistic",
                        }

                        coef_df = pd.DataFrame(
                            {
                                "feature": feature_cols,
                                "coefficient": coef,
                            }
                        ).sort_values("coefficient", ascending=False)

                        st.success("Trained logistic regression model from repository.")
                        st.markdown("#### Trained coefficients (document-based features)")
                        st.dataframe(coef_df)
                    except Exception as e:
                        st.error(f"Error training logistic regression model: {e}")

        if download_repo2:
            csv = st.session_state["repo_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download repository as CSV",
                data=csv,
                file_name="plan_checker_repository.csv",
                mime="text/csv",
            )
    else:
        st.info("No repository data yet – upload and process case bundles first.")
