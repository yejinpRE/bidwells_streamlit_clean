# engine3_model.py
from typing import Dict, Any, List, Optional
import math


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# 각 X-variable이 어떤 NPPF 문단과 연결되는지 표시하는 매핑
POLICY_LINKS: Dict[str, str] = {
    # --- Document X-variables (Engine 0 & 1) ---
    "X1_Heritage_Harm": "NPPF 202–215 (Conserving and enhancing the historic environment)",
    "X2_Design_Quality": "NPPF 131–139 (Achieving well-designed places)",
    "X3_Amenity_Harm": "NPPF 96–100 (Promoting healthy and safe communities)",
    "X4_Ecology_Harm": "NPPF 187–199 (Natural environment, biodiversity, irreplaceable habitats)",
    "X5_GB_Harm": "NPPF 142–160 (Green Belt policy & harm thresholds)",
    "X6_Flood_Risk": "NPPF 170–182 (Flood risk, sequential test, exception test)",
    "X7_Economic_Benefit": "NPPF 85–87 (Building a strong, competitive economy)",
    "X8_Social_Benefit": "NPPF 7–8, 96–100 (Social objective & community benefits)",
    "X9_Policy_Compliance": "NPPF 11–14, 36 (Plan-led system, presumption in favour of sustainable development)",
    "X10_Spin_Index": "Internal metric (developer optimism vs planning reality)",

    # --- Context variables (Engine 2) ---
    "X11_Housing_Pressure": "NPPF 61–81 (Housing need, supply, delivery, 5YHLS, HDT)",
    "X12_TB_Status": "NPPF 11(d), footnote 8 (Presumption & tilted balance logic)",
    "X13_Plan_Age": "NPPF 32–38 (Plan-making, plan age, out-of-date conditions)",
    "X14_Committee_Attitude": "Local decision-making culture (not directly in NPPF)",
    "X15_GB_Flag": "NPPF 142–160 (Green Belt designation & tests)",
    "X16_FloodZone_Level": "NPPF 170–182 (Flood Zones, sequential/exception test)",
    "X17_Land_Type": "NPPF 72–73, 124–129 (Brownfield-first, making effective use of land)",

    # --- Interaction terms (Z-variables) ---
    "Z1_Heritage_x_TB": "Balance between heritage harm and tilted balance (NPPF 202–215 + 11d)",
    "Z2_Benefits_x_TB": "Public benefits vs harms under tilted balance (NPPF 11, 202–215)",
    "Z3_Heritage_x_CA": "Local committee sensitivity to heritage harm (practice-driven)",
    "Z4_Design_x_Amenity": "Good design mitigating amenity issues (NPPF 131–139 + 96–100)",
    "Z5_GB_x_Housing": "Green Belt harm weighed against acute housing need (NPPF 11d, 142–160, 61–81)",
}


def build_interactions(X: Dict[str, Any]) -> Dict[str, float]:
    """
    Z1~Z5 interaction terms:
    - Z1 = Heritage × TB_Status
    - Z2 = (Economic + Social Benefit) × TB_Status
    - Z3 = Heritage × Committee_Attitude
    - Z4 = Design × Amenity
    - Z5 = GB_Harm × Housing_Pressure
    """
    z: Dict[str, float] = {}

    h = X.get("X1_Heritage_Harm", 0)
    d = X.get("X2_Design_Quality", 0)
    a = X.get("X3_Amenity_Harm", 0)
    econ = X.get("X7_Economic_Benefit", 0)
    soc = X.get("X8_Social_Benefit", 0)
    gb_harm = X.get("X5_GB_Harm", 0)
    hp = X.get("X11_Housing_Pressure", 0)
    tb = X.get("X12_TB_Status", 0)
    ca = X.get("X14_Committee_Attitude", 0)

    z["Z1_Heritage_x_TB"] = float(h) * float(tb)
    z["Z2_Benefits_x_TB"] = (float(econ) + float(soc)) * float(tb)
    z["Z3_Heritage_x_CA"] = float(h) * float(ca)
    z["Z4_Design_x_Amenity"] = float(d) * float(a)
    z["Z5_GB_x_Housing"] = float(gb_harm) * float(hp)

    return z


def predict_approval_probability(X_all: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based logistic-style model.
    X_all: Engine 1 + Engine 2 features (X1~X17 포함, Spin_Index 포함)

    - 정책 기반 baseline β (rule-based coefficients)를 사용해서
      Z_total, 확률, 기여도(β·X)를 계산한다.
    """

    Z = build_interactions(X_all)

    # ---- Rule-based weights (baseline β; NPPF 논리 기반) ----
    beta = {
        "X1_Heritage_Harm": -0.6,
        "X2_Design_Quality": 0.4,
        "X3_Amenity_Harm": -0.4,
        "X4_Ecology_Harm": -0.5,
        "X5_GB_Harm": -0.5,
        "X6_Flood_Risk": -0.4,
        "X7_Economic_Benefit": 0.5,
        "X8_Social_Benefit": 0.4,
        "X9_Policy_Compliance": 0.5,
        "X10_Spin_Index": -0.3,
        "X11_Housing_Pressure": 0.4,
        "X12_TB_Status": 0.5,
        "X13_Plan_Age": -0.2,
        "X14_Committee_Attitude": 0.4,
        "X15_GB_Flag": -0.3,
        "X16_FloodZone_Level": -0.2,
        # Brownfield / PDL이면 가점 (Greenfield = 0, Brownfield = 1)
        "X17_Land_Type": 0.4,
    }

    # Interaction 가중치
    gamma = {
        "Z1_Heritage_x_TB": -0.2,
        "Z2_Benefits_x_TB": 0.2,
        "Z3_Heritage_x_CA": -0.15,
        "Z4_Design_x_Amenity": 0.1,  # 좋은 디자인이 amenity harm 완화
        "Z5_GB_x_Housing": 0.15,
    }

    intercept = -0.5

    # Linear score (Z_total)
    z_linear = intercept
    contributions: List[Dict[str, Any]] = []

    # Base X-variables
    for name, coeff in beta.items():
        value = float(X_all.get(name, 0.0))
        contrib = coeff * value
        z_linear += contrib

        contributions.append(
            {
                "name": name,
                "type": "base",
                "value": value,
                "coefficient": coeff,
                "contribution": contrib,
                "abs_contribution": abs(contrib),
                "policy_ref": POLICY_LINKS.get(name, ""),
            }
        )

    # Interaction terms
    for name, coeff in gamma.items():
        value = float(Z.get(name, 0.0))
        contrib = coeff * value
        z_linear += contrib

        contributions.append(
            {
                "name": name,
                "type": "interaction",
                "value": value,
                "coefficient": coeff,
                "contribution": contrib,
                "abs_contribution": abs(contrib),
                "policy_ref": POLICY_LINKS.get(name, ""),
            }
        )

    # Probability
    prob = _sigmoid(z_linear)
    prob = max(0.0, min(1.0, prob))

    # Risk rating
    if prob >= 0.7:
        rating = "Green"
    elif prob >= 0.4:
        rating = "Amber"
    else:
        rating = "Red"

    return {
        "probability": prob,
        "rating": rating,
        "linear_score": z_linear,
        "interactions": Z,
        "contributions": contributions,
    }


def logistic_from_coeffs(
    X: Dict[str, float],
    feature_names: List[str],
    coef: List[float],
    intercept: float,
    policy_links: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generic logistic regression utility for the TRAINED model.

    Parameters
    ----------
    X : dict
        All available features for the case (e.g. X1~X17, Spin Index, etc.)
    feature_names : list[str]
        Names of features used by the trained model (order matters)
    coef : list[float]
        Coefficients learned from LogisticRegression (same order as feature_names)
    intercept : float
        Intercept term from LogisticRegression
    policy_links : dict (optional)
        Mapping from feature name to NPPF reference (for explainability)

    Returns
    -------
    dict with:
        - probability
        - linear_score
        - contributions (list of rows, similar 형식)
    """

    z_linear = intercept
    contributions: List[Dict[str, Any]] = []

    links = policy_links or {}

    for name, c in zip(feature_names, coef):
        value = float(X.get(name, 0.0))
        contrib = c * value
        z_linear += contrib

        contributions.append(
            {
                "name": name,
                "type": "trained",  # trained model에서 온 항목
                "value": value,
                "coefficient": c,
                "contribution": contrib,
                "abs_contribution": abs(contrib),
                "policy_ref": links.get(name, ""),  # 원하면 NPPF ref를 여기에도 연결 가능
            }
        )

    prob = _sigmoid(z_linear)
    prob = max(0.0, min(1.0, prob))

    return {
        "probability": prob,
        "linear_score": z_linear,
        "contributions": contributions,
    }
