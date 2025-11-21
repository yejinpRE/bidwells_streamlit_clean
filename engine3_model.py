# engine3_model.py
from typing import Dict, Any
import math


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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

    z["Z1_Heritage_x_TB"] = h * tb
    z["Z2_Benefits_x_TB"] = (econ + soc) * tb
    z["Z3_Heritage_x_CA"] = h * ca
    z["Z4_Design_x_Amenity"] = d * a
    z["Z5_GB_x_Housing"] = gb_harm * hp

    return z


def predict_approval_probability(X_all: Dict[str, Any]) -> Dict[str, Any]:
    """
    X_all: Engine 1 + Engine 2 feature 합친 것 (X1~X16 + Spin_Index 포함)
    로지스틱 형태 수식으로 승인 확률 계산.
    """

    Z = build_interactions(X_all)

    # ---- 가중치 (데모용, 나중에 실제 회귀로 대체 가능) ----
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
    }

    gamma = {
        "Z1_Heritage_x_TB": -0.2,
        "Z2_Benefits_x_TB": 0.2,
        "Z3_Heritage_x_CA": -0.15,
        "Z4_Design_x_Amenity": 0.1,  # 좋은 디자인이 amenity harm 완화
        "Z5_GB_x_Housing": 0.15,
    }

    intercept = -0.5

    z_linear = intercept
    for k, v in beta.items():
        z_linear += v * float(X_all.get(k, 0.0))
    for k, v in gamma.items():
        z_linear += v * float(Z.get(k, 0.0))

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
    }
