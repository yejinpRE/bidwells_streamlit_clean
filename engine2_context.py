# engine2_context.py
from typing import Dict, Any


def build_context_features(
    housing_pressure: float,
    tb_status: int,
    plan_age: int,
    committee_attitude: float,
    gb_flag: int,
    floodzone_level: int,
) -> Dict[str, Any]:
    """
    Engine 2 – Context features (X11~X16)
    - housing_pressure: 0~3 (높을수록 housing 필요가 큼)
    - tb_status: 0=none, 1=arguable, 2=tilted balance 명시
    - plan_age: 0=up-to-date, 1=중간, 2=낡음
    - committee_attitude: 0~3 (0=매우 보수적, 3=매우 permissive)
    - gb_flag: 0/1
    - floodzone_level: 1,2,3 (또는 0=미지정)
    """

    X: Dict[str, Any] = {}
    X["X11_Housing_Pressure"] = housing_pressure
    X["X12_TB_Status"] = tb_status
    X["X13_Plan_Age"] = plan_age
    X["X14_Committee_Attitude"] = committee_attitude
    X["X15_GB_Flag"] = gb_flag
    X["X16_FloodZone_Level"] = floodzone_level
    return X
