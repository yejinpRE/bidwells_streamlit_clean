# engine1_text.py
from typing import Dict, Any, Tuple, Optional
from pypdf import PdfReader

from engine0_rulebook import rulebook_scores


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def base_scores_from_text(text: str) -> Dict[str, Any]:
    """
    Engine 0 rulebook 점수 + flood, GB 등 추가 플래그.
    """
    base = rulebook_scores(text)
    t = text.lower()

    # X5 Green Belt Harm (rough)
    gb_harm = 0
    if "green belt" in t:
        gb_harm = -2  # GB에서 개발 자체가 harm 가정
        if "very special circumstances" in t:
            gb_harm = -1
    base["GB_Harm"] = gb_harm

    # X6 Flood risk (0~3)
    flood_risk = 0
    if "flood zone 3" in t:
        flood_risk = 3
    elif "flood zone 2" in t:
        flood_risk = 2
    elif "flood risk" in t:
        flood_risk = 1
    base["Flood_Risk"] = flood_risk

    # 간단 word count
    base["Word_Count"] = len(text.split())

    return base


def spin_index(ps_scores: Dict[str, Any], cr_scores: Dict[str, Any]) -> float:
    """
    PS vs CR 주요 변수 차이의 평균 절대값 = Spin Index (X10)
    """
    keys = [
        "Heritage_Harm",
        "Design_Quality",
        "Amenity_Harm",
        "Ecology_Harm",
        "GB_Harm",
        "Flood_Risk",
        "Economic_Benefit",
        "Social_Benefit",
        "Policy_Compliance",
    ]
    diffs = []
    for k in keys:
        if k in ps_scores and k in cr_scores:
            diffs.append(abs(ps_scores[k] - cr_scores[k]))
    return float(sum(diffs) / len(diffs)) if diffs else 0.0


def engine1_run(
    ps_text: Optional[str],
    cr_text: Optional[str],
    appeal_text: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Engine 1 전체 실행:
    - PS / CR / Appeal 각각 점수 계산
    - Spin Index까지 포함된 'CR 기반 X-variables' 반환
    """

    ps_scores = base_scores_from_text(ps_text) if ps_text else {}
    cr_scores = base_scores_from_text(cr_text) if cr_text else {}
    ap_scores = base_scores_from_text(appeal_text) if appeal_text else {}

    # X1~X9: 기본적으로 CR(Committee/Officer Report)을 기준으로 사용
    ref = cr_scores or ps_scores  # CR 없으면 PS로 fallback

    X: Dict[str, Any] = {}
    X["X1_Heritage_Harm"] = ref.get("Heritage_Harm", 0)
    X["X2_Design_Quality"] = ref.get("Design_Quality", 0)
    X["X3_Amenity_Harm"] = ref.get("Amenity_Harm", 0)
    X["X4_Ecology_Harm"] = ref.get("Ecology_Harm", 0)
    X["X5_GB_Harm"] = ref.get("GB_Harm", 0)
    X["X6_Flood_Risk"] = ref.get("Flood_Risk", 0)
    X["X7_Economic_Benefit"] = ref.get("Economic_Benefit", 0)
    X["X8_Social_Benefit"] = ref.get("Social_Benefit", 0)
    X["X9_Policy_Compliance"] = ref.get("Policy_Compliance", 0)

    # X10 – Spin Index
    if ps_scores and cr_scores:
        X["X10_Spin_Index"] = spin_index(ps_scores, cr_scores)
    else:
        X["X10_Spin_Index"] = 0.0

    return ps_scores, cr_scores, {**X, "Appeal_Scores": ap_scores}
