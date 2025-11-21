# engine0_rulebook.py
"""
Engine 0 – NPPF Rulebook
NPPF 기반 심각도/혜택 점수를 숫자로 바꾸는 규칙 모음.
(실제 NPPF 조항과 1:1 대응이라기보다는 구조를 맞춘 프로토타입)
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Scale:
    """텍스트 패턴 → 점수 매핑용 간단 스케일."""
    patterns: List[str]
    score: int


# Heritage (0~3)
HERITAGE_SCALES = [
    Scale(["substantial harm"], 3),
    Scale(["less than substantial harm", "ltsh"], 2),
    Scale(["harm to significance"], 2),
    Scale(["no harm", "no heritage harm"], 0),
]

# Design (-3~+3)
DESIGN_POSITIVE = ["high quality design", "excellent design", "outstanding design"]
DESIGN_GOOD = ["good design", "well-designed", "responds to context"]
DESIGN_NEGATIVE = ["poor design", "overly dominant", "out of keeping", "incongruous"]

# Amenity (0~3)
AMENITY_HARM_PATTERNS = [
    Scale(["loss of privacy", "overlooking"], 2),
    Scale(["overbearing", "over-dominant"], 2),
    Scale(["noise impact", "unacceptable noise"], 2),
    Scale(["significant adverse impact on residential amenity"], 3),
]

# Ecology (0~3)
ECOLOGY_SCALES = [
    Scale(["irreplaceable habitat", "ancient woodland"], 3),
    Scale(["significant harm to biodiversity", "net loss of biodiversity"], 2),
    Scale(["biodiversity net gain", "net gain for biodiversity"], -1),  # 혜택이므로 -
]

# Economic / Social benefits (0~3)
ECON_BENEFIT_WORDS = ["jobs", "employment", "investment", "economic benefit"]
SOCIAL_BENEFIT_WORDS = ["affordable housing", "public open space", "community facility"]

# Policy compliance (-3~+3)
POLICY_COMPLIANCE_POS = ["accords with policy", "complies with policy", "in accordance with policy"]
POLICY_COMPLIANCE_NEG = ["contrary to policy", "conflicts with policy", "non-compliant"]


def simple_keyword_score(text: str, patterns: List[str]) -> int:
    t = text.lower()
    return sum(1 for p in patterns if p in t)


def apply_scales(text: str, scales: List[Scale], default: int = 0) -> int:
    t = text.lower()
    best = default
    for s in scales:
        if any(p in t for p in s.patterns):
            best = max(best, s.score)
    return best


def rulebook_scores(text: str) -> Dict[str, int]:
    """
    하나의 문서(text)에 대해 X1~X9 기본 점수(0~3 또는 -3~+3)를 대략 추정.
    - X1 Heritage_Harm
    - X2 Design_Quality
    - X3 Amenity_Harm
    - X4 Ecology_Harm
    - X7 Economic_Benefit
    - X8 Social_Benefit
    - X9 Policy_Compliance
    나머지는 Engine 1/2에서 추가로 세팅.
    """
    t = text.lower()

    heritage_harm = apply_scales(t, HERITAGE_SCALES, default=0)

    # Design
    design_quality = 0
    if any(p in t for p in DESIGN_POSITIVE):
        design_quality = 3
    elif any(p in t for p in DESIGN_GOOD):
        design_quality = 1
    if any(p in t for p in DESIGN_NEGATIVE):
        design_quality = min(design_quality - 2, -3)

    # Amenity
    amenity_harm = apply_scales(t, AMENITY_HARM_PATTERNS, default=0)

    # Ecology
    ecology_harm = apply_scales(t, ECOLOGY_SCALES, default=0)

    # Economic / Social benefit
    econ_benefit = min(3, simple_keyword_score(t, ECON_BENEFIT_WORDS))
    social_benefit = min(3, simple_keyword_score(t, SOCIAL_BENEFIT_WORDS))

    # Policy compliance
    pc_pos = simple_keyword_score(t, POLICY_COMPLIANCE_POS)
    pc_neg = simple_keyword_score(t, POLICY_COMPLIANCE_NEG)
    policy_compliance = max(min(pc_pos - pc_neg, 3), -3)

    return {
        "Heritage_Harm": heritage_harm,
        "Design_Quality": design_quality,
        "Amenity_Harm": amenity_harm,
        "Ecology_Harm": ecology_harm,
        "Economic_Benefit": econ_benefit,
        "Social_Benefit": social_benefit,
        "Policy_Compliance": policy_compliance,
    }
