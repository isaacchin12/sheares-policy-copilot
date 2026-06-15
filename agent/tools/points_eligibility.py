"""
Points eligibility checker for Sheares Hall room retention priority.

Tool: check_points_eligibility
Determines tier (Priority A / B / C / Ineligible) based on total activity points.
"""

from __future__ import annotations

from pydantic import BaseModel


class PointsInput(BaseModel):
    total_points: int
    has_supplementary: bool = False
    supplementary_points: int = 0


class PointsResult(BaseModel):
    tier: str          # "Priority A" | "Priority B" | "Priority C" | "Ineligible"
    points_total: int
    eligible_for_retention: bool
    next_tier_gap: int | None  # points needed to reach next tier, or None if at top
    explanation: str


async def check_points_eligibility(input: PointsInput) -> PointsResult:
    """
    Evaluate a resident's room retention priority tier.

    Tiers:
        Priority A  ≥ 80 points
        Priority B  ≥ 50 points
        Priority C  ≥ 30 points
        Ineligible  < 30 points

    Supplementary points (e.g. welfare, special commendation) are added to the
    base total when has_supplementary is True.
    """
    total = input.total_points + (
        input.supplementary_points if input.has_supplementary else 0
    )

    if total >= 80:
        return PointsResult(
            tier="Priority A",
            points_total=total,
            eligible_for_retention=True,
            next_tier_gap=None,
            explanation=f"With {total} points you qualify for Priority A room retention.",
        )
    elif total >= 50:
        return PointsResult(
            tier="Priority B",
            points_total=total,
            eligible_for_retention=True,
            next_tier_gap=80 - total,
            explanation=(
                f"With {total} points you qualify for Priority B. "
                f"You need {80 - total} more points for Priority A."
            ),
        )
    elif total >= 30:
        return PointsResult(
            tier="Priority C",
            points_total=total,
            eligible_for_retention=True,
            next_tier_gap=50 - total,
            explanation=(
                f"With {total} points you qualify for Priority C. "
                f"You need {50 - total} more for Priority B."
            ),
        )
    else:
        return PointsResult(
            tier="Ineligible",
            points_total=total,
            eligible_for_retention=False,
            next_tier_gap=30 - total,
            explanation=(
                f"With {total} points you do not qualify for priority retention. "
                f"You need {30 - total} more points."
            ),
        )
