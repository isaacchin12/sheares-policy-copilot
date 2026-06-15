"""
Exam period subsidy calculator for Sheares Hall residents.

Tool: calculate_subsidy
Calculates daily rate and total subsidy based on stay duration, points, and deadline.
"""

from __future__ import annotations

from pydantic import BaseModel


class SubsidyInput(BaseModel):
    days_staying: int
    points_total: int
    applied_before_deadline: bool


class SubsidyResult(BaseModel):
    eligible: bool
    daily_rate: float | None
    total_subsidy: float | None
    reason: str


async def calculate_subsidy(input: SubsidyInput) -> SubsidyResult:
    """
    Calculate the exam-period accommodation subsidy.

    Eligibility requirements:
        - Minimum 30 activity points
        - Application submitted before the last Friday of the exam period

    Daily rates:
        ≤ 7 days   → $8.00/day
        8–14 days  → $7.00/day
        > 14 days  → $6.00/day
    """
    if input.points_total < 30:
        return SubsidyResult(
            eligible=False,
            daily_rate=None,
            total_subsidy=None,
            reason="Minimum 30 points required for subsidy eligibility.",
        )

    if not input.applied_before_deadline:
        return SubsidyResult(
            eligible=False,
            daily_rate=None,
            total_subsidy=None,
            reason=(
                "Application deadline missed. "
                "Subsidies require submission before the last Friday of exam period."
            ),
        )

    if input.days_staying <= 7:
        rate = 8.0
    elif input.days_staying <= 14:
        rate = 7.0
    else:
        rate = 6.0

    total = rate * input.days_staying

    return SubsidyResult(
        eligible=True,
        daily_rate=rate,
        total_subsidy=total,
        reason=(
            f"Eligible at ${rate}/day × {input.days_staying} days = ${total:.2f}"
        ),
    )
