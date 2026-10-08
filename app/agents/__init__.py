"""Aurevia's LLM agents.

Two agents, each with one narrow job and no ability to compute:

- :mod:`app.agents.planner` turns a question into a structured, validated plan.
- :mod:`app.agents.insights` turns computed results into written findings.

:mod:`app.agents.suggestions` sits alongside them but uses **no LLM at all**:
suggested questions are derived from the dataset schema by deterministic rules.
"""

from app.agents.insights import InsightAgent, InsightOutcome, explain_result
from app.agents.investigation_planner import (
    InvestigationPlannerAgent,
    InvestigationSummaryAgent,
)
from app.agents.planner import PlannerAgent, PlannerOutcome, build_plan
from app.agents.suggestions import (
    DatasetRoles,
    detect_roles,
    follow_up_questions,
    suggest_questions,
)

__all__ = [
    "DatasetRoles",
    "InsightAgent",
    "InsightOutcome",
    "InvestigationPlannerAgent",
    "InvestigationSummaryAgent",
    "PlannerAgent",
    "PlannerOutcome",
    "build_plan",
    "detect_roles",
    "explain_result",
    "follow_up_questions",
    "suggest_questions",
]
