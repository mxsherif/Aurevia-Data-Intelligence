"""The Insight Agent.

Given a question and the figures Python computed, the model writes the answer in
plain language. It sees the *result*, never the dataset, and never a tool — so
the only numbers available to it are the ones already calculated.

Two guards, because a prompt alone is not a control:

1. The prompt forbids arithmetic and new figures, and forbids causal claims.
2. :mod:`app.services.grounding` then checks the reply's numbers against the
   computed set and flags anything that does not appear there.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.models.results import AnalysisResult, Insight
from app.services.grounding import GroundingReport, verify_insight
from app.services.llm_service import (
    LLMError,
    LLMService,
    LLMUsage,
    get_llm_service,
)

logger = logging.getLogger(__name__)

#: Table rows included in the prompt. The headline figures carry the answer;
#: the table is context, and sending all of it would be waste.
MAX_PROMPT_ROWS = 25


SYSTEM_PROMPT = """\
You are the explanation component of Aurevia, a data-analysis platform. Python \
has already run the analysis. Your job is to state what the computed results \
say, in plain language.

ABSOLUTE RULE ON NUMBERS
Only use numerical values contained in the supplied computed results. Do not \
calculate, infer, estimate, or invent additional numbers. Do not add, subtract, \
divide, average, convert or re-scale any figure. If a number you want is not \
in the results, describe the finding without a number.

WHAT TO WRITE
- `answer`: one or two sentences answering the user's question directly, \
quoting the key figure from the results.
- `observations`: one to three further points that the results themselves show \
-- a gap between groups, a direction of travel, a concentration. Each must be \
checkable against the supplied figures.
- `caveat`: one short caution when the results warrant it (missing values, few \
rows, a correlation being association rather than cause, an outlier skewing a \
mean). Otherwise null.
- `follow_up_questions`: two to four questions this same dataset could answer \
next. Phrase them as a user would type them.

CAUSATION
Describe what the data shows, never why it is so. "Month-to-month contracts \
have the highest churn rate at 23.9%" is correct. "Month-to-month customers \
churn more because they are less committed" is not: the dataset contains no \
evidence of motive. Do not explain causes, motives or business reasons.

STYLE
Be concise and specific. No preamble, no restating the question, no bullet \
markers inside the strings. Name columns as a reader would say them, not as \
raw identifiers.
"""


#: Added on a retry, when the first explanation failed validation. Re-sending
#: the identical prompt and hoping for a different answer is not a retry.
RETRY_REMINDER = """\
A previous attempt at this explanation was rejected because it referenced \
figures that do not appear in the computed results, or did not answer the \
question that was asked.

Write it again, and this time:
- every number you write must appear verbatim in the supplied results -- copy \
them, do not recompute, convert or re-round them;
- if a figure you want is not in the results, write the sentence without it;
- answer the user's question directly in the first sentence.
"""


@dataclass
class InsightOutcome:
    """An insight, or the reason there isn't one."""

    insight: Insight | None = None
    error: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    grounding: GroundingReport | None = None

    @property
    def ok(self) -> bool:
        return self.insight is not None

    @property
    def is_grounded(self) -> bool:
        return self.grounding is None or self.grounding.is_grounded


class InsightAgent:
    """Turns an :class:`AnalysisResult` into a written answer."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or get_llm_service()

    @property
    def available(self) -> bool:
        return self._llm.available

    def explain(
        self,
        question: str,
        result: AnalysisResult,
        *,
        dataset_name: str = "the dataset",
        plan_steps: list[str] | None = None,
        stricter: bool = False,
    ) -> InsightOutcome:
        """Explain `result` in answer to `question`.

        `stricter` is set when a previous attempt failed validation; it adds a
        pointed reminder to the prompt rather than re-sending the same request.

        Never raises. The returned insight has been grounding-checked, and
        :attr:`InsightOutcome.grounding` records the outcome.
        """
        if not result.success:
            return InsightOutcome(error="There is no successful result to explain.")
        if not self._llm.available:
            return InsightOutcome(error=self._llm.unavailable_reason)

        payload = self._payload(question, result, dataset_name, plan_steps)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        if stricter:
            messages.append({"role": "system", "content": RETRY_REMINDER})
        messages.append({"role": "user", "content": payload})

        try:
            response = self._llm.complete_structured(
                messages=messages, schema=Insight
            )
        except LLMError as exc:
            logger.info("Insight generation failed: %s", exc)
            return InsightOutcome(error=str(exc))
        except Exception as exc:  # noqa: BLE001 - this method promises not to raise
            logger.exception("Unexpected failure generating an explanation")
            return InsightOutcome(
                error=(
                    f"an unexpected error occurred ({type(exc).__name__}); "
                    "the details were logged"
                )
            )

        insight = response.data
        if not isinstance(insight, Insight):  # pragma: no cover - defensive
            return InsightOutcome(error="The explanation step returned no content.")

        report = verify_insight(insight, result)
        if not report.is_grounded:
            logger.warning(
                "Ungrounded figures in the answer to %r: %s",
                question, report.ungrounded,
            )

        return InsightOutcome(
            insight=insight, usage=response.usage, grounding=report
        )

    # -- prompt construction ----------------------------------------------- #

    @staticmethod
    def _payload(
        question: str,
        result: AnalysisResult,
        dataset_name: str,
        plan_steps: list[str] | None,
    ) -> str:
        """The compact, computed-values-only message sent to the model."""
        body: dict[str, object] = {
            "question": question,
            "dataset": dataset_name,
            "computed_results": result.grounding_payload(max_rows=MAX_PROMPT_ROWS),
        }
        if plan_steps:
            body["analysis_performed"] = plan_steps
        return json.dumps(body, separators=(",", ":"), default=str)


def explain_result(
    question: str,
    result: AnalysisResult,
    *,
    llm: LLMService | None = None,
    dataset_name: str = "the dataset",
    plan_steps: list[str] | None = None,
    stricter: bool = False,
) -> InsightOutcome:
    """Convenience wrapper around :class:`InsightAgent`."""
    return InsightAgent(llm).explain(
        question, result, dataset_name=dataset_name, plan_steps=plan_steps,
        stricter=stricter,
    )


__all__ = [
    "MAX_PROMPT_ROWS",
    "RETRY_REMINDER",
    "SYSTEM_PROMPT",
    "InsightAgent",
    "InsightOutcome",
    "explain_result",
]
