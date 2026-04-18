"""Map token counts to USD using the same catalog as ``model_config.MODEL_PRICING``.

The OpenAI billing shape is per token. Our catalog stores **dollars per million tokens**
for input and output separately (see ``ModelPricing`` in ``model_config``).

For a single completion:

.. code-block:: text

    cost_usd = (prompt_tokens / 1_000_000) * input_cost_per_million_tokens
             + (completion_tokens / 1_000_000) * output_cost_per_million_tokens

Per 1k tokens (equivalent form):

.. code-block:: text

    cost_usd = (prompt_tokens / 1000) * (input_cost_per_million_tokens / 1000)
             + (completion_tokens / 1000) * (output_cost_per_million_tokens / 1000)

Unknown models fall back to ``get_model_pricing`` default resolution in ``estimate_cost_usd``.
"""

from __future__ import annotations

from model_config import estimate_cost_usd


def compute_request_cost_usd(
    *,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Return dollars for one request given split token counts and catalog rates."""
    return estimate_cost_usd(
        model_id=model_id,
        input_tokens=max(0, int(prompt_tokens)),
        output_tokens=max(0, int(completion_tokens)),
    )


def cost_label_for_session(*, byok: bool, counts_from_provider: bool) -> str:
    """Short label for sidebar / meta: BYOK + provider counts → cost; otherwise estimate."""
    if byok and counts_from_provider:
        return "Cost (your key)"
    if byok:
        return "Est. cost (your key)"
    return "Estimated cost"
