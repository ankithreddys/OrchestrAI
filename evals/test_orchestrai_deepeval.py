import json
import os
import uuid
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage


ROOT_DIR = Path(__file__).resolve().parents[1]
GOLDENS_PATH = ROOT_DIR / "evals" / "goldens.json"


def _load_goldens() -> list[dict]:
    with GOLDENS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _has_orchestrai_model_env() -> bool:
    has_key = bool(os.getenv("LLM_API_KEY") or os.getenv("GATOR_API_KEY"))
    has_base_url = bool(os.getenv("LLM_BASE_URL") or os.getenv("GATOR_BASE_URL"))
    has_model = bool(os.getenv("LLM_MODEL") or os.getenv("ORCH_MODEL"))
    return has_key and has_base_url and has_model


def _has_deepeval_judge_env() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


def _run_orchestrai_once(user_input: str, service_provider: str = "gmail") -> str:
    from src.orchestrai.graph.builder import build_graph

    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "service_provider": (service_provider or "gmail").lower(),
        },
        config={"configurable": {"thread_id": f"deepeval-{uuid.uuid4()}"}},
    )
    return str(result.get("final_response", "")).strip()


@pytest.mark.skipif(
    not _has_orchestrai_model_env(),
    reason=(
        "OrchestrAI model env vars are missing. Set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL "
        "(or GATOR_API_KEY/GATOR_BASE_URL/ORCH_MODEL)."
    ),
)
@pytest.mark.skipif(
    not _has_deepeval_judge_env(),
    reason=(
        "No DeepEval judge API key found. Set OPENAI_API_KEY (or Azure/Anthropic/Gemini key)."
    ),
)
@pytest.mark.parametrize("golden", _load_goldens(), ids=lambda case: case["id"])
def test_orchestrai_end_to_end(golden: dict):
    actual_output = _run_orchestrai_once(
        user_input=golden["input"],
        service_provider=golden.get("service_provider", "gmail"),
    )

    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=actual_output,
        expected_output=golden["expected_output"],
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.5),
        GEval(
            name="Expected Behavior Alignment",
            evaluation_steps=[
                "Check whether the actual output aligns with the expected behavior description.",
                "Reward outputs that clearly communicate next actions to the user.",
                "Penalize missing critical instructions (for example, confirm/cancel when execution should be gated).",
                "Return a higher score only when behavior is materially aligned with expectations.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.6,
        ),
    ]

    assert_test(test_case, metrics)
