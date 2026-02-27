import json
import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage

from src.orchestrai.agents.orchestrator import orchestrator_agent_func


ROOT_DIR = Path(__file__).resolve().parents[1]
GOLDENS_PATH = ROOT_DIR / "evals" / "agent_goldens.json"


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


def _run_orchestrator_agent(conversation: list[str]) -> dict:
    messages = [HumanMessage(content=turn.strip()) for turn in conversation if turn.strip()]
    result = orchestrator_agent_func(messages)
    return result.model_dump(mode="json")


def _agent_output_for_eval(raw_output: dict) -> str:
    action = raw_output.get("action")
    has_email = bool(raw_output.get("email_details"))
    has_calendar = bool(raw_output.get("calendar_details"))
    clarification_message = (raw_output.get("clarification_message") or "").strip()
    return (
        f"action={action}; "
        f"has_email_details={has_email}; "
        f"has_calendar_details={has_calendar}; "
        f"clarification_message={clarification_message}"
    )


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
def test_original_orchestrator_agent(golden: dict):
    raw_output = _run_orchestrator_agent(golden["conversation"])
    actual_output = _agent_output_for_eval(raw_output)

    expected_output = (
        f"The agent action should be '{golden['expected_action']}'. "
        f"Behavior expectation: {golden['expected_behavior']}"
    )

    test_case = LLMTestCase(
        input="\n".join(golden["conversation"]),
        actual_output=actual_output,
        expected_output=expected_output,
    )

    metrics = [
        GEval(
            name="Agent Action Correctness",
            evaluation_steps=[
                "Read the expected action in expected output.",
                "Verify whether the action in actual output matches exactly.",
                "Heavily penalize any mismatch in action.",
                "Only give high score when action is correct and consistent with the user request.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.8,
        ),
        GEval(
            name="Agent Behavior Alignment",
            evaluation_steps=[
                "Compare actual output fields to expected behavior description.",
                "Check if details presence/absence is appropriate for the selected action.",
                "If clarification is expected, verify a clarification message is present.",
                "Reward concise, structurally coherent agent output.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.65,
        ),
    ]

    assert_test(test_case, metrics)
