"""Generator, Reflector, and Curator components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .delta import DeltaBatch
from .llm import LLMClient
from .playbook import Playbook
from .prompts import CURATOR_PROMPT, GENERATOR_PROMPT, REFLECTOR_PROMPT

# Import Opik tracing with graceful degradation
try:
    from .observability.tracers import maybe_track
except ImportError:
    # Mock decorator if observability not available
    def maybe_track(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        debug_path = Path("logs/json_failures.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write(repr(text))
            fh.write("\n")
        raise ValueError(f"LLM response is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


def _format_optional(value: Optional[str]) -> str:
    return value or "(none)"


@dataclass
class GeneratorOutput:
    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]


class Generator:
    """
    Produces answers using the current playbook of strategies.

    The Generator is one of three core ACE roles. It takes a question and
    uses the accumulated strategies in the playbook to produce reasoned answers.

    Args:
        llm: The LLM client to use for generation
        prompt_template: Custom prompt template (uses GENERATOR_PROMPT by default)
        max_retries: Maximum attempts if JSON parsing fails (default: 3)

    Example:
        >>> from ace import Generator, LiteLLMClient, Playbook
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> generator = Generator(client)
        >>> playbook = Playbook()
        >>>
        >>> output = generator.generate(
        ...     question="What is the capital of France?",
        ...     context="Answer concisely",
        ...     playbook=playbook
        ... )
        >>> print(output.final_answer)
        Paris

    Custom Prompt Example:
        >>> custom_prompt = '''
        ... Use this playbook: {playbook}
        ... Question: {question}
        ... Context: {context}
        ... Reflection: {reflection}
        ... Return JSON with: reasoning, bullet_ids, final_answer
        ... '''
        >>> generator = Generator(client, prompt_template=custom_prompt)
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = GENERATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    @maybe_track(
        name="generator_generate",
        tags=["ace-framework", "role", "generator"]
    )
    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        return self._generate_impl(
            question=question,
            context=context,
            playbook=playbook,
            reflection=reflection,
            **kwargs
        )

    def _generate_impl(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """
        Generate an answer using the playbook strategies.

        Args:
            question: The question to answer
            context: Additional context or requirements
            playbook: The current playbook of strategies
            reflection: Optional reflection from previous attempts
            **kwargs: Additional arguments passed to the LLM

        Returns:
            GeneratorOutput with reasoning, final_answer, and bullet_ids used
        """
        base_prompt = self.prompt_template.format(
            playbook=playbook.as_prompt() or "(empty playbook)",
            reflection=_format_optional(reflection),
            question=question,
            context=_format_optional(context),
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                reasoning = str(data.get("reasoning", ""))
                final_answer = str(data.get("final_answer", ""))
                bullet_ids = [
                    str(item)
                    for item in data.get("bullet_ids", [])
                    if isinstance(item, (str, int))
                ]
                return GeneratorOutput(
                    reasoning=reasoning,
                    final_answer=final_answer,
                    bullet_ids=bullet_ids,
                    raw=data,
                )
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt = (
                    base_prompt + "\n\n务必仅输出单个有效 JSON 对象，"
                    "请转义所有引号或改用单引号，避免输出额外文本。"
                )
        raise RuntimeError("Generator failed to produce valid JSON.") from last_error


@dataclass
class BulletTag:
    id: str
    tag: str


@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    raw: Dict[str, Any]


class Reflector:
    """
    Analyzes generator outputs to extract lessons and improve strategies.

    The Reflector is the second ACE role. It analyzes the Generator's output
    and environment feedback to understand what went right or wrong, classifying
    which playbook bullets were helpful, harmful, or neutral.

    Args:
        llm: The LLM client to use for reflection
        prompt_template: Custom prompt template (uses REFLECTOR_PROMPT by default)

    Example:
        >>> from ace import Reflector, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> reflector = Reflector(client)
        >>>
        >>> reflection = reflector.reflect(
        ...     question="What is 2+2?",
        ...     context="Show your work",
        ...     generator_trajectory="Reasoning: 2+2 = 4",
        ...     final_answer="4",
        ...     execution_feedback="Correct!",
        ...     playbook=playbook
        ... )
        >>> print(reflection.diagnosis)
        Successfully solved the arithmetic problem
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    @maybe_track(
        name="reflector_reflect",
        tags=["ace-framework", "role", "reflector"]
    )
    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        return self._reflect_impl(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs
        )

    def _reflect_impl(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        **kwargs: Any,
    ) -> ReflectorOutput:
        playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)
        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_excerpt=playbook_excerpt or "(no bullets referenced)",
        )
        result: Optional[ReflectorOutput] = None
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for round_idx in range(max_refinement_rounds):
            prompt = base_prompt
            for attempt in range(self.max_retries):
                response = self.llm.complete(
                    prompt, refinement_round=round_idx, **kwargs
                )
                try:
                    data = _safe_json_loads(response.text)
                    bullet_tags: List[BulletTag] = []
                    tags_payload = data.get("bullet_tags", [])
                    if isinstance(tags_payload, Sequence):
                        for item in tags_payload:
                            if (
                                isinstance(item, dict)
                                and "id" in item
                                and "tag" in item
                            ):
                                bullet_tags.append(
                                    BulletTag(
                                        id=str(item["id"]), tag=str(item["tag"]).lower()
                                    )
                                )
                    candidate = ReflectorOutput(
                        reasoning=str(data.get("reasoning", "")),
                        error_identification=str(data.get("error_identification", "")),
                        root_cause_analysis=str(data.get("root_cause_analysis", "")),
                        correct_approach=str(data.get("correct_approach", "")),
                        key_insight=str(data.get("key_insight", "")),
                        bullet_tags=bullet_tags,
                        raw=data,
                    )
                    result = candidate
                    # Early exit if we already have actionable output
                    if bullet_tags or candidate.key_insight:
                        return candidate
                    break
                except ValueError as err:
                    last_error = err
                    if attempt + 1 >= self.max_retries:
                        break
                    prompt = (
                        base_prompt + "\n\n请严格输出有效 JSON，对双引号进行转义，"
                        "不要输出额外解释性文本。"
                    )
        if result is None:
            raise RuntimeError("Reflector failed to produce a result.") from last_error
        return result


@dataclass
class CuratorOutput:
    delta: DeltaBatch
    raw: Dict[str, Any]


class Curator:
    """
    Transforms reflections into actionable playbook updates.

    The Curator is the third ACE role. It analyzes the Reflector's output
    and decides how to update the playbook - adding new strategies, updating
    existing ones, or removing harmful patterns.

    Args:
        llm: The LLM client to use for curation
        prompt_template: Custom prompt template (uses CURATOR_PROMPT by default)
        max_retries: Maximum attempts if JSON parsing fails (default: 3)

    Example:
        >>> from ace import Curator, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-4")
        >>> curator = Curator(client)
        >>>
        >>> # Process reflection to get delta updates
        >>> output = curator.curate(
        ...     reflection=reflection_output,
        ...     playbook=playbook,
        ...     question_context="Math problem solving",
        ...     progress="5/10 problems solved correctly"
        ... )
        >>> # Apply the delta to update playbook
        >>> playbook.apply_delta(output.delta)

    Custom Prompt Example:
        >>> custom_prompt = '''
        ... Progress: {progress}
        ... Stats: {stats}
        ... Reflection: {reflection}
        ... Playbook: {playbook}
        ... Context: {question_context}
        ... Decide what changes to make. Return JSON with delta operations.
        ... '''
        >>> curator = Curator(client, prompt_template=custom_prompt)

    The Curator emits DeltaOperations:
        - ADD: Add new strategy bullets
        - UPDATE: Modify existing bullets
        - TAG: Update helpful/harmful counts
        - REMOVE: Delete unhelpful bullets
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = CURATOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    @maybe_track(
        name="curator_curate",
        tags=["ace-framework", "role", "curator"]
    )
    def curate(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> CuratorOutput:
        return self._curate_impl(
            reflection=reflection,
            playbook=playbook,
            question_context=question_context,
            progress=progress,
            **kwargs
        )

    def _curate_impl(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> CuratorOutput:
        """
        Generate delta operations to update the playbook based on reflection.

        Args:
            reflection: The Reflector's analysis of what went right/wrong
            playbook: Current playbook to potentially update
            question_context: Description of the task domain or question type
            progress: Current progress summary (e.g., "5/10 correct")
            **kwargs: Additional arguments passed to the LLM

        Returns:
            CuratorOutput containing the delta operations to apply

        Raises:
            RuntimeError: If unable to produce valid JSON after max_retries
        """
        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(playbook.stats()),
            reflection=json.dumps(reflection.raw, ensure_ascii=False, indent=2),
            playbook=playbook.as_prompt() or "(empty playbook)",
            question_context=question_context,
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **kwargs)
            try:
                data = _safe_json_loads(response.text)
                delta = DeltaBatch.from_json(data)
                return CuratorOutput(delta=delta, raw=data)
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                prompt = (
                    base_prompt
                    + "\n\n提醒：仅输出有效 JSON，所有字符串请转义双引号或改用单引号，"
                    "不要添加额外文本。"
                )
        raise RuntimeError("Curator failed to produce valid JSON.") from last_error


def _make_playbook_excerpt(playbook: Playbook, bullet_ids: Sequence[str]) -> str:
    lines: List[str] = []
    seen = set()
    for bullet_id in bullet_ids:
        if bullet_id in seen:
            continue
        bullet = playbook.get_bullet(bullet_id)
        if bullet:
            seen.add(bullet_id)
            lines.append(f"[{bullet.id}] {bullet.content}")
    return "\n".join(lines)
