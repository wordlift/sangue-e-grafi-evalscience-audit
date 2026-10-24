"""Ontology-driven reward functions for GRPO training.

This module provides the **key innovation** over SEOcrate's LLM-as-Judge
approach: rather than asking another LLM to score completions, we validate
agent traversal paths directly against the kinship OWL axioms.

Reward components
-----------------
1. **Format** (+1.0)  — valid XML tags (``<reasoning>``, ``<action>``, ``<answer>``)
2. **Ontological path** (variable) — each ``<action>`` scored against will
   clause ``requires``/``excludes`` and OWL axioms (symmetry, inverse, etc.)
3. **Correct answer** (+5.0 / −1.0) — gold-standard comparison
4. **Efficiency** (+1.0 / +0.5) — hop count vs. known optimal
5. **Dynamic Hop Control** (+0.5) — early termination within optimal+1 hops

All rewards are floored at 0.0.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace, OWL, RDF, RDFS

# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------

EX = Namespace("http://example.org/kinship#")

# ---------------------------------------------------------------------------
# Biological-lineage properties (properties that imply blood relation)
# ---------------------------------------------------------------------------

_BIOLOGICAL_LINEAGE_PROPERTIES: frozenset[str] = frozenset({
    str(EX.isBiologicalParentOf),
    str(EX.hasBiologicalParent),
})


# ═══════════════════════════════════════════════════════════════════════════
# 1.  OntologyRewardValidator
# ═══════════════════════════════════════════════════════════════════════════


class OntologyRewardValidator:
    """Score traversal steps against OWL axioms extracted from the ontology.

    On construction, the kinship Turtle file is parsed once and four lookup
    structures are pre-computed so that per-step validation is O(1) dict /
    set lookups.

    Args:
        ontology_path: Filesystem path to ``kinship.ttl``.

    Raises:
        FileNotFoundError: If *ontology_path* does not exist.
    """

    def __init__(self, ontology_path: str) -> None:
        path = Path(ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {path}")

        self._graph = Graph()
        self._graph.parse(str(path), format="turtle")

        # Pre-compute axiom caches
        self.symmetric_properties: set[str] = self._extract_symmetric()
        self.inverse_pairs: dict[str, str] = self._extract_inverses()
        self.property_domains: dict[str, str] = self._extract_domains()
        self.property_ranges: dict[str, str] = self._extract_ranges()

    # -- private extraction helpers ------------------------------------------

    def _extract_symmetric(self) -> set[str]:
        """Collect all properties declared as ``owl:SymmetricProperty``."""
        return {
            str(s)
            for s in self._graph.subjects(RDF.type, OWL.SymmetricProperty)
        }

    def _extract_inverses(self) -> dict[str, str]:
        """Build a bidirectional map of ``owl:inverseOf`` pairs."""
        pairs: dict[str, str] = {}
        for s, o in self._graph.subject_objects(OWL.inverseOf):
            pairs[str(s)] = str(o)
            pairs[str(o)] = str(s)
        return pairs

    def _extract_domains(self) -> dict[str, str]:
        """Map each property to its ``rdfs:domain`` class URI."""
        return {
            str(s): str(o)
            for s, o in self._graph.subject_objects(RDFS.domain)
        }

    def _extract_ranges(self) -> dict[str, str]:
        """Map each property to its ``rdfs:range`` class URI."""
        return {
            str(s): str(o)
            for s, o in self._graph.subject_objects(RDFS.range)
        }

    # -- public scoring API --------------------------------------------------

    def validate_traversal_step(
        self,
        property_used: str,
        will_clause: dict[str, Any],
    ) -> float:
        """Score a single traversal hop against the will's axiom constraints.

        Scoring rules (applied in priority order):

        - Property is in the will's ``excludes`` list → **−3.0**
          (ontological violation — the will explicitly forbids this relation).
        - Property is symmetric AND the will requires biological lineage
          → **−2.0** (symmetric props like ``isMarriedTo`` can never
          establish lineage).
        - Property is in the will's ``requires`` list → **+1.0**
          (traversal follows the intended kinship axis).
        - Otherwise → **0.0** (neutral step, neither helpful nor harmful).

        Args:
            property_used: Full URI string of the property followed
                (e.g. ``"http://example.org/kinship#isBiologicalParentOf"``).
            will_clause: Dict with at least ``requires`` (list[str]) and
                ``excludes`` (list[str]) keys.  Property names may be
                given as local names (e.g. ``"isMarriedTo"``) or full URIs.

        Returns:
            A float reward score for this single hop.
        """
        # Normalise will lists to full URI strings for comparison
        requires = _normalise_property_list(will_clause.get("requires", []))
        excludes = _normalise_property_list(will_clause.get("excludes", []))

        prop_uri = _ensure_full_uri(property_used)

        # --- Exclusion check (worst penalty) ---
        if prop_uri in excludes:
            return -3.0

        # --- Symmetry + lineage check ---
        requires_lineage = bool(requires & _BIOLOGICAL_LINEAGE_PROPERTIES)
        if prop_uri in self.symmetric_properties and requires_lineage:
            return -2.0

        # --- Positive match ---
        if prop_uri in requires:
            return 1.0

        # Also give credit for the inverse of a required property
        inverse = self.inverse_pairs.get(prop_uri)
        if inverse and inverse in requires:
            return 1.0

        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 2.  XML tag parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

# Pre-compiled patterns — kept at module level for speed.

_RE_REASONING = re.compile(
    r"<reasoning\s*>.*?</reasoning\s*>", re.DOTALL
)
_RE_ANSWER = re.compile(
    r"<answer\s*>(.*?)</answer\s*>", re.DOTALL
)
_RE_ACTION_SELF_CLOSING = re.compile(
    r"""<action\s+(?P<attrs>[^>]*?)\s*/\s*>""", re.DOTALL
)
_RE_ACTION_PAIR = re.compile(
    r"""<action\s+(?P<attrs>[^>]*?)\s*>.*?</action\s*>""", re.DOTALL
)
_RE_ATTR = re.compile(
    r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)


def has_valid_xml_tags(text: str) -> bool:
    """Check for valid ``<reasoning>`` and (``<action>`` or ``<answer>``) tags.

    A completion is considered structurally valid when it contains:

    - At least one ``<reasoning>…</reasoning>`` block, **and**
    - At least one ``<action …/>`` *or* one ``<answer>…</answer>`` block.

    Args:
        text: Raw model output string.

    Returns:
        ``True`` if the required structural tags are present.
    """
    has_reasoning = bool(_RE_REASONING.search(text))
    has_answer = bool(_RE_ANSWER.search(text))
    has_action = bool(
        _RE_ACTION_SELF_CLOSING.search(text) or _RE_ACTION_PAIR.search(text)
    )
    return has_reasoning and (has_answer or has_action)


def extract_answer_tag(text: str) -> str | None:
    """Extract content between ``<answer>`` and ``</answer>``.

    Returns the **first** match, stripped of leading/trailing whitespace.
    Returns ``None`` when no ``<answer>`` tag is found.

    Args:
        text: Raw model output string.

    Returns:
        The extracted answer string, or ``None``.
    """
    m = _RE_ANSWER.search(text)
    return m.group(1).strip() if m else None


def extract_action_tags(text: str) -> list[dict[str, str]]:
    """Extract all ``<action>`` tags and parse their attributes.

    Supports both self-closing (``<action … />``) and paired
    (``<action …>…</action>``) forms.  Attribute values may use
    single or double quotes.

    Expected format::

        <action tool="follow_entity_link" entity="…" property="…"/>

    Args:
        text: Raw model output string.

    Returns:
        List of dicts with keys like ``'tool'``, ``'entity'``,
        ``'property'`` — whatever attributes are present on each tag.
    """
    actions: list[dict[str, str]] = []

    # Gather attribute strings from both tag forms
    for pattern in (_RE_ACTION_SELF_CLOSING, _RE_ACTION_PAIR):
        for match in pattern.finditer(text):
            attrs_str = match.group("attrs")
            attrs: dict[str, str] = {}
            for attr_match in _RE_ATTR.finditer(attrs_str):
                key = attr_match.group(1)
                # group(2) is double-quoted value, group(3) is single-quoted
                value = (
                    attr_match.group(2)
                    if attr_match.group(2) is not None
                    else attr_match.group(3)
                )
                attrs[key] = value
            actions.append(attrs)

    return actions


def count_action_tags(text: str) -> int:
    """Count the number of ``<action>`` tags in *text*.

    Args:
        text: Raw model output string.

    Returns:
        Non-negative integer count.
    """
    return len(extract_action_tags(text))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Main reward function (GRPO-compatible)
# ═══════════════════════════════════════════════════════════════════════════


def kinship_ontology_reward(
    completions: list[str],
    prompts: list[str],
    **kwargs: Any,
) -> list[float]:
    """Ontology-grounded reward function for ``GRPOTrainer``.

    This function replaces the typical *LLM-as-Judge* evaluator with
    deterministic, axiom-based scoring.  Each completion is decomposed into
    its XML structure and every ``<action>`` hop is validated against the
    OWL axioms and the scenario's will clause.

    Reward components
    ~~~~~~~~~~~~~~~~~
    1. **Format** (+1.0) — valid XML tag structure.
    2. **Ontological path** — sum of per-hop scores from
       :meth:`OntologyRewardValidator.validate_traversal_step`.
    3. **Correct answer** (+5.0 correct, −1.0 wrong).
    4. **Efficiency** (+1.0 if hops ≤ optimal, +0.5 if ≤ optimal + 2).
    5. **Dynamic Hop Control** (+0.5 if the agent terminated with
       ``<answer>`` within optimal + 1 hops).

    All final scores are floored at **0.0**.

    Args:
        completions: List of model-generated output strings.
        prompts: Corresponding input prompts (unused by scoring but
            required by the ``GRPOTrainer`` callback signature).
        **kwargs:
            validator (:class:`OntologyRewardValidator`):
                Pre-initialised validator instance.
            gold_answers (list[str]):
                Expected correct answers (one per completion).
            will_clauses (list[dict]):
                Will clause dicts with ``requires`` / ``excludes`` keys.
            optimal_hops (list[int]):
                Known-optimal hop counts for each scenario.

    Returns:
        List of float rewards, one per completion, each ≥ 0.0.

    Raises:
        KeyError: If a required kwarg is missing.
    """
    validator: OntologyRewardValidator = kwargs["validator"]
    gold_answers: list[str] = kwargs["gold_answers"]
    will_clauses: list[dict[str, Any]] = kwargs["will_clauses"]
    optimal_hops: list[int] = kwargs["optimal_hops"]

    rewards: list[float] = []

    for idx, completion in enumerate(completions):
        score = 0.0

        # -- 1. Format reward ------------------------------------------------
        if has_valid_xml_tags(completion):
            score += 1.0

        # -- 2. Ontological path reward --------------------------------------
        will_clause = will_clauses[idx]
        actions = extract_action_tags(completion)
        n_required_props = 0

        for action in actions:
            prop = action.get("property", "")
            if prop:
                step_score = validator.validate_traversal_step(prop, will_clause)
                score += step_score
                # Track whether the model used a required property
                prop_uri = _ensure_full_uri(prop)
                requires = _normalise_property_list(will_clause.get("requires", []))
                if prop_uri in requires:
                    n_required_props += 1

        # -- 3. Correct answer reward (GATED by graph usage) -----------------
        # The answer bonus is tiered: structured action > ontology mention > blind
        # This prevents the model from gaming the reward via text shortcuts.
        predicted = extract_answer_tag(completion)
        gold = gold_answers[idx]

        if predicted is not None:
            correct = _answers_match(predicted, gold)
            # Check for ontology property mentions in free text
            requires_names = will_clause.get("requires", [])
            n_mentioned = sum(1 for p in requires_names if p in completion)
            n_ontology_aware = max(n_required_props, n_mentioned)

            if correct:
                if n_required_props >= 1:
                    score += 5.0   # Tier 1: structured KG action + correct
                elif n_ontology_aware >= 1:
                    score += 1.0   # Tier 2: ontology mention + correct
                else:
                    score += 0.5   # Tier 3: blind correct (minimal credit)
            else:
                score -= 1.0

        # -- 4. Efficiency reward --------------------------------------------
        num_hops = len(actions)
        optimal = optimal_hops[idx]

        if num_hops > 0:
            if num_hops <= optimal:
                score += 1.0
            elif num_hops <= optimal + 2:
                score += 0.5

        # -- 5. Dynamic Hop Control reward -----------------------------------
        if predicted is not None and num_hops <= optimal + 1:
            score += 0.5

        # -- Floor at 0.0 ----------------------------------------------------
        rewards.append(max(0.0, score))

    return rewards


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════


def _ensure_full_uri(prop: str) -> str:
    """Expand a local name to the full kinship namespace URI if needed."""
    if prop.startswith("http://") or prop.startswith("https://"):
        return prop
    return str(EX[prop])


def _normalise_property_list(props: list[str]) -> set[str]:
    """Convert a list of (possibly local) property names to full URI set."""
    return {_ensure_full_uri(p) for p in props}


def _answers_match(predicted: str, gold: str) -> bool:
    """Case-insensitive, whitespace-normalised answer comparison.

    Also handles the common case where the model wraps the answer in
    quotes or adds a trailing period.
    """
    def _clean(s: str) -> str:
        s = s.strip().strip('"').strip("'").rstrip(".")
        return " ".join(s.lower().split())

    return _clean(predicted) == _clean(gold)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import textwrap

    # Resolve ontology path relative to repo root
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _ONTOLOGY_PATH = str(_REPO_ROOT / "data" / "kinship.ttl")

    if not Path(_ONTOLOGY_PATH).exists():
        print(f"✗ Ontology not found at {_ONTOLOGY_PATH}", file=sys.stderr)
        sys.exit(1)

    # -- Initialise validator ------------------------------------------------
    validator = OntologyRewardValidator(_ONTOLOGY_PATH)

    print("=== OntologyRewardValidator pre-computed caches ===")
    print(f"  Symmetric properties : {validator.symmetric_properties}")
    print(f"  Inverse pairs        : {validator.inverse_pairs}")
    print(f"  Domains ({len(validator.property_domains)}): "
          f"{list(validator.property_domains.items())[:3]} …")
    print(f"  Ranges  ({len(validator.property_ranges)}): "
          f"{list(validator.property_ranges.items())[:3]} …")
    print()

    # -- Test will clause (eldest_grandchild) --------------------------------
    will_clause: dict[str, Any] = {
        "requires": ["isBiologicalParentOf"],
        "excludes": ["isMarriedTo"],
    }

    # --- Known-good completion: correct traversal, correct answer -----------
    good_completion = textwrap.dedent("""\
        <reasoning>
        I need to find the eldest living biological grandchild.
        Let me start by finding the patriarch's biological children,
        then look at their children.
        </reasoning>
        <action tool="follow_entity_link" entity="Person1" property="isBiologicalParentOf"/>
        <action tool="follow_entity_link" entity="Person2" property="isBiologicalParentOf"/>
        <reasoning>
        Person5 is the eldest living grandchild at age 28.
        </reasoning>
        <answer>Leonardo Rossi</answer>
    """)

    # --- Known-bad completion: follows marriage link, wrong answer -----------
    bad_completion = textwrap.dedent("""\
        <reasoning>
        The spouse managed everything, so they must be the heir.
        </reasoning>
        <action tool="follow_entity_link" entity="Person1" property="isMarriedTo"/>
        <answer>Giulia Bianchi</answer>
    """)

    # --- Malformed completion: missing tags ---------------------------------
    ugly_completion = "I think the answer is Leonardo Rossi."

    # -- Run reward function -------------------------------------------------
    completions = [good_completion, bad_completion, ugly_completion]
    prompts = ["prompt"] * 3
    gold_answers = ["Leonardo Rossi", "Leonardo Rossi", "Leonardo Rossi"]
    will_clauses = [will_clause] * 3
    optimal_hops_list = [2, 2, 2]

    rewards = kinship_ontology_reward(
        completions=completions,
        prompts=prompts,
        validator=validator,
        gold_answers=gold_answers,
        will_clauses=will_clauses,
        optimal_hops=optimal_hops_list,
    )

    print("=== Reward function self-test ===")
    labels = ["GOOD (correct path + answer)", "BAD (excluded prop + wrong)",
              "UGLY (no XML tags at all)"]
    for label, reward, completion in zip(labels, rewards, completions):
        print(f"\n--- {label} ---")
        print(f"  Actions: {count_action_tags(completion)}")
        print(f"  Answer : {extract_answer_tag(completion)!r}")
        print(f"  Reward : {reward:.1f}")

    # -- Assertions ----------------------------------------------------------
    assert rewards[0] > rewards[1], (
        f"Good completion ({rewards[0]}) should outscore bad ({rewards[1]})"
    )
    assert rewards[1] >= 0.0, "All rewards must be ≥ 0.0 (floor)"
    assert rewards[2] == 0.0, (
        f"Malformed completion should score 0.0, got {rewards[2]}"
    )

    print("\n✓ All assertions passed.")
