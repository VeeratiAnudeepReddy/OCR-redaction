"""
PII module — Presidio-based detection and anonymisation of personally
identifiable information.

Privacy guarantees:
  * Raw text and PII values are NEVER logged.
  * Only entity types and confidence scores are surfaced externally.

Design:
  * AnalyzerEngine and AnonymizerEngine are initialised ONCE at import time
    (module-level singletons) so they are not recreated per request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import backend.config as config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons (initialised once)
# ---------------------------------------------------------------------------

def _build_analyzer() -> AnalyzerEngine:
    """
    Construct and return a configured AnalyzerEngine.

    Uses the spaCy NLP engine backed by the configured model.
    """
    logger.info("Initialising Presidio AnalyzerEngine with spaCy model '%s' …", config.SPACY_MODEL)
    try:
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": config.SPACY_MODEL}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        logger.info("Presidio AnalyzerEngine initialised successfully.")
        return analyzer
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise AnalyzerEngine: %s", type(exc).__name__)
        raise


def _build_anonymizer() -> AnonymizerEngine:
    """Construct and return an AnonymizerEngine."""
    logger.info("Initialising Presidio AnonymizerEngine …")
    engine = AnonymizerEngine()
    logger.info("Presidio AnonymizerEngine initialised successfully.")
    return engine


# Eagerly build engines at import time so the first request is fast.
_analyzer: AnalyzerEngine = _build_analyzer()
_anonymizer: AnonymizerEngine = _build_anonymizer()


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedEntity:
    """A detected PII entity (sanitised — no raw text)."""
    entity_type: str
    score: float
    start: int
    end: int


@dataclass
class AnalysisResult:
    """Holds the sanitised output from the full PII pipeline."""
    safe_text: str
    entities: List[DetectedEntity] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Operator map builder
# ---------------------------------------------------------------------------

def _build_operators() -> dict[str, OperatorConfig]:
    """
    Build a Presidio operator map that replaces each entity type with its
    configured placeholder.  Unknown entities fall back to [REDACTED].
    """
    operators: dict[str, OperatorConfig] = {}
    for entity_type, label in config.ANONYMIZATION_LABELS.items():
        operators[entity_type] = OperatorConfig(
            operator_name="replace",
            params={"new_value": label},
        )
    # "DEFAULT" catches any entity type not listed above
    operators["DEFAULT"] = OperatorConfig(
        operator_name="replace",
        params={"new_value": config.DEFAULT_REDACTION_LABEL},
    )
    return operators


_OPERATORS: dict[str, OperatorConfig] = _build_operators()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> list[RecognizerResult]:
    """
    Run Presidio Analyzer on *text* and return raw recogniser results.

    Args:
        text: The text to analyse (may contain PII).

    Returns:
        List of RecognizerResult objects above the configured threshold.

    Privacy note: *text* and the matched PII values are NEVER logged.
    """
    if not text or not text.strip():
        return []

    results: list[RecognizerResult] = _analyzer.analyze(
        text=text,
        language="en",
        entities=config.ENTITIES_TO_DETECT,
        score_threshold=config.PII_SCORE_THRESHOLD,
    )

    # Log only types and counts — never raw values.
    types_found = [r.entity_type for r in results]
    logger.info(
        "Presidio detected %d PII entit%s. Types: %s",
        len(results),
        "y" if len(results) == 1 else "ies",
        ", ".join(types_found) if types_found else "none",
    )
    return results


def anonymize_text(text: str, analyzer_results: list[RecognizerResult]) -> str:
    """
    Replace detected PII spans in *text* with their configured placeholders.

    Args:
        text: Original text (may contain PII).
        analyzer_results: Output of :func:`analyze_text`.

    Returns:
        Sanitised text with PII replaced by placeholder labels.

    Privacy note: *text* and PII values are NEVER logged.
    """
    if not analyzer_results:
        return text

    anonymized = _anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=_OPERATORS,
    )
    return anonymized.text


def process_text(text: str) -> AnalysisResult:
    """
    High-level convenience function: detect + anonymise in one call.

    Args:
        text: Raw OCR text (may contain PII).

    Returns:
        :class:`AnalysisResult` with sanitised text and entity metadata.
        Raw PII values are NOT included.
    """
    analyzer_results = analyze_text(text)

    safe_text = anonymize_text(text, analyzer_results)

    entities = [
        DetectedEntity(
            entity_type=r.entity_type,
            score=round(r.score, 4),
            start=r.start,
            end=r.end,
        )
        for r in analyzer_results
    ]

    return AnalysisResult(safe_text=safe_text, entities=entities)
