"""
PII module — Presidio-based detection and anonymisation of personally
identifiable information.

Privacy guarantees:
  * Raw text and PII values are NEVER logged.
  * Only entity types and confidence scores are surfaced externally.

Design:
  * AnalyzerEngine and AnonymizerEngine are initialised ONCE at import time
    (module-level singletons) so they are not recreated per request.
  * Custom PatternRecognizers for USERNAME and SOCIAL_HANDLE are registered
    at startup to catch label-prefixed fields and bare @handles that Presidio's
    built-in recognizers do not cover.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import re
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import backend.config as config
from backend.ocr import OCRWord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Context-Aware Recognizers
# ---------------------------------------------------------------------------

class CredentialRecognizer(EntityRecognizer):
    """
    Context-aware recogniser for USERNAME and PASSWORD credentials.

    Uses label proximity (same line or next line) and spatial layout to detect
    credentials associated with form labels while avoiding false positives on
    help text, links, or standalone headers.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entities=["USERNAME", "PASSWORD"],
            name="credential_context_recognizer",
        )
        self.username_labels = [
            "username", "user name", "login id", "user id", "email address",
            "account name", "email", "user", "login", "account",
        ]
        self.password_labels = [
            "password", "passcode", "passward", "pasword", "possword", "pwd", "secret",
        ]
        self.exclusion_keywords = [
            "forgot", "reset", "remember", "change", "don't have", "dont have",
            "register", "sign up", "credentials", "copyright", "rights reserved",
            "click here", "interviews", "enter your", "confirm",
        ]

    def _is_excluded(self, text: str) -> bool:
        lower = text.lower()
        return any(ex in lower for ex in self.exclusion_keywords)

    def _is_label_match(self, text: str, label_list: list[str]) -> bool:
        lower = text.lower()
        for lbl in label_list:
            if re.search(r"\b" + re.escape(lbl) + r"\b", lower):
                return True
        return False

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list[RecognizerResult]:
        if not text or not text.strip():
            return []

        results: list[RecognizerResult] = []
        target_entities = set(entities) if entities else {"USERNAME", "PASSWORD"}
        want_username = "USERNAME" in target_entities
        want_password = "PASSWORD" in target_entities

        if not (want_username or want_password):
            return []

        lines = text.splitlines()
        line_offsets: list[tuple[str, int, int]] = []
        offset = 0
        for l in lines:
            line_offsets.append((l, offset, offset + len(l)))
            offset += len(l) + 1  # newline

        all_labels = self.username_labels + self.password_labels

        for idx, (line_str, l_start, l_end) in enumerate(line_offsets):
            if self._is_excluded(line_str):
                continue

            u_label = want_username and self._is_label_match(line_str, self.username_labels)
            p_label = want_password and self._is_label_match(line_str, self.password_labels)

            if not (u_label or p_label):
                continue

            target_type = "USERNAME" if u_label else "PASSWORD"
            active_labels = self.username_labels if u_label else self.password_labels

            # 1. Inline check (same line value)
            inline_found = False
            for lbl in active_labels:
                pattern = r"\b" + re.escape(lbl) + r"\b\s*[:\-=?*]*\s*(.+)"
                m = re.search(pattern, line_str, re.IGNORECASE)
                if m:
                    candidate = m.group(1).strip()
                    candidate_clean = candidate.strip("?:;=|*")
                    if (
                        sum(c.isalnum() for c in candidate_clean) >= 2
                        and not self._is_excluded(candidate)
                        and not self._is_label_match(candidate, all_labels)
                    ):
                        rel_start = line_str.find(candidate_clean, m.start(1))
                        v_start = l_start + rel_start
                        v_end = v_start + len(candidate_clean)
                        results.append(
                            RecognizerResult(
                                entity_type=target_type,
                                start=v_start,
                                end=v_end,
                                score=0.85,
                            )
                        )
                        inline_found = True
                        break

            if inline_found:
                continue

            # 2. Stacked check (next line value)
            if idx + 1 < len(line_offsets):
                next_line_str, n_start, n_end = line_offsets[idx + 1]
                if (
                    next_line_str.strip()
                    and not self._is_excluded(next_line_str)
                    and not self._is_label_match(next_line_str, all_labels)
                ):
                    words = next_line_str.strip().split()
                    if words:
                        first_word = words[0].rstrip("|")
                        if sum(c.isalnum() for c in first_word) >= 2:
                            w_rel_start = next_line_str.find(first_word)
                            w_start = n_start + w_rel_start
                            w_end = w_start + len(first_word)
                            results.append(
                                RecognizerResult(
                                    entity_type=target_type,
                                    start=w_start,
                                    end=w_end,
                                    score=0.85,
                                )
                            )

        return results


def _build_username_recognizer() -> PatternRecognizer:
    pattern = Pattern(
        name="username_label_pattern",
        regex=(
            r"(?i)"
            r"(?:username|user\s*id|user|handle|login|screen\s*name|account\s*name)"
            r"\s*[:\-]\s*"
            r"([A-Za-z0-9_.@\-]{3,32})"
        ),
        score=0.85,
    )
    return PatternRecognizer(
        supported_entity="USERNAME",
        patterns=[pattern],
        context=["username", "user", "handle", "login", "account"],
    )


def _build_handle_recognizer() -> PatternRecognizer:
    pattern = Pattern(
        name="social_handle_pattern",
        regex=r"@[A-Za-z0-9_]{3,20}\b",
        score=0.7,
    )
    return PatternRecognizer(
        supported_entity="SOCIAL_HANDLE",
        patterns=[pattern],
    )


# ---------------------------------------------------------------------------
# Module-level singletons (initialised once)
# ---------------------------------------------------------------------------

def _build_analyzer() -> AnalyzerEngine:
    """
    Construct and return a configured AnalyzerEngine.
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

        # Register custom recognizers
        analyzer.registry.add_recognizer(CredentialRecognizer())
        analyzer.registry.add_recognizer(_build_handle_recognizer())

        logger.info(
            "Presidio AnalyzerEngine initialised successfully. "
            "Custom recognizers registered: CredentialRecognizer, USERNAME, SOCIAL_HANDLE."
        )
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
    """A detected PII entity."""
    entity_type: str
    score: float
    start: int
    end: int
    value: Optional[str] = None


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

def analyze_text(text: str, ocr_words: Optional[List[OCRWord]] = None) -> list[RecognizerResult]:
    """
    Run Presidio Analyzer on *text* and return raw recogniser results.

    Args:
        text: The text to analyse (may contain PII).
        ocr_words: Optional structured word-level bounding box data from OCR.

    Returns:
        List of RecognizerResult objects above the configured threshold.

    Privacy note: *text* and the matched PII values are NEVER logged.
    Debug logging (when PII_DEBUG_THRESHOLD >= 0) logs only entity_type
    and score — never the matched text or surrounding context.
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

    # Optional debug pass: log entity_type + score for candidates above the
    # debug threshold but below the production threshold, so developers can
    # tune sensitivity without changing production behaviour.
    if config.PII_DEBUG_THRESHOLD >= 0.0:
        debug_candidates: list[RecognizerResult] = _analyzer.analyze(
            text=text,
            language="en",
            entities=config.ENTITIES_TO_DETECT,
            score_threshold=config.PII_DEBUG_THRESHOLD,
        )
        sub_threshold = [
            r for r in debug_candidates
            if r.score < config.PII_SCORE_THRESHOLD
        ]
        if sub_threshold:
            debug_info = [
                f"{r.entity_type}={r.score:.3f}" for r in sub_threshold
            ]
            logger.debug(
                "[PII_DEBUG] %d candidate(s) above debug threshold (%.2f) "
                "but below production threshold (%.2f): %s",
                len(sub_threshold),
                config.PII_DEBUG_THRESHOLD,
                config.PII_SCORE_THRESHOLD,
                ", ".join(debug_info),
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


def process_text(
    text: str,
    ocr_words: Optional[List[OCRWord]] = None,
    debug_pii_report: Optional[bool] = None,
) -> AnalysisResult:
    """
    High-level convenience function: detect + anonymise in one call.

    Args:
        text: Raw OCR text (may contain PII).
        ocr_words: Optional word bounding boxes from OCR.
        debug_pii_report: Whether debug report mode is enabled to include raw PII values.

    Returns:
        :class:`AnalysisResult` with sanitised text and entity metadata.
        Raw PII values are included ONLY when debug_pii_report is True.
    """
    if debug_pii_report is None:
        debug_pii_report = config.DEBUG_PII_REPORT

    analyzer_results = analyze_text(text, ocr_words=ocr_words)

    safe_text = anonymize_text(text, analyzer_results)

    entities = [
        DetectedEntity(
            entity_type=r.entity_type,
            score=round(r.score, 4),
            start=r.start,
            end=r.end,
            value=text[r.start:r.end] if debug_pii_report else None,
        )
        for r in analyzer_results
    ]

    return AnalysisResult(safe_text=safe_text, entities=entities)



