"""
Tests for backend/pii.py

Tests Presidio-based PII detection and anonymisation.
"""

from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.pii import analyze_text, anonymize_text, process_text, AnalysisResult

# Synthetic PII — these are fictional characters / numbers only
SYNTHETIC_NAME = "Peter Parker"
SYNTHETIC_EMAIL = "peter.parker@example.com"
SYNTHETIC_PHONE = "+91 9876543210"

SYNTHETIC_TEXT = f"""
Name: {SYNTHETIC_NAME}
Email: {SYNTHETIC_EMAIL}
Phone: {SYNTHETIC_PHONE}
Course: Computer Science
College: ABC University
Application ID: APP-2026-001
"""


class TestAnalyzeText:
    def test_returns_list(self):
        results = analyze_text(SYNTHETIC_TEXT)
        assert isinstance(results, list)

    def test_detects_entities(self):
        results = analyze_text(SYNTHETIC_TEXT)
        assert len(results) > 0, "Expected at least one PII entity to be detected"

    def test_detects_email(self):
        text = "Please email alice@example.com for details."
        results = analyze_text(text)
        types = [r.entity_type for r in results]
        assert "EMAIL_ADDRESS" in types, f"EMAIL_ADDRESS not detected. Found: {types}"

    def test_detects_person(self):
        text = "The report was filed by John Smith yesterday."
        results = analyze_text(text)
        types = [r.entity_type for r in results]
        assert "PERSON" in types, f"PERSON not detected. Found: {types}"

    def test_empty_text_returns_empty(self):
        assert analyze_text("") == []
        assert analyze_text("   ") == []

    def test_non_pii_text_has_no_entities(self):
        """Generic technical text should not trigger PII detections."""
        text = "HTTP/1.1 200 OK Content-Type: application/json"
        results = analyze_text(text)
        # IP_ADDRESS or URL might fire — filter for personal PII only
        personal = [r for r in results if r.entity_type in ("PERSON", "EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD")]
        assert personal == [], f"Unexpected personal PII detected: {personal}"


class TestAnonymizeText:
    def test_email_replaced(self):
        text = "Contact bob@example.com for info."
        results = analyze_text(text)
        safe = anonymize_text(text, results)
        assert "bob@example.com" not in safe
        assert "[EMAIL_ADDRESS]" in safe

    def test_person_replaced(self):
        text = "Alice Johnson submitted the application."
        results = analyze_text(text)
        safe = anonymize_text(text, results)
        # Presidio may or may not detect "Alice Johnson" depending on model confidence
        if any(r.entity_type == "PERSON" for r in results):
            assert "Alice Johnson" not in safe

    def test_empty_results_returns_original(self):
        text = "Course: Computer Science"
        safe = anonymize_text(text, [])
        assert safe == text

    def test_non_pii_text_preserved(self):
        text = "Application ID: APP-2026-001\nCourse: Computer Science"
        results = analyze_text(text)
        safe = anonymize_text(text, results)
        # These tokens should still be present
        assert "APP-2026-001" in safe or "Application" in safe


class TestProcessText:
    def test_returns_analysis_result(self):
        result = process_text(SYNTHETIC_TEXT)
        assert isinstance(result, AnalysisResult)

    def test_safe_text_is_string(self):
        result = process_text(SYNTHETIC_TEXT)
        assert isinstance(result.safe_text, str)
        assert len(result.safe_text) > 0

    def test_raw_pii_not_in_safe_text(self):
        """Core privacy assertion: raw PII must not appear in output."""
        result = process_text(SYNTHETIC_TEXT)
        safe = result.safe_text
        assert SYNTHETIC_EMAIL not in safe, f"Raw email found in safe_text!"
        # Name may or may not be detected by Presidio — only assert if detected
        name_detected = any(e.entity_type == "PERSON" for e in result.entities)
        if name_detected:
            assert SYNTHETIC_NAME not in safe, f"Raw name found in safe_text!"

    def test_placeholders_present_when_detected(self):
        """When email is detected, [EMAIL_ADDRESS] placeholder must appear."""
        result = process_text(SYNTHETIC_TEXT)
        email_detected = any(e.entity_type == "EMAIL_ADDRESS" for e in result.entities)
        if email_detected:
            assert "[EMAIL_ADDRESS]" in result.safe_text

    def test_entity_list_has_correct_structure(self):
        result = process_text(SYNTHETIC_TEXT)
        for entity in result.entities:
            assert hasattr(entity, "entity_type")
            assert hasattr(entity, "score")
            assert 0.0 <= entity.score <= 1.0

    def test_non_pii_preserved(self):
        """Non-PII content should NOT be redacted."""
        text = "Course: Computer Science\nCollege: ABC University\nID: APP-2026-001"
        result = process_text(text)
        # At least some of this non-PII text should survive
        assert "Computer Science" in result.safe_text or "APP-2026-001" in result.safe_text

    def test_empty_text(self):
        result = process_text("")
        assert result.safe_text == ""
        assert result.entities == []

    def test_only_email(self):
        result = process_text("Send results to charlie@test.org")
        email_entities = [e for e in result.entities if e.entity_type == "EMAIL_ADDRESS"]
        if email_entities:
            assert "charlie@test.org" not in result.safe_text
            assert "[EMAIL_ADDRESS]" in result.safe_text

    def test_only_phone(self):
        result = process_text("Call us at +1 555-867-5309")
        phone_entities = [e for e in result.entities if e.entity_type == "PHONE_NUMBER"]
        if phone_entities:
            assert "555-867-5309" not in result.safe_text

    def test_confidence_scores_above_threshold(self):
        """All returned entities should meet the configured threshold."""
        from backend.config import PII_SCORE_THRESHOLD
        result = process_text(SYNTHETIC_TEXT)
        for entity in result.entities:
            assert entity.score >= PII_SCORE_THRESHOLD, \
                f"Entity {entity.entity_type} has score {entity.score} below threshold"
