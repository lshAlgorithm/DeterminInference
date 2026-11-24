from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import b2_cpu_gate


def reference_fixture(session_count: int = 8) -> dict:
    rows = [
        {
            "session": f"{index:03}.json",
            "request": 0,
            "output_ids": [index + 1, index + 2],
        }
        for index in range(session_count)
    ]
    return {"complete": True, "requests_exact": len(rows), "rows": rows}


def provenance_fixture() -> dict:
    reference = reference_fixture()
    artifact = {"path": "evidence.json", "sha256": "a" * 64}
    return {
        "schema_version": 1,
        "reference_sha256": hashlib.sha256(
            json.dumps(reference, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "frozen_plan": artifact,
        "runner_receipts": {arm: artifact for arm in b2_cpu_gate.ARMS},
        "request_logs": {arm: [artifact] for arm in b2_cpu_gate.ARMS},
    }


def evaluate_fixture_gate(reference: dict, bundle: dict) -> dict:
    return b2_cpu_gate.evaluate_gate(reference, bundle, fixture_only=True)


def event_bundle(selected_sessions: list[str]) -> dict:
    events = []
    warmup_events = []
    for arm in b2_cpu_gate.ARMS:
        for index, session in enumerate(selected_sessions):
            source_index = int(session[:3])
            events.append(
                {
                    "event": "request.finished",
                    "arm": arm,
                    "request_id": f"{arm}-{index}",
                    "source_file": session,
                    "source_row": 0,
                    "phase": "profiling",
                    "logical_root_id": f"root-{index}",
                    "tree_occurrence_id": f"tree-{index}",
                    "turn": 0,
                    "reset_ordinal": 0,
                    "cache_salt_class": f"salt-{index}",
                    "output_ids": [source_index + 1, source_index + 2],
                    "output_length": 2,
                    "cache_hit_tokens": 7,
                    "new_prefill_tokens": 9,
                    "total_retractions": 0,
                }
            )
    for arm in b2_cpu_gate.ARMS:
        for index, session in enumerate(selected_sessions[:2]):
            source_index = int(session[:3])
            warmup_events.append(
                {
                    "event": "request.finished",
                    "arm": arm,
                    "request_id": f"{arm}-warmup-{index}",
                    "source_file": session,
                    "source_row": 0,
                    "phase": "warmup",
                    "logical_root_id": f"warmup-root-{index}",
                    "tree_occurrence_id": f"warmup-tree-{index}",
                    "turn": index,
                    "reset_ordinal": 0,
                    "cache_salt_class": f"warmup-salt-{index}",
                    "output_ids": [source_index + 1, source_index + 2],
                    "output_length": 2,
                    "cache_hit_tokens": 7,
                    "new_prefill_tokens": 9,
                    "total_retractions": 0,
                }
            )
    expected_occurrences = [
        {
            **{key: event[key] for key in b2_cpu_gate.OCCURRENCE_FIELDS},
            "count": 1,
        }
        for event in events
        if event["arm"] == "capture"
    ]
    expected_warmup_occurrences = [
        {
            **{key: event[key] for key in b2_cpu_gate.OCCURRENCE_FIELDS},
            "count": 1,
        }
        for event in warmup_events
        if event["arm"] == "capture"
    ]
    per_arm = {
        arm: {
            "sent": 6,
            "completed": 6,
            "errors": 0,
            "skips": 0,
            "cancellations": 0,
            "pending": 0,
            "phase_drains": {
                "warmup": {
                    "sent": 2,
                    "completed": 2,
                    "errors": 0,
                    "cancellations": 0,
                    "pending": 0,
                },
                "profiling": {
                    "sent": 4,
                    "completed": 4,
                    "errors": 0,
                    "cancellations": 0,
                    "pending": 0,
                },
            },
        }
        for arm in b2_cpu_gate.ARMS
    }
    return {
        "schema_version": 2,
        "loaded_sessions": selected_sessions,
        "profiling_root_admissions": [
            {
                "source_file": selected_sessions[index],
                "logical_root_id": f"root-{index}",
            }
            for index in range(4)
        ],
        "observed_profiling_root_admissions": {
            arm: [
                {
                    "source_file": selected_sessions[index],
                    "logical_root_id": f"root-{index}",
                }
                for index in range(4)
            ]
            for arm in b2_cpu_gate.ARMS
        },
        "events": events,
        "warmup_events": warmup_events,
        "provenance": provenance_fixture(),
        "expected_occurrences": expected_occurrences,
        "expected_warmup_occurrences": expected_warmup_occurrences,
        "expected_warmup_request_ids": {
            arm: [f"{arm}-warmup-{index}" for index in range(2)]
            for arm in b2_cpu_gate.ARMS
        },
        "drain": {
            "sent": 18,
            "completed": 18,
            "errors": 0,
            "skips": 0,
            "cancellations": 0,
            "pending": 0,
            "by_arm": per_arm,
        },
    }


class TestB2CpuGate(unittest.TestCase):
    def test_sampling_is_seeded_from_sorted_session_identities(self) -> None:
        first = b2_cpu_gate.select_cohort(reference_fixture())
        second = b2_cpu_gate.select_cohort(reference_fixture())
        self.assertEqual(
            first["eligible_sessions"], [f"{index:03}.json" for index in range(8)]
        )
        self.assertEqual(
            first["selected_sessions"], ["001.json", "000.json", "005.json", "002.json"]
        )
        self.assertEqual(first, second)

    def test_absent_instrumentation_is_deferred(self) -> None:
        result = b2_cpu_gate.evaluate_gate(reference_fixture())
        self.assertEqual(result["status"], "DEFERRED")
        self.assertEqual(result["eligible_session_count"], 8)
        self.assertIn(
            "request.finished instrumentation is absent", result["deferred_reason"]
        )

    def test_complete_symmetric_bundle_passes_and_separates_load_from_roots(
        self,
    ) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        result = evaluate_fixture_gate(
            reference_fixture(), event_bundle(cohort["selected_sessions"])
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["loaded_sessions"]), 4)
        self.assertEqual(len(result["profiling_root_admissions"]), 4)
        self.assertEqual(result["event_counts"], {arm: 4 for arm in b2_cpu_gate.ARMS})
        self.assertEqual(result["evidence_scope"], "fixture-only")

    def test_empty_warmup_requires_explicit_zero_phase_receipts(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        bundle["warmup_events"] = []
        bundle["expected_warmup_occurrences"] = []
        bundle["expected_warmup_request_ids"] = {arm: [] for arm in b2_cpu_gate.ARMS}
        for arm in b2_cpu_gate.ARMS:
            bundle["drain"]["by_arm"][arm]["phase_drains"]["warmup"] = {
                "sent": 0,
                "completed": 0,
                "errors": 0,
                "cancellations": 0,
                "pending": 0,
            }
            bundle["drain"]["by_arm"][arm]["sent"] = 4
            bundle["drain"]["by_arm"][arm]["completed"] = 4
        bundle["drain"]["sent"] = 12
        bundle["drain"]["completed"] = 12
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "PASS")

        bundle["drain"]["by_arm"]["capture"]["phase_drains"].pop("warmup")
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("explicit warmup" in error for error in result["errors"]))

    def test_profiling_root_admissions_require_exactly_four(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        valid_bundle = event_bundle(cohort["selected_sessions"])
        for count in (3, 5):
            with self.subTest(count=count):
                bundle = json.loads(json.dumps(valid_bundle))
                bundle["profiling_root_admissions"] = bundle[
                    "profiling_root_admissions"
                ][:count]
                if count == 5:
                    bundle["profiling_root_admissions"].append(
                        {
                            "source_file": cohort["selected_sessions"][0],
                            "logical_root_id": "root-extra",
                        }
                    )
                result = evaluate_fixture_gate(reference_fixture(), bundle)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(
                    any(
                        "must contain exactly four roots" in error
                        for error in result["errors"]
                    )
                )

    def test_missing_independent_artifact_provenance_is_deferred(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        del bundle["provenance"]
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "DEFERRED")
        self.assertIn(
            "independently hashed runtime provenance", result["deferred_reason"]
        )

    def test_missing_arm_is_deferred_not_pass(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        bundle["events"] = [
            event for event in bundle["events"] if event["arm"] != "replay"
        ]
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "DEFERRED")
        self.assertIn("replay", result["deferred_reason"])

    def test_common_mode_wrong_output_still_fails_against_reference(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        for event in bundle["events"]:
            event["output_ids"] = [99, 100]
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("immutable reference" in error for error in result["errors"])
        )

    def test_common_mode_missing_occurrence_fails_against_frozen_ledger(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        bundle["events"] = [
            event for event in bundle["events"] if event["logical_root_id"] != "root-0"
        ]
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("frozen ledger" in error for error in result["errors"]))

    def test_each_arm_completion_receipt_must_cover_its_events(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        receipt = bundle["drain"]["by_arm"]["capture"]
        receipt["sent"] += 1
        receipt["completed"] += 1
        bundle["drain"]["sent"] += 1
        bundle["drain"]["completed"] += 1
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("completion receipt" in error for error in result["errors"])
        )

    def test_explicit_warmups_must_match_the_frozen_ledger_and_count_toward_drain(
        self,
    ) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        bundle["warmup_events"] = [
            event
            for event in bundle["warmup_events"]
            if not (
                event["arm"] == "capture"
                and event["logical_root_id"] == "warmup-root-0"
            )
        ]
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("warmup identities differ" in error for error in result["errors"])
        )

        bundle = event_bundle(cohort["selected_sessions"])
        bundle["warmup_events"][0]["phase"] = "profiling"
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("phase warmup" in error for error in result["errors"]))

        bundle = event_bundle(cohort["selected_sessions"])
        bundle["drain"]["by_arm"]["capture"]["completed"] -= 1
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("warmup counts" in error for error in result["errors"]))

        bundle = event_bundle(cohort["selected_sessions"])
        next(event for event in bundle["warmup_events"] if event["arm"] == "generated")[
            "cache_hit_tokens"
        ] += 1
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "warmup outputs or cache metrics differ" in error
                for error in result["errors"]
            )
        )

    def test_missing_occurrence_or_arm_receipts_are_deferred(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        del bundle["expected_occurrences"]
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "DEFERRED")
        self.assertIn("expected occurrence ledger", result["deferred_reason"])

        bundle = event_bundle(cohort["selected_sessions"])
        del bundle["drain"]["by_arm"]
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "DEFERRED")
        self.assertIn("per-arm completion receipts", result["deferred_reason"])

    def test_output_and_cache_metrics_must_match_exactly(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        changes = (
            ("output_ids", [11, 13]),
            ("output_length", 3),
            ("cache_hit_tokens", 8),
            ("new_prefill_tokens", 10),
        )
        for field, value in changes:
            with self.subTest(field=field):
                bundle = event_bundle(cohort["selected_sessions"])
                next(event for event in bundle["events"] if event["arm"] == "replay")[
                    field
                ] = value
                result = evaluate_fixture_gate(reference_fixture(), bundle)
                self.assertEqual(result["status"], "FAIL")

    def test_event_schema_must_be_symmetric(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        next(event for event in bundle["events"] if event["arm"] == "replay")[
            "extra"
        ] = "unexpected"
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "field set" in error or "schemas are asymmetric" in error
                for error in result["errors"]
            )
        )

    def test_event_source_row_must_be_in_selected_corpus(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        next(event for event in bundle["events"] if event["arm"] == "generated")[
            "source_row"
        ] = 99
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("source row outside" in error for error in result["errors"])
        )

    def test_malformed_occurrence_values_fail_without_crashing(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        next(event for event in bundle["events"] if event["arm"] == "generated")[
            "logical_root_id"
        ] = []
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("invalid logical_root_id" in error for error in result["errors"])
        )

    def test_duplicate_occurrence_multiplicity_must_match(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        replay = next(event for event in bundle["events"] if event["arm"] == "replay")
        bundle["events"].append(dict(replay, request_id="replay-duplicate"))
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("multiplicities differ" in error for error in result["errors"])
        )

    def test_drain_must_complete_without_failures(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        bundle["drain"]["pending"] = 1
        result = evaluate_fixture_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("pending must be zero" in error for error in result["errors"])
        )

    def test_unverified_prebuilt_bundle_is_fixture_only(self) -> None:
        cohort = b2_cpu_gate.select_cohort(reference_fixture())
        bundle = event_bundle(cohort["selected_sessions"])
        result = b2_cpu_gate.evaluate_gate(reference_fixture(), bundle)
        self.assertEqual(result["status"], "DEFERRED")
        self.assertFalse(result["runtime_artifacts_verified"])
        self.assertIn("fixture-only", result["deferred_reason"])

    def test_cli_rejects_prebuilt_event_bundle_bypass(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(b2_cpu_gate.__file__)),
                "--reference",
                "unused-reference.json",
                "--events",
                "forged-events.json",
                "--output",
                "unused-output.json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --events", result.stderr)


if __name__ == "__main__":
    unittest.main()
