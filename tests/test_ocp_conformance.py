"""OCP conformance: v0.2 schema and referential rules, v0.1 still passing.

Every test mutates a copy of a shipped example and pins the exact finding code
the checker must (or must not) produce, so a checker that accepts everything
or rejects everything fails here.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

from loopmath.ingest.ocp import from_ocp

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "spec"
EXAMPLES = SPEC / "examples"

_spec = importlib.util.spec_from_file_location("ocp_conformance", SPEC / "ocp_conformance.py")
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)

EXAMPLE_FILES = [
    "minimal.ocp.json",
    "review-loop-v01.ocp.json",
    "swarm-v02.ocp.json",
]


def codes(findings, level=None):
    return sorted(f.code for f in findings if level is None or f.level == level)


def errors(findings):
    return codes(findings, "error")


def warnings(findings):
    return codes(findings, "warning")


@pytest.fixture
def swarm():
    return json.loads((EXAMPLES / "swarm-v02.ocp.json").read_text())


@pytest.fixture
def minimal():
    return json.loads((EXAMPLES / "minimal.ocp.json").read_text())


def attempt(doc, aid):
    return next(a for a in doc["attempts"] if a["id"] == aid)


def edge(doc, frm, to, kind):
    return next(e for e in doc["edges"] if e["from"] == frm and e["to"] == to and e["kind"] == kind)


# --- shipped examples and version routing -----------------------------------

@pytest.mark.parametrize("name", EXAMPLE_FILES)
def test_shipped_examples_have_no_errors_and_no_warnings(name):
    findings = conf.validate_file(EXAMPLES / name)
    assert errors(findings) == [], findings
    assert warnings(findings) == [], findings


def test_schema_files_declare_their_version():
    v01 = json.loads((SPEC / "ocp-v0.schema.json").read_text())
    v02 = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
    assert v01["properties"]["ocp"]["const"] == "0.1"
    assert v02["properties"]["ocp"]["const"] == "0.2"
    # The checker reads the package-data copies (spec 01 section 3); spec/ keeps identical copies.
    for version, name in (("0.1", "ocp-v0.schema.json"), ("0.2", "ocp-v0.2.schema.json")):
        assert conf.schema_path_for({"ocp": version}).read_bytes() == (SPEC / name).read_bytes()


@pytest.mark.parametrize(
    ("retired_key", "expected_prefix"),
    [
        ("da" "gr.example", "dev.dagr."),
        ("oc" "p.example", "io.orchestrationcontextprotocol."),
    ],
)
def test_retired_extension_prefix_is_rejected_by_named_rule(
    swarm, retired_key, expected_prefix
):
    swarm["nodes"][0]["ext"] = {retired_key: {"kept": True}}
    found = conf.validate_doc(swarm)
    matching = [finding for finding in found if finding.code == "E181"]
    assert len(matching) == 1, found
    assert matching[0].path.endswith(f"['ext'][{retired_key!r}]")
    assert f"expected prefix {expected_prefix!r}" in matching[0].message
    assert "E010" not in errors(found), found

    replacement = expected_prefix + retired_key.split(".", 1)[1]
    swarm["nodes"][0]["ext"] = {replacement: {"kept": True}}
    assert errors(conf.validate_doc(swarm)) == []


def test_extension_prefix_rule_reaches_nested_ext_objects(swarm):
    swarm["attempts"][0].setdefault("cost", {})["ext"] = {
        "da" "gr.cost-detail": {"kept": True}
    }
    found = conf.validate_doc(swarm)
    assert errors(found) == ["E181"], found
    assert found[[f.code for f in found].index("E181")].path.startswith(
        "$['attempts'][0]['cost']['ext']"
    )


def test_extension_prefix_rule_does_not_enter_foreign_payload(swarm):
    swarm["artifacts"][2]["ext"] = {
        "com.example.tool": {"ext": {"da" "gr.thing": {"kept": True}}}
    }
    assert errors(conf.validate_doc(swarm)) == []


def test_extension_prefix_decision_pins_unknown_top_level_retired_reach(swarm):
    retired_key = "da" "gr.future"
    swarm["decision_probe"] = {"ext": {retired_key: {"kept": True}}}
    assert conf._schema_findings(swarm, conf.schema_path_for(swarm)) == []

    found = conf.validate_doc(swarm)
    matching = [finding for finding in found if finding.code == "E181"]
    assert errors(found) == ["E181"], found
    assert len(matching) == 1, found
    assert matching[0].path == f"$['decision_probe']['ext'][{retired_key!r}]"


def test_extension_prefix_decision_pins_unknown_top_level_corrected_limit(swarm):
    swarm["decision_probe"] = {
        "ext": {"dev.dagr.future": {"kept": True}},
    }
    assert errors(conf.validate_doc(swarm)) == []


def test_extension_prefix_decision_pins_unknown_top_level_foreign_limit(swarm):
    swarm["decision_probe"] = {
        "ext": {"com.example.future": {"kept": True}},
    }
    assert errors(conf.validate_doc(swarm)) == []


def test_extension_prefix_decision_pins_unknown_run_member_retired_reach(swarm):
    retired_key = "da" "gr.future"
    swarm["run"]["decision_probe"] = {"ext": {retired_key: {"kept": True}}}

    found = conf.validate_doc(swarm)
    matching = [finding for finding in found if finding.code == "E181"]
    assert errors(found) == ["E181"], found
    assert len(matching) == 1, found
    assert matching[0].path == (
        f"$['run']['decision_probe']['ext'][{retired_key!r}]"
    )


@pytest.mark.parametrize(
    ("case", "expected_ignored"),
    [
        pytest.param("shipped", 0, id="shipped"),
        pytest.param("top-retired", 1, id="unknown-top-retired"),
        pytest.param("top-corrected", 0, id="unknown-top-corrected"),
        pytest.param("top-foreign", 0, id="unknown-top-foreign"),
        pytest.param("top-no-ext", 0, id="unknown-top-no-ext"),
        pytest.param("run-retired", 1, id="unknown-run-retired"),
    ],
)
def test_reader_decision_accepts_schema_valid_unknown_members(
    swarm, case, expected_ignored
):
    retired_key = "da" "gr.future"
    if case == "top-retired":
        swarm["decision_probe"] = {"ext": {retired_key: {"kept": True}}}
    elif case == "top-corrected":
        swarm["decision_probe"] = {
            "ext": {"dev.dagr.future": {"kept": True}},
        }
    elif case == "top-foreign":
        swarm["decision_probe"] = {
            "ext": {"com.example.future": {"kept": True}},
        }
    elif case == "top-no-ext":
        swarm["decision_probe"] = {"future_value": {"kept": True}}
    elif case == "run-retired":
        swarm["run"]["decision_probe"] = {
            "ext": {retired_key: {"kept": True}},
        }

    assert conf._schema_findings(swarm, conf.schema_path_for(swarm)) == []
    graph = from_ocp(swarm)
    assert graph.meta["ocp_reader_retired_prefix_findings_ignored"] == expected_ignored


def test_layout_spike_v02_schema_matches_canonical_bytes():
    canonical = SPEC / "ocp-v0.2.schema.json"
    spike_copy = ROOT / "layout-spike" / "ocp" / "ocp" / "ocp-v0.2.schema.json"
    assert spike_copy.read_bytes() == canonical.read_bytes(), (
        f"{spike_copy.relative_to(ROOT)} differs from {canonical.relative_to(ROOT)}"
    )


def test_v02_schema_has_every_new_definition():
    v02 = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
    defs = v02["$defs"]
    for name in ("artifact", "origin", "role", "phase", "evidenceTier"):
        assert name in defs, name
    assert "artifacts" in v02["properties"]
    assert set(defs["edge"]["properties"]["kind"]["x-recommended"]) == {
        "dep", "fan_in", "spawn", "launch", "artifact",
    }
    for f in ("id", "path", "kind", "producer", "writers", "consumers", "first_write_at", "n_writes", "n_reads"):
        assert f in defs["artifact"]["properties"], f
    for f in ("launched_by", "workspace", "external", "how", "tier"):
        assert f in defs["origin"]["properties"], f
    # Amendment 13:10 (a) and (b): every listed field is required, tier included.
    assert set(defs["origin"]["required"]) == {"launched_by", "workspace", "external", "how", "tier"}
    for name in ("role", "phase"):
        assert set(defs[name]["required"]) == {"value", "tier", "evidence"}
    assert set(defs["artifact"]["required"]) == {
        "id", "path", "kind", "producer", "writers", "consumers", "first_write_at", "n_writes", "n_reads",
    }
    assert set(defs["artifact"]["properties"]["kind"]["required"]) == {"value", "tier"}
    assert set(defs["edge"]["required"]) == {"from", "to", "tier"}
    assert defs["evidenceTier"]["enum"] == ["verified", "heuristic", "reported"]
    cost = defs["cost"]["properties"]
    assert "cache_creation_5m_tokens" in cost and "cache_creation_1h_tokens" in cost
    assert cost["basis"]["x-recommended"] == ["measured", "allocated"]


def _recommended_schema_objects(schema):
    found = []

    def walk(value, field_path=()):
        if isinstance(value, dict):
            if "x-recommended" in value:
                found.append((".".join(field_path), value))
            for key, child in value.items():
                if key == "$defs":
                    for name, definition in child.items():
                        walk(definition, (name,))
                elif key == "properties":
                    for name, property_schema in child.items():
                        walk(property_schema, field_path + (name,))
                elif key != "x-recommended":
                    walk(child, field_path)
        elif isinstance(value, list):
            for item in value:
                walk(item, field_path)

    walk(schema)
    return found


def test_schema_and_checker_recommended_vocabularies_match():
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
    schema_vocabularies = {
        field: set(value["x-recommended"])
        for field, value in _recommended_schema_objects(schema)
    }
    checker_vocabularies = {
        field: set(values) for field, values in conf.RECOMMENDED_VOCABULARY.items()
    }
    mismatches = {}
    for field in sorted(schema_vocabularies.keys() | checker_vocabularies.keys()):
        schema_values = schema_vocabularies.get(field)
        checker_values = checker_vocabularies.get(field)
        if schema_values != checker_values:
            mismatches[field] = {
                "schema": sorted(schema_values) if schema_values is not None else None,
                "checker": sorted(checker_values) if checker_values is not None else None,
            }

    assert mismatches == {}


def test_recommended_values_appear_in_their_schema_descriptions():
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
    missing = {}
    for field, value in _recommended_schema_objects(schema):
        description_words = set(
            re.findall(r"[A-Za-z0-9_]+", value.get("description", ""))
        )
        absent = [item for item in value["x-recommended"] if item not in description_words]
        if absent:
            missing[field] = absent

    assert missing == {}


def test_v02_schema_accepts_capabilities_and_rejects_non_boolean(swarm):
    capabilities = swarm["producer"]["capabilities"]
    capabilities["producer_extension"] = True
    assert errors(conf.validate_doc(swarm)) == []

    capabilities["producer_extension"] = "yes"
    assert "E010" in errors(conf.validate_doc(swarm))


def test_v02_missing_producer_capabilities_is_a_warning(swarm):
    del swarm["producer"]["capabilities"]
    found = conf.validate_doc(swarm)
    assert errors(found) == []
    assert warnings(found) == ["W180"]


def test_false_capability_with_present_element_is_an_error(swarm):
    swarm["producer"]["capabilities"]["artifacts"] = False
    found = conf.validate_doc(swarm)
    assert errors(found) == ["E180"]
    assert any(
        finding.code == "E180"
        and finding.path == "$['producer']['capabilities']['artifacts']"
        for finding in found
    )


def test_v01_document_is_checked_against_v01_schema(minimal):
    # 'launch' is a v0.2 edge kind; a v0.1 document using it must fail its own schema.
    minimal["edges"].append({"from": "T1", "to": "R1", "kind": "launch"})
    assert "E010" in errors(conf.validate_doc(minimal))
    # The same edge in a v0.2 document is fine once every edge carries a tier,
    # which v0.2 requires and v0.1 never had.
    minimal["ocp"] = "0.2"
    found = errors(conf.validate_doc(minimal))
    assert set(found) == {"E010", "E160"}, found
    for e in minimal["edges"]:
        e["tier"] = "verified"
    assert errors(conf.validate_doc(minimal)) == []


def test_unknown_version_is_an_error_but_referential_checks_still_run(minimal):
    minimal["ocp"] = "0.9"
    minimal["attempts"][0]["node"] = "nope"
    found = errors(conf.validate_doc(minimal))
    assert "E003" in found and "E112" in found


def test_unknown_fields_are_ignored(swarm):
    swarm["future_field"] = {"anything": 1}
    swarm["attempts"][0]["future_field"] = "x"
    swarm["artifacts"][0]["future_field"] = 3
    swarm["edges"][0]["future_field"] = []
    assert errors(conf.validate_doc(swarm)) == []


def test_kind_vocabularies_are_open(swarm):
    swarm["artifacts"][0]["kind"]["value"] = "screenshot"
    swarm["nodes"][0]["kind"] = "wrangling"
    attempt(swarm, "S-lead.a1")["role"]["value"] = "conductor"
    attempt(swarm, "S-lead.a1")["phase"]["value"] = "warmup"
    assert errors(conf.validate_doc(swarm)) == []


def test_edge_kind_is_open_with_a_warning(swarm):
    swarm["edges"][0]["kind"] = "teleport"
    found = conf.validate_doc(swarm)
    assert errors(found) == []
    assert warnings(found) == ["W200"]


# --- artifacts ---------------------------------------------------------------

def test_artifact_edge_must_name_an_existing_artifact(swarm):
    e = edge(swarm, "S-plan", "S-dev", "artifact")
    del e["artifact"]
    assert "E116" in errors(conf.validate_doc(swarm))
    e["artifact"] = "docs/NOPE.md"
    found = errors(conf.validate_doc(swarm))
    assert "E117" in found and "E116" not in found


def test_artifact_edge_attempts_must_exist_and_sit_at_the_edge_nodes(swarm):
    e = edge(swarm, "S-plan", "S-dev", "artifact")
    e["from_attempt"] = "ghost.a1"
    assert "E123" in errors(conf.validate_doc(swarm))
    e["from_attempt"] = "S-dev.a1"  # exists, but is at S-dev, not S-plan
    found = errors(conf.validate_doc(swarm))
    assert "E124" in found and "E123" not in found


def test_artifact_edge_endpoints_must_be_the_producer_and_a_consumer(swarm):
    e = edge(swarm, "S-plan", "S-rev", "artifact")
    assert e["to_attempt"] == "S-rev.a1"
    art = next(a for a in swarm["artifacts"] if a["id"] == "docs/BUILD-PLAN.md")
    art["consumers"] = ["S-dev.a1"]  # the edge says S-rev.a1 consumed it
    found = conf.validate_doc(swarm)
    assert errors(found) == ["E164"], found
    assert found[0].path.endswith("['to_attempt']")
    # from_attempt must be the producer itself, not merely some writer: the
    # plan gets a second writer, S-lead.a1, and an edge from that writer is
    # rejected even though S-lead.a1 is among the writers.
    art["consumers"] = ["S-dev.a1", "S-rev.a1"]
    art["writers"] = ["S-plan.a1", "S-lead.a1"]
    art["n_writes"] = 2
    assert errors(conf.validate_doc(swarm)) == []
    swarm["edges"].append({"from": "S-lead", "to": "S-dev", "kind": "artifact", "tier": "verified",
                           "artifact": "docs/BUILD-PLAN.md",
                           "from_attempt": "S-lead.a1", "to_attempt": "S-dev.a1"})
    found = conf.validate_doc(swarm)
    assert errors(found) == ["E164"], found
    assert found[0].path.endswith("['from_attempt']") and "producer" in found[0].message


def test_artifact_edge_requires_both_attempt_refs(swarm):
    e = edge(swarm, "S-plan", "S-dev", "artifact")
    del e["to_attempt"]
    found = errors(conf.validate_doc(swarm))
    assert "E125" in found and "E010" in found  # checker rule and schema agree
    e["to_attempt"] = "S-dev.a1"
    del e["from_attempt"]
    assert "E125" in errors(conf.validate_doc(swarm))
    # A launch edge may still omit them.
    e["from_attempt"] = "S-plan.a1"
    launch = edge(swarm, "S-lead", "S-rev", "launch")
    del launch["from_attempt"], launch["to_attempt"]
    assert errors(conf.validate_doc(swarm)) == []


def test_artifact_refs_must_name_existing_attempts(swarm):
    art = swarm["artifacts"][0]
    art["producer"] = "ghost.a1"
    assert "E118" in errors(conf.validate_doc(swarm))
    art["producer"] = "S-plan.a1"
    art["consumers"].append("ghost.a2")
    assert "E118" in errors(conf.validate_doc(swarm))


def test_duplicate_artifact_id(swarm):
    swarm["artifacts"].append(copy.deepcopy(swarm["artifacts"][0]))
    assert "E103" in errors(conf.validate_doc(swarm))


def test_artifact_producer_must_be_the_first_writer(swarm):
    art = swarm["artifacts"][0]
    assert art["producer"] == "S-plan.a1" and art["writers"] == ["S-plan.a1"]
    art["producer"] = "S-dev.a1"  # exists, but not among writers at all
    found = conf.validate_doc(swarm)
    assert "E162" in errors(found) and warnings(found) == []
    # Among the writers but not first is still an error: the producer is writers[0].
    art["writers"] = ["S-plan.a1", "S-dev.a1"]
    art["n_writes"] = 2
    found = conf.validate_doc(swarm)
    assert "E162" in errors(found), found
    art["producer"] = "S-plan.a1"
    assert errors(conf.validate_doc(swarm)) == []


def test_artifact_writers_must_be_non_empty_and_unique(swarm):
    art = swarm["artifacts"][0]
    art["writers"] = []
    found = conf.validate_doc(swarm)
    assert "E166" in errors(found) and "E010" in errors(found)  # checker and schema agree
    art["writers"] = ["S-plan.a1", "S-plan.a1"]
    art["n_writes"] = 2
    found = conf.validate_doc(swarm)
    assert "E166" in errors(found) and "E010" in errors(found)
    # Consumers are a set too.
    art["writers"] = ["S-plan.a1"]
    art["n_writes"] = 1
    art["consumers"] = ["S-dev.a1", "S-rev.a1", "S-dev.a1"]
    art["n_reads"] = 3
    found = conf.validate_doc(swarm)
    assert "E166" in errors(found) and "E010" in errors(found)


@pytest.mark.parametrize("members,count", [("writers", "n_writes"), ("consumers", "n_reads")])
def test_artifact_counts_below_the_listed_members_are_errors(swarm, members, count):
    art = next(a for a in swarm["artifacts"] if a["id"] == "docs/BUILD-PLAN.md")
    art[count] = len(art[members]) - 1  # one fewer event than listed attempts
    found = conf.validate_doc(swarm)
    assert "E163" in errors(found), found
    assert warnings(found) == []
    assert any(f.code == "E163" and f.path.endswith(f"['{count}']") for f in found)
    art[count] = len(art[members]) + 1  # more events than attempts is fine
    assert errors(conf.validate_doc(swarm)) == []


@pytest.mark.parametrize("count,members", [("n_writes", "writers"), ("n_reads", "consumers")])
def test_null_count_is_unknown_not_an_undercount(swarm, count, members):
    # Amendment 13:10 (a), clarified 13:40: the schema accepts null for the
    # counts and the undercount check applies only to non-null values. A null
    # count with listed members is accepted; it is neither an undercount nor
    # treated as matching the members, it is simply unknown.
    art = next(a for a in swarm["artifacts"] if a["id"] == "docs/BUILD-PLAN.md")
    assert len(art[members]) >= 1
    art[count] = None
    found = conf.validate_doc(swarm)
    assert errors(found) == [] and warnings(found) == [], found
    assert not any(f.path.endswith(f"['{count}']") for f in found)
    # A known count below the listed members is still an error.
    art[count] = len(art[members]) - 1
    found = conf.validate_doc(swarm)
    assert "E163" in errors(found), found
    assert any(f.code == "E163" and f.path.endswith(f"['{count}']") for f in found)
    # An absent key is not the same as null: the field stays required.
    del art[count]
    assert "E010" in errors(conf.validate_doc(swarm))
    # A boolean is not a count.
    art[count] = True
    assert "E010" in errors(conf.validate_doc(swarm))


def test_null_first_write_at_is_accepted_but_absent_or_junk_is_not(swarm):
    art = swarm["artifacts"][0]
    art["first_write_at"] = None
    assert errors(conf.validate_doc(swarm)) == []
    art["first_write_at"] = "yesterday"
    assert "E010" in errors(conf.validate_doc(swarm))
    del art["first_write_at"]
    assert "E010" in errors(conf.validate_doc(swarm))


def test_all_three_artifact_unknowns_null_together_still_conforms(swarm):
    for art in swarm["artifacts"]:
        art["first_write_at"] = None
        art["n_writes"] = None
        art["n_reads"] = None
    found = conf.validate_doc(swarm)
    assert errors(found) == [] and warnings(found) == [], found


def test_shipped_v02_example_counts_match_its_events(swarm):
    # Amendment 13:10 (d): the example is internally consistent, and the event
    # log shows every write and cross-attempt read behind each count.
    for art in swarm["artifacts"]:
        wrote = [e for e in swarm["events"]
                 if e["type"] == "artifact_written" and e["detail"].startswith(art["id"])]
        read = [e for e in swarm["events"]
                if e["type"] == "artifact_read" and e["detail"].startswith(art["id"])]
        assert art["n_writes"] == len(wrote), art["id"]
        assert art["n_reads"] == len(read), art["id"]
        assert set(art["writers"]) == {e["attempt"] for e in wrote}, art["id"]
        assert set(art["consumers"]) == {e["attempt"] for e in read}, art["id"]
        assert art["producer"] == art["writers"][0] == wrote[0]["attempt"], art["id"]
        assert art["first_write_at"] == wrote[0]["at"], art["id"]


@pytest.mark.parametrize("field", [
    "kind", "producer", "writers", "consumers", "first_write_at", "n_writes", "n_reads",
])
def test_artifact_entity_fields_are_all_required(swarm, field):
    del swarm["artifacts"][1][field]
    assert "E010" in errors(conf.validate_doc(swarm))


def test_artifact_kind_requires_a_tier(swarm):
    del swarm["artifacts"][0]["kind"]["tier"]
    assert "E010" in errors(conf.validate_doc(swarm))
    swarm["artifacts"][0]["kind"]["tier"] = "guessed"
    assert "E010" in errors(conf.validate_doc(swarm))


def test_artifact_and_launch_edges_are_exempt_from_the_dag_rule(swarm):
    # S-lead -> S-rev (launch) and S-rev -> S-lead (artifact) already form a cycle.
    assert "E122" not in errors(conf.validate_doc(swarm))
    swarm["edges"].append({"from": "S-lead", "to": "S-rev", "kind": "dep", "tier": "verified"})
    swarm["edges"].append({"from": "S-rev", "to": "S-lead", "kind": "dep", "tier": "verified"})
    assert "E122" in errors(conf.validate_doc(swarm))


# --- origin, role, phase -------------------------------------------------------

def test_origin_launched_by_must_name_another_existing_attempt(swarm):
    rev = attempt(swarm, "S-rev.a1")
    rev["origin"]["launched_by"] = "ghost.a1"
    assert "E119" in errors(conf.validate_doc(swarm))
    rev["origin"]["launched_by"] = "S-rev.a1"
    assert "E119" in errors(conf.validate_doc(swarm))


def test_origin_launched_by_without_launch_edge_is_an_error(swarm):
    swarm["edges"] = [e for e in swarm["edges"] if e["kind"] != "launch"]
    assert errors(conf.validate_doc(swarm)) == ["E161"]
    # A launch edge between the wrong nodes does not satisfy it either.
    swarm["edges"].append({"from": "S-plan", "to": "S-rev", "kind": "launch", "tier": "heuristic"})
    assert errors(conf.validate_doc(swarm)) == ["E161"]
    swarm["edges"].append({"from": "S-lead", "to": "S-rev", "kind": "launch", "tier": "heuristic"})
    assert errors(conf.validate_doc(swarm)) == []


def test_origin_requires_a_tier(swarm):
    del attempt(swarm, "S-ext.a1")["origin"]["tier"]
    assert "E010" in errors(conf.validate_doc(swarm))
    attempt(swarm, "S-ext.a1")["origin"]["tier"] = "asserted"  # a v0.1 outcome word, not a tier
    assert "E010" in errors(conf.validate_doc(swarm))


@pytest.mark.parametrize("field", ["launched_by", "workspace", "external", "how"])
def test_origin_fields_are_required_and_unknown_is_an_explicit_null(swarm, field):
    origin = attempt(swarm, "S-ext.a1")["origin"]
    del origin[field]
    assert "E010" in errors(conf.validate_doc(swarm)), field
    origin[field] = None
    assert errors(conf.validate_doc(swarm)) == [], field


def test_origin_external_with_a_launcher_is_a_contradiction(swarm):
    rev = attempt(swarm, "S-rev.a1")
    assert rev["origin"]["launched_by"] == "S-lead.a1" and rev["origin"]["external"] is False
    rev["origin"]["external"] = True
    found = conf.validate_doc(swarm)
    assert "E165" in errors(found) and "E010" in errors(found), found  # checker and schema agree
    assert any(f.code == "E165" and f.path.endswith("['origin']") for f in found)
    # external with a null launcher is the documented shape (S-ext) and passes.
    rev["origin"]["launched_by"] = None
    swarm["edges"] = [e for e in swarm["edges"] if e["kind"] != "launch"]
    assert errors(conf.validate_doc(swarm)) == []
    # external null (no harness signal) with a launcher is not a contradiction.
    rev["origin"]["launched_by"] = "S-lead.a1"
    rev["origin"]["external"] = None
    swarm["edges"].append({"from": "S-lead", "to": "S-rev", "kind": "launch", "tier": "heuristic"})
    assert errors(conf.validate_doc(swarm)) == []


@pytest.mark.parametrize("record", ["origin", "role", "phase"])
def test_attempt_need_not_carry_origin_role_or_phase(swarm, record):
    # Amendment 13:10 (a), clarified 13:40: attempts are NOT required to carry
    # origin, role or phase at all. Removing the whole record from an attempt
    # (and from every attempt) leaves a conforming document. Inside a record
    # that is present, every listed field stays required (tests above).
    lead = attempt(swarm, "S-lead.a1")
    assert record in lead
    del lead[record]
    found = conf.validate_doc(swarm)
    assert errors(found) == [] and warnings(found) == [], found
    for a in swarm["attempts"]:
        a.pop(record, None)
    found = conf.validate_doc(swarm)
    assert errors(found) == [] and warnings(found) == [], found
    # The absence is at the attempt level only: an empty record is not a
    # substitute for no record.
    lead[record] = {}
    assert "E010" in errors(conf.validate_doc(swarm))


def test_attempt_with_none_of_the_three_records_conforms(swarm):
    lead = attempt(swarm, "S-lead.a1")
    for record in ("origin", "role", "phase"):
        del lead[record]
    found = conf.validate_doc(swarm)
    assert errors(found) == [] and warnings(found) == [], found
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
    required = schema["$defs"]["attempt"]["required"]
    assert not {"origin", "role", "phase"} & set(required), required
    assert "need not carry origin, role or phase at all" in schema["$defs"]["attempt"]["$comment"]


@pytest.mark.parametrize("label", ["role", "phase"])
def test_role_and_phase_require_value_tier_and_evidence(swarm, label):
    rec = attempt(swarm, "S-lead.a1")[label]
    del rec["evidence"]
    assert "E010" in errors(conf.validate_doc(swarm))
    rec["evidence"] = None  # unknown is an explicit null
    assert errors(conf.validate_doc(swarm)) == []
    tier = rec.pop("tier")
    assert "E010" in errors(conf.validate_doc(swarm))
    rec["tier"] = "asserted"
    assert "E010" in errors(conf.validate_doc(swarm))
    rec["tier"] = tier
    del rec["value"]
    assert "E010" in errors(conf.validate_doc(swarm))


# --- edge tiers ------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["spawn", "launch", "artifact", "dep"])
def test_edge_without_tier_is_an_error_whoever_the_producer_is(swarm, kind):
    if kind == "dep":
        swarm["edges"].append({"from": "S-plan", "to": "S-dev", "kind": "dep", "tier": "verified"})
    e = next(e for e in swarm["edges"] if e["kind"] == kind)
    del e["tier"]
    assert swarm["producer"]["name"] == "loopmath"
    found = conf.validate_doc(swarm)
    assert "E160" in errors(found) and "E010" in errors(found), found  # checker and schema agree
    swarm["producer"]["name"] = "some-other-converter"
    found = conf.validate_doc(swarm)
    assert "E160" in errors(found), found
    assert warnings(found) == []


def test_edge_tier_vocabulary_is_closed(swarm):
    swarm["edges"][0]["tier"] = "asserted"  # allowed on v0.1 outcome.evidence, never as a tier
    found = errors(conf.validate_doc(swarm))
    assert "E010" in found and "E160" in found


@pytest.mark.parametrize("name,expected", [
    ("loopmath", True), ("loopmath/graph", True), ("loopmath graph 0.1", True),
    ("dagrx", False), ("ocp-from-contractv3", False), ("herdr-dagr", False),
])
def test_extractor_producer_detection(name, expected):
    assert conf.is_extractor_producer({"producer": {"name": name}}) is expected


# --- cost ---------------------------------------------------------------------------

def test_extractor_costs_must_be_measured(swarm):
    lead = attempt(swarm, "S-lead.a1")
    lead["cost"]["basis"] = "allocated"
    assert "E171" in errors(conf.validate_doc(swarm))
    del lead["cost"]["basis"]
    assert "E171" in errors(conf.validate_doc(swarm))
    swarm["producer"]["name"] = "ocp-from-contractv3"
    assert "E171" not in errors(conf.validate_doc(swarm))


def test_cache_creation_buckets_must_sum_to_total(swarm):
    lead = attempt(swarm, "S-lead.a1")
    lead["cost"]["cache_creation_1h_tokens"] += 1
    assert errors(conf.validate_doc(swarm)) == ["E170"]
    lead["cost"]["cache_creation_1h_tokens"] -= 1
    assert errors(conf.validate_doc(swarm)) == []
    # Buckets are optional; a record with only the total is not checked.
    del lead["cost"]["cache_creation_1h_tokens"]
    assert errors(conf.validate_doc(swarm)) == []
    lead["cost"]["cache_creation_5m_tokens"] = -1
    assert "E010" in errors(conf.validate_doc(swarm))


def test_v01_document_does_not_gain_new_findings(minimal):
    # A v0.1 document has no tiers, no basis and no artifacts; none of the v0.2
    # rules may fire on it, and the v0.1 schema still governs it.
    found = conf.validate_doc(minimal)
    assert errors(found) == [] and warnings(found) == []


def test_v02_rules_are_gated_on_the_declared_version(minimal):
    # v0.2 fields inside a v0.1 document are unknown fields: ignored, never
    # checked, even when they would be wrong under v0.2. The same document
    # relabeled "0.2" trips every one of those rules.
    minimal["producer"]["name"] = "loopmath"
    minimal["artifacts"] = [{"id": "x", "path": "x"}]  # short of v0.2's required fields
    a0 = minimal["attempts"][0]
    a0["origin"] = {"launched_by": "ghost.a1", "tier": "heuristic"}
    a0["cost"] = {"basis": "allocated", "cache_creation_tokens": 5,
                  "cache_creation_5m_tokens": 1, "cache_creation_1h_tokens": 1}
    for e in minimal["edges"]:
        e.pop("tier", None)
    minimal["edges"].append({"from": "T1", "to": "R1", "kind": "dep",
                             "from_attempt": "ghost.a2", "to_attempt": "ghost.a3"})
    found = conf.validate_doc(minimal)
    assert errors(found) == [] and warnings(found) == [], found
    minimal["ocp"] = "0.2"
    found = errors(conf.validate_doc(minimal))
    for code in ("E010", "E119", "E123", "E160", "E170", "E171"):
        assert code in found, (code, found)


# --- CLI ------------------------------------------------------------------------------

def test_cli_exit_codes(tmp_path, capsys, swarm):
    ok = [str(EXAMPLES / n) for n in EXAMPLE_FILES]
    assert conf.main(ok) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 3 and "FAIL" not in out
    edge(swarm, "S-plan", "S-dev", "artifact")["artifact"] = "nope"
    bad = tmp_path / "bad.ocp.json"
    bad.write_text(json.dumps(swarm))
    assert conf.main([str(bad)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "E117" in out
    assert conf.main([]) == 2


# --- amendment (b), settled 14:15 -----------------------------------------------------

@pytest.mark.parametrize("kind", ["dep", "fan_in", "spawn"])
def test_tier_is_required_on_every_v02_edge_whatever_its_kind(swarm, kind):
    """Amendment (b), settled 14:15 (Analyst): a tier is required on every edge
    of a v0.2 document whatever its kind, not only on artifact and launch
    edges. A dep, fan_in or spawn edge without one is a conformance error from
    both the schema (E010) and the checker (E160)."""
    if kind == "spawn":
        e = edge(swarm, "S-lead", "S-plan", "spawn")
    else:
        e = {"from": "S-plan", "to": "S-dev", "kind": kind, "tier": "verified"}
        swarm["edges"].append(e)
    assert errors(conf.validate_doc(swarm)) == []
    del e["tier"]
    found = errors(conf.validate_doc(swarm))
    assert "E160" in found and "E010" in found, found
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
    assert "required on every edge of a v0.2 document, whatever its kind" in schema["$defs"]["edge"]["$comment"]


@pytest.mark.parametrize("tier", ["asserted", "measured", "", "VERIFIED", None, 1])
def test_edge_tier_accepts_only_the_three_section_0_2_words(swarm, tier):
    """Settled 14:15: an edge tier is verified, heuristic or reported and
    nothing else, whatever other tier words v0.1 allows on outcome.evidence."""
    edge(swarm, "S-lead", "S-plan", "spawn")["tier"] = tier
    found = errors(conf.validate_doc(swarm))
    assert "E160" in found and "E010" in found, found


@pytest.mark.parametrize("tier", ["verified", "heuristic", "reported"])
def test_each_of_the_three_tier_words_conforms_on_an_edge(swarm, tier):
    edge(swarm, "S-lead", "S-plan", "spawn")["tier"] = tier
    assert errors(conf.validate_doc(swarm)) == []


def _labelled_record(doc, which):
    if which == "artifact.kind":
        return doc["artifacts"][0]["kind"]
    return attempt(doc, "S-lead.a1")[which]


@pytest.mark.parametrize("which", ["role", "phase", "artifact.kind"])
def test_null_value_with_a_tier_conforms(swarm, which):
    """Settled 14:15: role.value, phase.value and artifact.kind.value accept
    null for unknown, with the tier still present."""
    rec = _labelled_record(swarm, which)
    rec["value"] = None
    assert "tier" in rec
    found = conf.validate_doc(swarm)
    assert errors(found) == [] and warnings(found) == [], found


@pytest.mark.parametrize("which", ["role", "phase", "artifact.kind"])
def test_null_value_without_a_tier_is_an_error(swarm, which):
    rec = _labelled_record(swarm, which)
    rec["value"] = None
    del rec["tier"]
    assert "E010" in errors(conf.validate_doc(swarm))
    rec["tier"] = None  # a null tier is not a tier
    assert "E010" in errors(conf.validate_doc(swarm))


@pytest.mark.parametrize("which", ["role", "phase", "artifact.kind"])
def test_absent_value_key_is_still_an_error(swarm, which):
    rec = _labelled_record(swarm, which)
    del rec["value"]
    assert "tier" in rec
    assert "E010" in errors(conf.validate_doc(swarm))


def test_null_writers_and_consumers_are_findings_not_exceptions(swarm):
    """Settled 14:15: the checker handles null in every list field without
    raising. writers: null and consumers: null each produce a finding from the
    checker itself (E167) and from the schema (E010)."""
    art = swarm["artifacts"][0]
    art["writers"] = None
    art["consumers"] = None
    found = conf.validate_doc(swarm)  # must not raise
    codes_found = errors(found)
    assert "E010" in codes_found, found
    assert codes_found.count("E167") == 2, found
    paths = {f.path for f in found if f.code == "E167"}
    assert paths == {"$['artifacts'][0]['writers']", "$['artifacts'][0]['consumers']"}, paths
    # the checker's own rule fires even with schema validation switched off
    checker_only = conf._referential_findings(swarm)
    assert codes(checker_only).count("E167") == 2, checker_only


@pytest.mark.parametrize("key", ["nodes", "edges", "attempts", "artifacts", "events", "groups"])
def test_null_top_level_list_is_a_finding_not_an_exception(swarm, key):
    swarm[key] = None
    found = conf.validate_doc(swarm)  # must not raise
    assert "E104" in errors(found), found
    assert any(f.code == "E104" and f.path == f"$['{key}']" for f in found)
