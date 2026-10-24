"""Offline, constructed counterexamples to a frozen public reward function."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

from rdflib import Graph, Literal, Namespace, RDF

ROOT = Path(__file__).resolve().parent
EX = Namespace("http://example.org/kinship#")
BIO = "isBiologicalParentOf"
COMMIT = "bb7344811e45625b558d60c02df183f23b8a8fcb"


def load_vendor(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "vendor" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward = load_vendor("reward_functions")
graph_tools = load_vendor("tools")


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return sorted((canonical(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fixture():
    return json.loads((ROOT / "fixture.json").read_text())


def build_graph(data):
    graph = Graph()
    for node, fields in data["persons"].items():
        graph.add((EX[node], RDF.type, EX.Person))
        for key, value in fields.items():
            graph.add((EX[node], EX[key], Literal(value)))
    for key, prop in [("biological_edges", BIO), ("marriage_edges", "isMarriedTo")]:
        for source, target in data[key]:
            graph.add((EX[source], EX[prop], EX[target]))
    return graph


def gold_answer(data):
    """Independent rule oracle over JSON, not scorer or RDF query helpers."""
    children = {b for a, b in data["biological_edges"] if a == data["root"]}
    grandchildren = {b for a, b in data["biological_edges"] if a in children}
    living = [data["persons"][p] for p in grandchildren if data["persons"][p]["isAlive"]]
    return max(living, key=lambda p: p["age"])["fullName"]


def action(entity, tool="follow_entity_link", prop=BIO):
    attrs = {"tool": tool, "entity": entity}
    if prop is not None:
        attrs["property"] = prop
    return attrs


def get(entity):
    return action(entity, "get_entity", None)


def completion(actions, answer="Carla", reasoning="Check the stipulated rule."):
    parts = [f"<reasoning>{reasoning}</reasoning>"]
    parts += [ET.tostring(ET.Element("action", a), encoding="unicode") for a in actions]
    parts += [f"<answer>{answer}</answer>"]
    return "\n".join(parts)


def parse(text):
    root = ET.fromstring(f"<trace>{text}</trace>")
    return [dict(e.attrib) for e in root.findall("action")], root.findtext("answer")


def replay(actions, data):
    backend = graph_tools.RdflibTools(build_graph(data))
    events = []
    for index, attrs in enumerate(actions):
        tool, source = attrs.get("tool"), attrs.get("entity", "")
        if tool == "follow_entity_link":
            result = backend.follow_entity_link(str(EX[source]), str(EX[attrs["property"]]))
        elif tool == "get_entity":
            result = backend.get_entity(str(EX[source]))
        else:
            result = {"error": "unsupported_tool"}
        events.append({"index": index, "graph_sha256": digest(data),
                       "action": attrs, "observation": canonical(result)})
    return events


def expected_observation(attrs, data):
    """JSON-only check of actual tool responses, independent of RDF backend."""
    source, tool = attrs.get("entity", ""), attrs.get("tool")
    if tool == "follow_entity_link":
        prop = attrs["property"]
        edges = data["biological_edges"] if prop == BIO else data["marriage_edges"] if prop == "isMarriedTo" else []
        return canonical([{"property": str(EX[prop]), "target_iri": str(EX[b]),
                           "target_label": data["persons"][b]["fullName"]}
                          for a, b in edges if a == source])
    if tool == "get_entity":
        result = {"iri": str(EX[source])}
        if source in data["persons"]:
            result.update(data["persons"][source], type=str(EX.Person))
        for key, prop in [("biological_edges", BIO), ("marriage_edges", "isMarriedTo")]:
            targets = [str(EX[b]) for a, b in data[key] if a == source]
            if targets:
                result[prop] = targets[0] if len(targets) == 1 else targets
        return canonical(result)
    return {"error": "unsupported_tool"}


def verify(text, events, data):
    """Check provenance first; derive witness using only authenticated observations.

    The fixed fixture asserts complete adjacency/property results (closed world).
    This certifies sufficiency of evidence, not the model's internal reasoning.
    """
    actions, answer = parse(text)
    authentic = len(actions) == len(events)
    for i, (attrs, event) in enumerate(zip(actions, events)):
        authentic &= (event["index"] == i and event["action"] == attrs
                      and event["graph_sha256"] == digest(data)
                      and event["observation"] == expected_observation(attrs, data))
    execution_ok = authentic and all(a.get("tool") in {"get_entity", "follow_entity_link"}
                                    and a.get("entity") in data["persons"] for a in actions)
    enumerations, details = {}, {}
    if execution_ok:
        for attrs, event in zip(actions, events):
            source, obs = attrs["entity"], event["observation"]
            if attrs["tool"] == "follow_entity_link" and attrs.get("property") == BIO:
                enumerations[source] = {r["target_iri"].split("#")[-1] for r in obs}
            elif attrs["tool"] == "get_entity":
                details[source] = obs
                children = obs.get(BIO, [])
                if isinstance(children, str):
                    children = [children]
                enumerations[source] = {c.split("#")[-1] for c in children}
    root = data["root"]
    covered = root in enumerations and all(c in enumerations for c in enumerations.get(root, set()))
    candidates = set().union(*(enumerations[c] for c in enumerations[root])) if covered else set()
    attributes = bool(candidates) and all(
        p in details and {"fullName", "isAlive", "age"} <= details[p].keys() for p in candidates)
    winner = None
    if attributes:
        living = [details[p] for p in candidates if details[p]["isAlive"]]
        if living:
            winner = max(living, key=lambda p: p["age"])["fullName"]
    sufficient = execution_ok and covered and attributes
    return {"authentic": bool(authentic), "execution_ok": bool(execution_ok),
            "candidate_coverage": bool(covered), "attribute_coverage": bool(attributes),
            "evidence_sufficient": bool(sufficient), "derived_answer": winner,
            "strict_success": bool(sufficient and answer == winner)}


def components(text, data, validator):
    """Analytical decomposition; total cross-checked against unchanged function."""
    actions, predicted = parse(text)
    required = data["will_clause"]["requires"]
    n, optimal = len(actions), data["optimal_hops"]
    path = sum(validator.validate_traversal_step(a["property"], data["will_clause"])
               for a in actions if a.get("property"))
    direct = any(a.get("property") in required for a in actions)
    mention = any(p in text for p in required)
    correct = predicted == gold_answer(data)
    raw = {"format": 1.0, "ontology": path,
           "answer": (5.0 if direct else 1.0 if mention else 0.5) if correct else -1.0,
           "efficiency": (1.0 if n <= optimal else 0.5 if n <= optimal + 2 else 0.0) if n else 0.0,
           "termination": 0.5 if n <= optimal + 1 else 0.0}
    raw["floor_adjustment"] = max(0.0, -sum(raw.values()))
    return raw


def cases():
    full = [action("A"), action("B"), get("C"), get("D"), get("E")]
    yield "complete_witness", full, "Carla", "Check the stipulated rule."
    yield "get_only_witness", [get(x) for x in "ABCDE"], "Carla", "Check the stipulated rule."
    yield "nonexistent_sources", [action("Ghost1"), action("Ghost2"), *full[2:]], "Carla", "Check the stipulated rule."
    yield "disconnected_sources", [action("U"), action("V"), *full[2:]], "Carla", "Check the stipulated rule."
    yield "unknown_tool", [action("A", "invented_tool"), action("B", "invented_tool"), *full[2:]], "Carla", "Check the stipulated rule."
    yield "omit_all_attributes", full[:2], "Carla", "Check the stipulated rule."
    yield "omit_competitor", full[:3] + full[4:], "Carla", "Check the stipulated rule."
    yield "blind_correct", [], "Carla", "Check the stipulated rule."
    yield "mention_correct", [], "Carla", f"Use {BIO}."
    yield "wrong_answer", full, "Sofia", "Check the stipulated rule."
    yield "excluded_spouse", [action("A", prop="isMarriedTo")], "Sofia", "Check the stipulated rule."
    yield "repeat10_wrong", [action("A")] * 10, "Sofia", "Check the stipulated rule."
    yield "repeat10_correct", [action("A")] * 10, "Carla", "Check the stipulated rule."


def run_case(name, actions, answer, reasoning, data, validator):
    text = completion(actions, answer, reasoning)
    events = replay(actions, data)
    actual = reward.kinship_ontology_reward(
        [text], [data["rule"]], validator=validator, gold_answers=[gold_answer(data)],
        will_clauses=[data["will_clause"]], optimal_hops=[data["optimal_hops"]])[0]
    breakdown = components(text, data, validator)
    assert sum(breakdown.values()) == actual, (name, breakdown, actual)
    return {"id": name, "completion": text, "events": events, "reward": actual,
            "components": breakdown, "gold": gold_answer(data), "predicted": answer,
            "correct": answer == gold_answer(data), "calls": len(actions),
            "verification": verify(text, events, data)}


def run():
    data = fixture()
    validator = reward.OntologyRewardValidator(str(ROOT / "vendor" / "kinship.ttl"))
    rows = [run_case(*case, data, validator) for case in cases()]
    sweep = [run_case(f"repeat_{n}_{answer}", [action("A")] * n, answer,
                     "Check the stipulated rule.", data, validator)
             for n in [1, 2, 4, 8, 10, 16, 32] for answer in ["Carla", "Sofia"]]
    counterfactuals = []
    for node, field, value in [("D", "age", 35), ("E", "isAlive", True)]:
        altered = copy.deepcopy(data)
        altered["persons"][node][field] = value
        for answer in ["Carla", gold_answer(altered)]:
            result = run_case(f"{node}_{field}_{answer}", next(cases())[1], answer,
                              "Check the stipulated rule.", altered, validator)
            result["fixture"] = altered
            counterfactuals.append(result)
    report = {"commit": COMMIT, "fixture_sha256": digest(data),
              "python": platform.python_version(), "main_cases": rows,
              "repetition_sweep": sweep, "counterfactuals": counterfactuals}
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "results.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "calls", "reward", "correct", "evidence_sufficient", "strict_success"])
        for row in rows:
            v = row["verification"]
            writer.writerow([row["id"], row["calls"], row["reward"], row["correct"], v["evidence_sufficient"], v["strict_success"]])
            print(f"{row['id']:24} reward={row['reward']:4.1f} correct={row['correct']} witness={v['strict_success']}")


if __name__ == "__main__":
    run()
