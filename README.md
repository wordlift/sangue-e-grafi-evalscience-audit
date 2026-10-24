# Sangue e Grafi: reproducible scorer audit

Companion artifact for **Can Structure Beat Scale? Evaluating Sangue e Grafi**, prepared for EvalScience 2026 by David Riccitelli, Andrea Volpini and Chiara Carrozza (WordLift).

Executed October 24, 2026. This is a constructed, deterministic audit of a public reward function, not a benchmark of model behavior or a replication of historical training runs.

## Reproduce the full audit

Python 3.14.2 was used. From this directory, with `uv` installed:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python audit.py
.venv/bin/python -m unittest -v test_audit
```

Dependency installation requires network access. The full audit runs offline after installation. It imports the unmodified vendored scorer, ontology and local graph backend. No model, credential, API key, training or remote graph service is used.

For coverage of the authored harness:

```sh
.venv/bin/python -m coverage run --source=audit -m unittest test_audit
.venv/bin/python -m coverage run --append --source=audit audit.py
.venv/bin/python -m coverage report -m
```

Observed: 9 test methods passed; all 175 executable statements in `audit.py` were covered across tests and the reporting entry point. This is statement coverage, not branch coverage or proof of verifier generality. A separate verification run repeated the 9 test methods and verified vendor hashes.

`minimal_reproduction.py` is the standalone eight-comparison reproduction script retained in this package. It downloads the pinned reward module and ontology and checks the eight headline scorer comparisons. It was also executed successfully, independently of the vendored full harness.

The revised PDF omits Python listings and links the upstream source at the frozen revision. This separate package contains the complete audit, fixture, tests, provenance, dependency versions and results. The reconstruction helpers remain in the manuscript source ZIP under reproduction/. Reconstruction reproduced the original JSON and CSV exactly, and all 9 tests passed. No experimental case or outcome changed.

## Findings

| Completion | Observed reward | Correct answer | Contract met |
|---|---:|---|---|
| Complete witness | 9.5 | Yes | Yes |
| Complete witness via entity retrieval | 3.0 | Yes | Yes |
| Nonexistent sources | 9.5 | Yes | No |
| Unrelated sources | 9.5 | Yes | No |
| Unsupported tool names | 9.5 | Yes | No |
| Omitted candidate attributes | 9.5 | Yes | No |
| Omitted competitor record | 9.5 | Yes | No |
| Complete witness, wrong answer | 3.5 | No | Yes |
| Ten repeated permitted calls, wrong answer | 10.0 | No | No |

All 13 main cases, 14 repetition-sweep calls and 4 graph-counterfactual calls are in `results.json`, including completion strings, actual tool observations, component scores, and verifier judgments. `results.csv` is the compact main-case table. These counts describe selected diagnostic controls, not a population rate. They are intentionally not summarized as a classifier false-positive rate: the original scalar reward has no audited acceptance threshold.

## Methods and independence

The fixture in `fixture.json` specifies a finite, closed-world graph: A has child B; B has grandchildren C (Carla, 30, alive), D (Dario, 20, alive), and E (Elena, 40, deceased). An unrelated U-V-W branch and spouse S are negative controls. The rule selects the eldest living biological grandchild of A, so Carla is correct. No ties or inference closure are included.

The scorer receives `optimal_hops=5`, a declared reference API-call count, not a proven optimum. Matched complete and retrieval-only witnesses both have 5 calls. Negative omissions can reduce the call count. A separate sensitivity check at reference counts 2 and 8 preserves the wrong-answer repetition's ranking above the complete witness.

The oracle computes the gold answer directly from JSON edges. The verifier separately checks replayed RDF responses against a JSON-only observation function, then derives the eligible winner from complete observed child lists and candidate attributes. It does not call the gold oracle or reward parser. Both supported query representations pass. Calls may be reordered. A pass is a conservative sufficient certificate under the declared assumptions, not a general necessary-and-sufficient test. Rejection means this contract is unmet; a deceased candidate's age, for example, need not be known to exclude them, and other tool representations are outside this checker. It does not certify temporal discovery or causal reasoning faithfulness. The legacy output field `evidence_sufficient` means “contract met”; `strict_success` additionally requires the answer to match the observed beneficiary. The word `authentic` denotes local consistency, not cryptographic authentication.

The `graph_sha256` field is the digest of the complete JSON fixture, including rule and reference count, rather than a serialized RDF graph. It is an identity binding in a local replay, not cryptographic authentication of external logs. Tests reject missing, forged, mismatched and stale records under these declared assumptions.

## Files and provenance

- `audit.py`: case construction, unmodified scorer calls, actual tool replay, independent verifier, output generation.
- `test_audit.py`: 9 behavioral test methods covering positive/negative controls, response tampering, counterfactuals and reference-count sensitivity.
- `fixture.json`: explicit graph and rule.
- `results.json` / `results.csv`: executed outputs.
- `coverage.json`: machine-readable statement coverage from tests plus CLI execution.
- `provenance.json`: exact upstream commit, source SHA-256 hashes and runtime versions.
- `requirements.lock.txt`: resolved versions used in the audit, including the coverage tool; version-pinned, not package-hash-locked or cross-platform validated.
- `vendor/`: unchanged upstream reward module, ontology and graph backend.
- `minimal_reproduction.py`: self-contained network-assisted reproduction of eight headline scorer comparisons.

Upstream: https://huggingface.co/spaces/cyberandy/sangue-e-grafi/tree/bb7344811e45625b558d60c02df183f23b8a8fcb

The upstream Space metadata declares the MIT license. Original source bytes and embedded notices are preserved. Authorship of those files remains with the upstream project. `provenance.json` records their origin; the audit code and fixture are newly prepared for this submission.

## Interpretation and reusable diagnostics

The target is a training reward whose source describes path validation and prevention of text shortcuts. Shaping may legitimately favor particular actions; the representation contrast alone does not invalidate every such preference. The wrong-answer ranking reversal and absent evidence under the stated validation aims motivate three checks for evidence-validating evaluators: removing required evidence removes evidence credit, repeated unchanged observations create no progress, and equivalent evidence receives consistent validation with tool costs reported separately.

## Interpretation limits

This audit demonstrates counterexamples in one public implementation. It does not establish how often models exploit them, a change in model accuracy, a cause of any historical training result, universal validity of the verifier, real-world inheritance competence, or enterprise performance of MOSAIC-KG. No event submission has been sent as of artifact preparation.
