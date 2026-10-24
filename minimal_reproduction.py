from pathlib import Path
from urllib.request import urlopen
import hashlib
import importlib.util
import tempfile

commit = "bb7344811e45625b558d60c02df183f23b8a8fcb"
base = ("https://huggingface.co/spaces/cyberandy/"
        "sangue-e-grafi/resolve/" + commit + "/")
d = Path(tempfile.mkdtemp())
for src, dst in [("src/training/reward_functions.py", "r.py"),
                 ("data/kinship.ttl", "k.ttl")]:
    (d / dst).write_bytes(urlopen(base + src).read())
assert hashlib.sha256((d / "r.py").read_bytes()).hexdigest() == (
    "ca5a2585f4667e2080e51729f2cad8d424a57"
    "67b90c8960b9c43c8ea1e79715b")
spec = importlib.util.spec_from_file_location("r", d / "r.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
v = r.OntologyRewardValidator(str(d / "k.ttl"))
def f(x, tool="follow_entity_link"):
    return (f'<action tool="{tool}" entity="{x}" '
            'property="isBiologicalParentOf"/>')
def g(x):
    return f'<action tool="get_entity" entity="{x}"/>'
def score(actions, answer="Carla"):
    text = ("<reasoning>Check the stipulated rule.</reasoning>"
            + "".join(actions) + f"<answer>{answer}</answer>")
    return r.kinship_ontology_reward(
        [text], ["eldest living biological grandchild of A"],
        validator=v, gold_answers=["Carla"], optimal_hops=[5],
        will_clauses=[{"requires": ["isBiologicalParentOf"],
                       "excludes": ["isMarriedTo"]}])[0]
tail = [g(x) for x in "CDE"]
assert score([f("A"), f("B")] + tail) == 9.5
assert score([g(x) for x in "ABCDE"]) == 3.0
assert score([f("A")] * 10, "Sofia") == 10.0
bad = [[f("Ghost1"), f("Ghost2")] + tail,
       [f("U"), f("V")] + tail,
       [f("A", "invented_tool"), f("B", "invented_tool")] + tail,
       [f("A"), f("B")],
       [f("A"), f("B"), g("C"), g("E")]]
assert [score(x) for x in bad] == [9.5] * 5
print("All eight scorer comparisons reproduced.")
