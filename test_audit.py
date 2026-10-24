"""Behavioral tests for the audit's independent evidence contract."""
import copy
import unittest

import audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.data = audit.fixture()
        self.validator = audit.reward.OntologyRewardValidator(str(audit.ROOT / "vendor/kinship.ttl"))
        self.actions = next(audit.cases())[1]

    def check(self, actions=None, answer="Carla", data=None):
        actions = self.actions if actions is None else actions
        data = self.data if data is None else data
        text = audit.completion(actions, answer)
        return audit.verify(text, audit.replay(actions, data), data)

    def test_positive_representations_and_reordered_calls(self):
        for actions in [self.actions, [audit.get(x) for x in "ABCDE"], list(reversed(self.actions))]:
            with self.subTest(actions=actions):
                self.assertTrue(self.check(actions)["strict_success"])

    def test_fixed_case_expectations(self):
        expected = {"complete_witness": (9.5, True), "get_only_witness": (3.0, True),
                    "nonexistent_sources": (9.5, False), "disconnected_sources": (9.5, False),
                    "unknown_tool": (9.5, False), "omit_all_attributes": (9.5, False),
                    "omit_competitor": (9.5, False), "blind_correct": (2.0, False),
                    "mention_correct": (2.5, False), "wrong_answer": (3.5, False),
                    "excluded_spouse": (0.0, False), "repeat10_wrong": (10.0, False),
                    "repeat10_correct": (16.0, False)}
        for case in audit.cases():
            with self.subTest(case=case[0]):
                row = audit.run_case(*case, self.data, self.validator)
                self.assertEqual((row["reward"], row["verification"]["strict_success"]), expected[case[0]])

    def test_mutated_age_and_survival_change_winner(self):
        for node, field, value, winner in [("D", "age", 35, "Dario"), ("E", "isAlive", True, "Elena")]:
            data = copy.deepcopy(self.data)
            data["persons"][node][field] = value
            self.assertEqual(audit.gold_answer(data), winner)
            self.assertEqual(self.check(data=data)["derived_answer"], winner)
            self.assertFalse(self.check(data=data)["strict_success"])
            self.assertTrue(self.check(answer=winner, data=data)["strict_success"])

    def test_forged_or_stale_observations_rejected(self):
        text = audit.completion(self.actions)
        original = audit.replay(self.actions, self.data)
        mutations = []
        missing_competitor = copy.deepcopy(original)
        missing_competitor[1]["observation"].pop()
        mutations.append(missing_competitor)
        fake_age = copy.deepcopy(original)
        fake_age[2]["observation"]["age"] = 99
        mutations.append(fake_age)
        stale_version = copy.deepcopy(original)
        stale_version[0]["graph_sha256"] = "stale"
        mutations.append(stale_version)
        wrong_binding = copy.deepcopy(original)
        wrong_binding[0]["action"]["entity"] = "U"
        mutations.append(wrong_binding)
        mutations.append(original[:-1])
        for events in mutations:
            self.assertFalse(audit.verify(text, events, self.data)["strict_success"])

    def test_wrong_answer_has_sufficient_evidence_but_fails(self):
        result = self.check(answer="Sofia")
        self.assertTrue(result["evidence_sufficient"])
        self.assertFalse(result["strict_success"])

    def test_missing_root_branch_or_candidate_details_fail(self):
        for index in range(len(self.actions)):
            with self.subTest(removed=index):
                actions = self.actions[:index] + self.actions[index + 1:]
                self.assertFalse(self.check(actions)["strict_success"])

    def test_missing_person_and_empty_adjacency(self):
        actions = [audit.get("Ghost"), audit.action("C"), audit.action("A", prop="unknown")]
        self.assertFalse(self.check(actions)["strict_success"])

    def test_no_living_grandchild_fails_closed(self):
        data = copy.deepcopy(self.data)
        for node in "CDE":
            data["persons"][node]["isAlive"] = False
        self.assertIsNone(self.check(data=data)["derived_answer"])
        self.assertFalse(self.check(data=data)["strict_success"])

    def test_repetition_and_reference_budget_sensitivity(self):
        for budget in [2, 5, 8]:
            data = copy.deepcopy(self.data)
            data["optimal_hops"] = budget
            valid = audit.run_case("valid", self.actions, "Carla", "Check.", data, self.validator)
            wrong = audit.run_case("wrong", [audit.action("A")] * 10, "Sofia", "Check.", data, self.validator)
            self.assertGreater(wrong["reward"], valid["reward"])
        for n in [8, 10, 16, 32]:
            wrong = audit.run_case("wrong", [audit.action("A")] * n, "Sofia", "Check.", self.data, self.validator)
            self.assertEqual(wrong["reward"], float(n))


if __name__ == "__main__":
    unittest.main()
