import io
import json
import os
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

# Required because inference.py validates HF_TOKEN at import time.
os.environ.setdefault("HF_TOKEN", "test-token")

import inference  # noqa: E402


class TestInferenceOutputFormat(unittest.TestCase):
    def _mock_create(self, **kwargs):
        user_content = kwargs["messages"][1]["content"]
        obs_json = user_content.replace("Current Observation: ", "", 1)
        obs = json.loads(obs_json)

        if not obs.get("issue_category"):
            action = {"action_type": "classify_issue", "category_guess": "Billing"}
        elif not obs.get("knowledge_base_result"):
            action = {"action_type": "search_kb", "search_query": "billing receipt"}
        else:
            action = {
                "action_type": "resolve_ticket",
                "message_to_customer": "Your billing receipt is now available.",
            }

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(action)))]
        )

    def test_stdout_format_for_one_episode(self):
        start_re = re.compile(r"^\[START\] task=\S+ env=\S+ model=.+$")
        step_re = re.compile(
            r"^\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d{2} done=(true|false) error=.*$"
        )
        end_re = re.compile(
            r"^\[END\] success=(true|false) steps=\d+ rewards=-?\d+\.\d{2}(,-?\d+\.\d{2})*$"
        )

        with patch.object(inference, "TASK_ORDER", ["easy"]):
            with patch.object(
                inference.client.chat.completions, "create", side_effect=self._mock_create
            ):
                out = io.StringIO()
                err = io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    inference.run_baseline()

        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)

        start_lines = [line for line in lines if line.startswith("[START]")]
        step_lines = [line for line in lines if line.startswith("[STEP]")]
        end_lines = [line for line in lines if line.startswith("[END]")]

        self.assertEqual(len(start_lines), 1)
        self.assertGreaterEqual(len(step_lines), 1)
        self.assertEqual(len(end_lines), 1)

        self.assertTrue(start_re.match(start_lines[0]))
        for line in step_lines:
            self.assertTrue(step_re.match(line))
        self.assertTrue(end_re.match(end_lines[0]))

        self.assertEqual(lines[0][:7], "[START]")
        self.assertEqual(lines[-1][:5], "[END]")
        self.assertTrue(all(line.startswith("[STEP]") for line in lines[1:-1]))


if __name__ == "__main__":
    unittest.main()
