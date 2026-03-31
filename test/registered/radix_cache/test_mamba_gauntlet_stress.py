from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=300, suite="nightly-amd-1-gpu", nightly=True)

import concurrent.futures
import json
import os
import re
import time
import unittest
import uuid

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:30000")
LONG_SYSTEM = "You are a helpful assistant. " * 60


def strip_markdown_json(content: str) -> str:
    cleaned = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        return cleaned[1:-1].strip()
    return cleaned


class TestMambaGauntletStress(unittest.TestCase):
    def setUp(self):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code != 200:
                self.skipTest("Server not available")
        except Exception:
            self.skipTest("Server not available")

    def _chat(self, messages, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 30)
        payload = {"model": "default", "messages": messages, **kwargs}
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def test_high_concurrency_shared_prefix(self):
        N = 32
        base_messages = [{"role": "system", "content": LONG_SYSTEM}]
        questions = [f"What is {i} + {i}?" for i in range(N)]
        def send(q):
            resp = self._chat(base_messages + [{"role": "user", "content": q}])
            return resp["choices"][0]["message"]["content"].strip()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
            results = list(ex.map(send, questions))
        self.assertEqual(len(results), N)
        for r in results:
            self.assertGreater(len(r), 0, f"Empty response: {r}")

    def test_rapid_distinct_requests_eviction_pressure(self):
        errors = []
        for i in range(100):
            try:
                resp = self._chat([
                    {"role": "user", "content": f"Unique-{uuid.uuid4().hex}: say ok{i}"},
                ], max_tokens=5)
                self.assertIn(resp["choices"][0]["finish_reason"], ("stop", "length"))
            except Exception as e:
                errors.append(str(e))
        self.assertEqual(errors, [], f"Errors during eviction stress: {errors}")

    def test_repeated_same_request_cache_stability(self):
        messages = [{"role": "user", "content": "Reply with exactly: STABLE"}]
        outputs = []
        for _ in range(50):
            resp = self._chat(messages, max_tokens=10)
            outputs.append(resp["choices"][0]["message"]["content"].strip())
        for o in outputs:
            self.assertGreater(len(o), 0, f"Empty output in repetition run: {outputs}")
        unique = set(outputs)
        self.assertEqual(len(unique), 1,
            f"Outputs diverged across runs — expected all 50 identical at temperature=0: {unique}")

    def test_alternating_long_and_short_requests(self):
        long_msgs = [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "Summarize your role."},
        ]
        short_msgs = [{"role": "user", "content": "Say: short"}]
        for i in range(20):
            long_resp = self._chat(long_msgs, max_tokens=20)
            short_resp = self._chat(short_msgs, max_tokens=5)
            self.assertGreater(len(long_resp["choices"][0]["message"]["content"]), 0)
            self.assertGreater(len(short_resp["choices"][0]["message"]["content"]), 0)

    def test_server_health_after_stress(self):
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        self.assertEqual(r.status_code, 200, "Server became unhealthy after stress tests")

    def test_concurrent_multi_turn_conversations(self):
        personas = [f"User{i}" for i in range(8)]
        def run_conversation(persona):
            history = []
            history.append({"role": "user", "content": f"My name is {persona}. Reply with JSON: {{\"name\":\"{persona}\"}}"})
            resp = self._chat(history, max_tokens=60)
            history.append({"role": "assistant", "content": resp["choices"][0]["message"]["content"]})
            try:
                parsed = json.loads(strip_markdown_json(resp["choices"][0]["message"]["content"]))
            except (json.JSONDecodeError, ValueError):
                return f"FAIL: {persona} gave non-JSON response: {resp['choices'][0]['message']['content']}"
            if parsed.get("name") != persona:
                return f"FAIL: {persona} name mismatch: {parsed.get('name')}"
            for turn in range(4):
                history.append({"role": "user", "content": f"Turn {turn+2}: what is my name? Reply with JSON: {{\"name\":\"{persona}\"}}"})
                resp = self._chat(history, max_tokens=60)
                content = resp["choices"][0]["message"]["content"]
                history.append({"role": "assistant", "content": content})
                try:
                    parsed = json.loads(strip_markdown_json(content))
                except (json.JSONDecodeError, ValueError):
                    return f"FAIL: {persona} turn {turn+2} non-JSON: {content}"
                if parsed.get("name") != persona:
                    return f"FAIL: {persona} turn {turn+2} name mismatch: {parsed.get('name')}"
            return f"PASS: {persona}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(run_conversation, personas))
        failures = [r for r in results if r.startswith("FAIL")]
        self.assertEqual(failures, [], f"Conversation coherence failures: {failures}")


if __name__ == "__main__":
    unittest.main()
