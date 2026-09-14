"""Exercise the real runner's lifecycle with fake model and native clients."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from controlled_reader import DATA_SHA256, MODEL, MODEL_REVISION
from materialize_inputs import write_once
from native_six import EMBEDDING_MODEL, EMBEDDING_REVISION
from prepare_inputs import REFERENCE_SHA256, digest
import run_history


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "generalization_20260908/full_transfer/verified_qwen35_lme500_start/protocol.json"


class RunnerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(self.temp).resolve()
        self.events, self.requests = [], []
        self.fail_init = self.fail_reader = False
        old_cwd, old_path = Path.cwd(), list(sys.path)
        self.addCleanup(os.chdir, old_cwd)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        self.enterContext(patch.dict(os.environ, {"LIGHTMEM_MODEL": "stale-model"}))
        protocol = json.loads(REFERENCE.read_bytes())
        qid = protocol["selection"]["selected_ids"][0]
        source = [dict(session_id="s1", date="2023/05/20 (Sat) 02:21",
                       turns=[dict(role="user", content="original fact"),
                              dict(role="assistant", content="original reply")])]
        query = dict(question_id=qid, question="TARGET QUESTION", question_date="date")
        entry = dict(source_sha256=digest(source), query_sha256=digest(query))
        manifest = dict(dataset_sha256=DATA_SHA256, reference_protocol_sha256=REFERENCE_SHA256,
                        questions=[dict(question_id=item, **entry)
                                   for item in protocol["selection"]["selected_ids"]])
        inputs = root / "inputs"
        write_once(inputs / "manifest.json", manifest)
        write_once(inputs / "sources" / f'{entry["source_sha256"]}.json', source)
        write_once(inputs / "queries" / f'{entry["query_sha256"]}.json', query)
        roots = {}
        for name in ("source", "higmem", MODEL_REVISION):
            folder = root / name
            folder.mkdir()
            (folder / "test_fixture.txt").write_text("fixture")
            roots[name] = folder
        stub_hash = hashlib.sha256(b"fixture").hexdigest()
        receipt = dict(
            status="baseline_services_verified_after_qwen_complete",
            model=MODEL, model_revision=MODEL_REVISION, dtype="bfloat16",
            embedding_model=EMBEDDING_MODEL, embedding_revision=EMBEDDING_REVISION,
            embedding_dimensions=1024, qwen_completed_questions_per_arm=500,
            qwen_completed_arms=3, gemma_held=True, run_id="CPU_FAKE_TEST_ONLY",
            llm_url="http://fake-llm/v1", embedding_url="http://fake-embedding/v1",
            input_manifest_sha256=digest(manifest),
            source_files_sha256={"test_fixture.txt": stub_hash},
            higmem_files_sha256={"test_fixture.txt": stub_hash},
            tokenizer_files_sha256={"test_fixture.txt": stub_hash},
            harness_files_sha256={"run_history.py": hashlib.sha256(
                Path(run_history.__file__).read_bytes()).hexdigest()},
        )
        write_once(root / "receipt.json", receipt)
        self.args = argparse.Namespace(
            method="simplemem", question_id=qid, inputs=inputs, reference=REFERENCE,
            source_root=roots["source"], higmem_root=roots["higmem"],
            tokenizer=roots[MODEL_REVISION], output=root / "attempt",
            runtime_receipt=root / "receipt.json",
            llm_url=receipt["llm_url"], embedding_url=receipt["embedding_url"],
        )
        owner = self

        class Agent:
            def __init__(self, config, dataset, load_agent_from):
                owner.events.append("init")
                self.state = Path(load_agent_from)
                (self.state / "native_state.txt").write_text("preserve this")
                if owner.fail_init:
                    raise RuntimeError("native init failed")
                owner.assertEqual(os.environ["LIGHTMEM_MODEL"], MODEL)
                owner.assertEqual(os.environ["BASELINE_STRICT_COMPARISON"], "1")
                self.simplemem = self
                self._initialize_tokenizer()

            def add_chunk(self, text, timestamp):
                owner.events.append(("write", text, timestamp))
                owner.assertNotIn("TARGET QUESTION", text)

            def save_agent(self):
                owner.events.append("finalize")

            def retrieve_entries(self, query):
                owner.assertIn("finalize", owner.events)
                owner.events.append("retrieve")
                return [{"text": "original fact"}]

            def close(self):
                owner.events.append("close")

        class Client:
            def __init__(self, base_url, **kwargs):
                model = MODEL if base_url == owner.args.llm_url else EMBEDDING_MODEL
                self.models = NS(list=lambda: NS(data=[NS(id=model)]))
                self.chat = NS(completions=NS(create=self.create))

            def create(self, **request):
                owner.requests.append(request)
                if owner.fail_reader:
                    raise RuntimeError("reader transport failed")
                return NS(
                    choices=[NS(finish_reason="length", message=NS(content="answer"))],
                    usage=NS(model_dump=lambda: {"prompt_tokens": 30, "completion_tokens": 96}),
                    model_dump=lambda: {"test": "fake reader response"},
                )

            def close(self):
                pass

        modules = {
            "utils": NS(__path__=[]),
            "utils.agent": NS(AgentWrapper=Agent),
            "utils.request_metering": NS(meter_operation=lambda *a, **k: nullcontext()),
            "utils.provider_utils": NS(HuggingFaceTokenizerAdapter=lambda tokenizer: tokenizer),
            "transformers": NS(AutoTokenizer=NS(from_pretrained=lambda *a, **k: NS(
                encode=lambda text, **kw: list(text)))),
            "openai": NS(OpenAI=Client),
        }
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(run_history, "build_config", return_value=({}, {})))

    def test_flush_source_isolation_and_delayed_query_read(self):
        original = run_history.read_json

        def checked_read(path):
            if path.parent.name == "queries":
                self.assertIn("finalize", self.events)
            return original(path)

        with patch.object(run_history, "read_json", side_effect=checked_read):
            run_history.run(self.args)
        prediction = json.loads((self.args.output / "prediction.json").read_bytes())
        self.assertEqual(prediction["finish_reason"], "length")
        self.assertEqual(self.requests[0]["max_tokens"], 96)
        self.assertFalse((self.args.output / "running.lock").exists())

    def test_failed_init_state_preserved_and_not_reinitialized(self):
        self.fail_init = True
        with self.assertRaisesRegex(RuntimeError, "native init failed"):
            run_history.run(self.args)
        self.assertEqual((self.args.output / "memory/native_state.txt").read_text(), "preserve this")
        with self.assertRaisesRegex(RuntimeError, "Prior build preserved"):
            run_history.run(self.args)
        self.assertEqual(self.events.count("init"), 1)
        self.assertFalse((self.args.output / "running.lock").exists())

    def test_reader_transport_retry_does_not_reopen_or_retrieve_native_memory(self):
        self.fail_reader = True
        with self.assertRaisesRegex(RuntimeError, "reader transport failed"):
            run_history.run(self.args)
        reader_bytes = (self.args.output / "reader_input.json").read_bytes()
        self.fail_reader = False
        run_history.run(self.args)
        self.assertEqual(self.events.count("init"), 1)
        self.assertEqual(self.events.count("retrieve"), 1)
        self.assertEqual(self.requests[0], self.requests[1])
        self.assertEqual((self.args.output / "reader_input.json").read_bytes(), reader_bytes)


if __name__ == "__main__":
    unittest.main()