"""The comparison score must use every native row and the original QA metadata."""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def scorer_module():
    path = Path(__file__).parents[1] / "scripts" / "score_certmem_full.py"
    spec = importlib.util.spec_from_file_location("certmem_score_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(module):
    data = [{"sample_id": "conv-test", "qa": [
        {"question": "same?", "answer": "a", "category": 1, "evidence": []},
        {"question": "adversarial", "answer": "x", "category": 5, "evidence": []},
        {"question": "same?", "answer": 42, "category": 3, "evidence": ["D1:1"]},
    ]}]
    rows = []
    for config, policy in sorted(module.POLICIES):
        for ordinal, qa_index in enumerate((0, 2)):
            qa = data[0]["qa"][qa_index]
            rows.append({"conv_id": "conv-test", "config": config, "policy": policy,
                         "question_ordinal": str(ordinal), "question": qa["question"],
                         "gold": str(qa["answer"]), "category": str(qa["category"]),
                         "pred": "output", "f1": "0.2", "lenient": "1"})
    return data, rows


def test_official_score_is_separate_and_uses_original_indices():
    module = scorer_module()
    data, rows = fixture(module)

    def score(records):
        assert [r["index"] for r in records] == [0, 2]
        assert records[1]["answer"] == 42
        assert records[1]["evidence"] == ["D1:1"]
        return [0.5, 1.0]

    result = module.build_report(data, rows, score)
    assert result["predictions_complete"] is True
    assert result["reader_outputs"] == 74
    assert len(result["by_policy"]) == 37
    assert all(r["official_f1"] == 0.75 for r in result["by_policy"])
    assert all(r["native_normalized_f1"] == 0.2 for r in result["by_policy"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "gold", "question", "category", "ordinal", "policy", "nan"])
def test_invalid_rows_cannot_be_reported_complete(mutation):
    module = scorer_module()
    data, rows = fixture(module)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        field = {"ordinal": "question_ordinal", "nan": "f1"}.get(mutation, mutation)
        rows[0][field] = "nan" if mutation == "nan" else "wrong"
    with pytest.raises(ValueError):
        module.build_report(data, rows, lambda records: [1.0] * len(records))


def test_empty_native_answer_is_counted_not_dropped():
    module = scorer_module()
    data, rows = fixture(module)
    rows[0]["pred"] = ""
    result = module.build_report(data, rows, lambda records: [0.0] * len(records))
    assert result["empty_predictions"] == 1
    assert result["reader_outputs"] == 74


@pytest.mark.parametrize("scores", [[1.0], [float("nan"), 1], [2, 0]])
def test_invalid_scorer_output_rejected(scores):
    module = scorer_module()
    data, rows = fixture(module)
    with pytest.raises(ValueError):
        module.build_report(data, rows, lambda records: scores)


def test_offline_cli_pins_inputs_and_preserves_existing_output(monkeypatch, tmp_path):
    module = scorer_module()
    data, rows = fixture(module)
    dataset, items = tmp_path / "dataset.json", tmp_path / "items.csv"
    scorer, output = tmp_path / "official.py", tmp_path / "report.json"
    dataset.write_text(json.dumps(data), encoding="utf-8")
    with items.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scorer.write_text("def eval_question_answering(rows, eval_key, metric):\n    return [0.5] * len(rows), None, None\n", encoding="utf-8")
    monkeypatch.setattr(module, "DATASET_SHA256", hashlib.sha256(dataset.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "SCORER_SHA256", hashlib.sha256(scorer.read_bytes()).hexdigest())
    monkeypatch.setattr(sys, "argv", ["score", "--dataset", str(dataset), "--items", str(items),
                                      "--scorer", str(scorer), "--output", str(output)])
    module.main()
    report = json.loads(output.read_text())
    assert report["predictions_complete"] is True
    assert report["reader_outputs"] == 74
    assert report["sources_sha256"]["items"] == hashlib.sha256(items.read_bytes()).hexdigest()
    before = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        module.main()
    assert output.read_bytes() == before
    monkeypatch.setattr(sys, "argv", ["score", "--dataset", str(dataset), "--items", str(items),
                                      "--scorer", str(scorer), "--output", str(tmp_path / "new.json")])
    monkeypatch.setattr(module, "SCORER_SHA256", "wrong")
    with pytest.raises(ValueError, match="scorer hash"):
        module.main()
    monkeypatch.setattr(module, "DATASET_SHA256", "wrong")
    with pytest.raises(ValueError, match="dataset hash"):
        module.main()
