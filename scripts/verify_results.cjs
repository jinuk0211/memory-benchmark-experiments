#!/usr/bin/env node
'use strict';

// Offline checks only. Run: node verify_results.cjs [repository-root]
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const root = path.resolve(process.argv[2] || process.cwd());
const report = {
  status: 'PASS',
  verified_at: new Date().toISOString(),
  network_calls: 0,
  scope: 'Recount preserved verdicts and stored per-question F1; no new model calls or F1 rescoring.',
  checks: {},
  files: {},
};
function read(file) {
  const bytes = fs.readFileSync(path.join(root, file));
  report.files[file] = {
    bytes: bytes.length,
    sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
  };
  return bytes.toString('utf8').replace(/^\uFEFF/, '');
}
const json = file => JSON.parse(read(file));
const jsonl = file => read(file).split(/\r?\n/).filter(line => line.trim()).map(JSON.parse);
const sameKeys = (actual, expected) => assert.deepEqual([...actual].sort(), [...expected].sort());
function close(actual, expected, label) {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected)
    && Math.abs(actual - expected) < 1e-9, label);
}
function check(name, fn) {
  try {
    report.checks[name] = { status: 'PASS', ...fn() };
  } catch (error) {
    report.status = 'FAIL';
    report.checks[name] = { status: 'FAIL', error: error.message };
  }
}
function verdicts(file, requireOk) {
  const receipts = jsonl(file);
  const byHash = new Map();
  for (const row of receipts) {
    assert.equal(row.returned_model, 'gpt-4o-2024-08-06');
    assert.equal(row.finish_reason, 'stop');
    assert.equal(typeof row.content, 'string');
    const text = row.content.trim().toLowerCase();
    assert.ok(['yes', 'no', 'yes.', 'no.'].includes(text), 'Unsupported raw verdict');
    assert.ok(typeof row.payload_hash === 'string' && !byHash.has(row.payload_hash),
      'Missing or duplicate receipt payload hash');
    const correct = text.includes('yes');
    if (requireOk) {
      assert.equal(row.status, 'ok');
      assert.equal(row.label, correct);
    }
    // Pilot finalization recovered valid period-terminated text from invalid_response receipts.
    byHash.set(row.payload_hash, { correct, content: row.content });
  }
  return byHash;
}
function verifySemantic(directory, filename, expected, totalRows, uniqueReceipts, strict) {
  const result = json(directory + '/' + filename);
  const receipts = verdicts(directory + '/judge_responses.jsonl', strict);
  assert.equal(result.judge_model, 'gpt-4o-2024-08-06');
  assert.equal(result.scored_rows.length, totalRows);
  assert.equal(receipts.size, uniqueReceipts);
  assert.equal(result.unique_judged, uniqueReceipts);
  assert.equal(result.unique_requests, uniqueReceipts);
  const groups = new Map();
  const used = new Set();
  for (const row of result.scored_rows) {
    const receipt = receipts.get(row.payload_hash);
    assert.ok(receipt, 'Scored row has no raw judge receipt');
    assert.equal(row.correct, receipt.correct);
    if ('raw_verdict' in row) assert.equal(row.raw_verdict, receipt.content);
    used.add(row.payload_hash);
    if (!groups.has(row.method)) groups.set(row.method, []);
    groups.get(row.method).push(row);
  }
  sameKeys(used, receipts.keys());
  sameKeys(groups.keys(), Object.keys(result.methods));
  for (const [method, rows] of groups) {
    assert.equal(new Set(rows.map(row => row.question_id)).size, rows.length,
      'Duplicate method/question ID');
    const summary = result.methods[method];
    assert.equal(summary.available, rows.length);
    assert.equal(summary.judged, rows.length);
    assert.equal(summary.correct, rows.filter(row => row.correct).length);
  }
  let firstIds;
  const verified = {};
  for (const [method, target] of Object.entries(expected)) {
    const rows = groups.get(method);
    assert.ok(rows, 'Missing expected method: ' + method);
    assert.equal(rows.length, target.n);
    assert.equal(rows.filter(row => row.correct).length, target.correct);
    const ids = rows.map(row => row.question_id).sort();
    if (firstIds) assert.deepEqual(ids, firstIds, 'Methods use different questions');
    firstIds = ids;
    const summary = result.methods[method];
    assert.equal(summary.expected, target.n);
    close(summary['accuracy_on_same' + target.n], target.correct / target.n, 'Accuracy mismatch');
    verified[method] = { n: rows.length, correct: target.correct };
  }
  if (strict) sameKeys(groups.keys(), Object.keys(expected));
  return { rows: totalRows, unique_receipts: receipts.size, methods: verified };
}

check('longmemeval_common50_gpt4o', () => verifySemantic(
  'outputs/longmemeval_common50_gpt4o_20260912', 'results.json',
  { full_context: { n: 50, correct: 46 }, langmem: { n: 50, correct: 37 },
    simplemem: { n: 50, correct: 38 }, lightmem: { n: 50, correct: 45 } },
  200, 182, true));
check('longmemeval_minilm_dev12_gpt4o', () => verifySemantic(
  'outputs/longmemeval_semantic_dev12_20260912', 'official_results.json',
  { minilm_seed: { n: 12, correct: 11 }, minilm_r40: { n: 12, correct: 7 },
    minilm_refined: { n: 12, correct: 6 }, minilm_seed_parent: { n: 12, correct: 11 } },
  135, 54, false));

check('locomo_gpt4omini', () => {
  const directory = 'outputs/locomo_gpt4omini_judge_20260911';
  const rows = jsonl(directory + '/scores.jsonl');
  const summary = json(directory + '/summary.json');
  const expected = { Naive: 1099, Mem0: 839, 'A-MEM': 799, LangMem: 644,
    SimpleMem: 842, LightMem_official: 1008, HiGMem: 860, 'E-Mem': 1188,
    s_parent_single_2000: 1120 };
  assert.equal(rows.length, 13860);
  assert.equal(summary.logical_rows, rows.length);
  assert.equal(summary.scored_rows, rows.length);
  assert.deepEqual(summary.returned_models, { 'gpt-4o-mini-2024-07-18': 10624 });
  const groups = new Map();
  const shared = new Map();
  for (const row of rows) {
    assert.ok(row.judge_label === 'CORRECT' || row.judge_label === 'WRONG');
    assert.equal(row.judge_score, row.judge_label === 'CORRECT' ? 1 : 0);
    if (shared.has(row.payload_hash)) assert.equal(shared.get(row.payload_hash), row.judge_score);
    shared.set(row.payload_hash, row.judge_score);
    if (!groups.has(row.method)) groups.set(row.method, []);
    groups.get(row.method).push(row);
  }
  sameKeys(groups.keys(), Object.keys(expected));
  sameKeys(Object.keys(summary.methods), Object.keys(expected));
  assert.equal(shared.size, 10624);
  assert.equal(summary.unique_payloads, shared.size);
  let firstIds;
  const methods = {};
  for (const [method, correct] of Object.entries(expected)) {
    const items = groups.get(method);
    const ids = items.map(row => row.conversation_id + ':' + row.qa_index).sort();
    assert.equal(items.length, 1540);
    assert.equal(new Set(ids).size, 1540);
    if (firstIds) assert.deepEqual(ids, firstIds);
    firstIds = ids;
    assert.equal(items.reduce((sum, row) => sum + row.judge_score, 0), correct);
    const saved = summary.methods[method];
    assert.equal(saved.expected, 1540);
    assert.equal(saved.scored, 1540);
    assert.equal(saved.correct, correct);
    close(saved.accuracy_pct, correct / 1540 * 100, 'LoCoMo accuracy mismatch');
    methods[method] = { n: 1540, correct };
  }
  return { rows: rows.length, unique_payloads: shared.size, methods };
});

function verifyAblation(directory, arms, n) {
  const scored = json(directory + '/results/scored_predictions.json');
  const summary = json(directory + '/results/RESULTS.json');
  sameKeys(Object.keys(scored), arms);
  sameKeys(Object.keys(summary.results), arms);
  let referenceIds;
  const results = {};
  for (const arm of arms) {
    const rows = scored[arm];
    assert.equal(rows.length, n);
    const ids = rows.map(row => row.id).sort();
    assert.equal(new Set(ids).size, n);
    if (referenceIds) assert.deepEqual(ids, referenceIds, 'Unpaired ablation questions');
    referenceIds = ids;
    for (const row of rows) {
      assert.equal(row.arm, arm);
      assert.ok(Number.isFinite(row.official_f1) && row.official_f1 >= 0 && row.official_f1 <= 1);
    }
    const f1 = rows.reduce((sum, row) => sum + row.official_f1, 0) / n * 100;
    const saved = summary.results[arm];
    assert.equal(saved.n, n);
    assert.equal(new Set(rows.map(row => row.conv_id)).size, 10);
    close(f1, saved.f1, 'Stored F1 aggregate mismatch: ' + arm);
    close(saved.delta_pp, f1 - summary.results.ours.f1, 'Ablation delta mismatch: ' + arm);
    results[arm] = { n, f1_percent: f1 };
  }
  return { rows: n * arms.length, arms: results };
}
check('locomo_ablation300', () => verifyAblation('experiments/locomo_ablation300_20260913',
  ['ours', 'no_cues', 'no_audit', 'no_temporal', 'no_binding', 'random_cues', 'payload_keys'], 300));
check('locomo_binding1540', () => verifyAblation('experiments/locomo_binding1540_20260913',
  ['ours', 'no_binding'], 1540));

// Quoted commas, escaped quotes and embedded CR/LF are fields, not record boundaries.
function csv(text) {
  const records = [];
  let fields = [], field = '', quoted = false, closed = false, active = false;
  function endField() { fields.push(field); field = ''; closed = false; }
  function endRecord() {
    endField();
    records.push(fields);
    fields = [];
    active = false;
  }
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') { quoted = false; closed = true; }
      else field += ch;
      continue;
    }
    if (ch === ',') { endField(); active = true; }
    else if (ch === '\r' || ch === '\n') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      endRecord();
    } else if (ch === '"') {
      assert.ok(field === '' && !closed, 'Malformed CSV quoting');
      quoted = true; active = true;
    } else {
      assert.ok(!closed, 'Unexpected text after CSV closing quote');
      field += ch; active = true;
    }
  }
  assert.ok(!quoted, 'Unterminated CSV quote');
  if (active || fields.length || field.length) endRecord();
  const header = records.shift();
  assert.ok(header && new Set(header).size === header.length, 'Invalid CSV header');
  return records.map(record => {
    assert.equal(record.length, header.length, 'CSV column count mismatch');
    return Object.fromEntries(header.map((key, index) => [key, record[index]]));
  });
}
check('locomo_consolidated_csv', () => {
  const directory = 'outputs/locomo_all_results_20260913';
  const history = csv(read(directory + '/HISTORY_129.csv'));
  const full = csv(read(directory + '/FULL1540.csv'));
  assert.equal(history.length, 129);
  assert.equal(new Set(history.map(row => row.configuration)).size, 129);
  assert.equal(full.length, 16);
  assert.equal(new Set(full.map(row => row.method)).size, 16);
  for (const row of full) {
    assert.equal(Number(row.n), 1540);
    assert.ok(Number.isFinite(Number(row.f1_percent)));
  }
  return { history_configurations: history.length, full1540_configurations: full.length };
});

fs.writeFileSync(path.join(root, 'VERIFICATION_EXPORT.json'), JSON.stringify(report, null, 2) + '\n');
for (const [name, result] of Object.entries(report.checks)) {
  console.log(result.status + ' ' + name + (result.error ? ': ' + result.error : ''));
}
console.log(report.status + ': VERIFICATION_EXPORT.json');
process.exitCode = report.status === 'PASS' ? 0 : 1;

