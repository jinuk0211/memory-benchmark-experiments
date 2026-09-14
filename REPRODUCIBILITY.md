# 재현 및 코드 지도

## 오프라인 결과 확인

저장된 답변과 판정을 확인하는 데 GPU나 API 키는 필요하지 않습니다. Node.js 20 이상에서 저장소 루트 기준:

```sh
node scripts/verify_results.cjs .
```

`VERIFICATION_EXPORT.json`을 생성합니다. LongMemEval 원시 GPT-4o 응답, LoCoMo 원시 GPT-4o-mini 응답, ablation 질문별 F1 평균과 표, 129개 이력·16개 설정의 행 수를 대조합니다. API 요청을 보내지 않습니다.

Python 3.11 이상에서 ablation의 정답·공식 F1·메모리 잠금·generation 영수증까지 다시 확인하려면:

```sh
python -m pip install -r requirements-analysis.txt
python experiments/locomo_ablation300_20260913/report_results.py --out reproduced/ablation300
python experiments/locomo_binding1540_20260913/report_results.py --out reproduced/binding1540
```

300문항은 7조건×300=2,100개, 전체 비교는 2조건×1,540=3,080개 답변을 검증합니다. 후자의 참조 generation 파일 3,058개를 포함했습니다. 저장된 결과를 덮어쓰지 않도록 별도 출력 경로를 지정합니다. 포장 시 두 명령을 실제로 실행해 원본 결과와 일치함을 확인했습니다.

기존 15설정×1,540문항의 원문 답변은 `outputs/baseline_drive_export_20260909/Analysis_bundle.zip`에 있습니다. `outputs/locomo_comparability_audit_20260911/audit_scores.py`는 포함된 canonical dataset 및 `HiGMem/official_locomo_evaluation.py`로 공식 F1을 다시 계산합니다. 이 스크립트는 인자 없이 실행하며 같은 폴더의 `scores_audit.json`을 재생성하므로 새 clone/작업 복사본에서 실행하세요. 포장 때 별도 출력 위치에서 23,100개 방법×문항 재계산을 수행했습니다.

## 코드 진입점

| 범위 | 실행·분석 코드 |
|---|---|
| 실행 당시 LoCoMo baseline | [버전별 코드 아카이브](code/locomo_execution_archive/README.md) |
| Seed / r40 / Refined | [run_transfer.py](experiments/recursive_minilm_20260909/run_transfer.py), [refine.py](experiments/recursive_minilm_20260909/source/refine.py), [parent_evidence.py](experiments/recursive_minilm_20260909/source/parent_evidence.py) |
| 과거 dev70 자기개선 | [source](generalization_20260908/source), [129 구성 감사](generalization_20260908/modern/REFINEMENT_HISTORY_COUNT.md) |
| Recursive v1/v2 | [recursive_minilm](experiments/recursive_minilm_20260909) |
| Qwen/Gemma 전이 | [modern/run_transfer.py](generalization_20260908/modern/run_transfer.py), [동결 소스](generalization_20260908/frozen_source) |
| Seed-parent | [seed_parent_ablation](migration_20260910/seed_parent_ablation) |
| Packed-marginal | [pilot](migration_20260910/packed_marginal_pilot_r1), [결과](migration_20260910/packed_marginal_results_r1) |
| LongMemEval 우리 방법 | [paper_lme_adapter](migration_20260910/paper_lme_adapter), [date adapter](migration_20260910/paper_lme_date_adapter) |
| LongMemEval baseline 생성 | [native7](experiments/longmemeval_s_native7_20260910), [native3](experiments/longmemeval_s_native3_5090_20260910), [초기 Qwen 실행](experiments/longmemeval_qwen35_20260908) |
| LongMemEval GPT-4o 채점 | [common50](outputs/longmemeval_common50_gpt4o_20260912), [기존 dev12](outputs/longmemeval_semantic_dev12_20260912) |
| 300문항 ablation | [잠긴 실행본](experiments/locomo_ablation300_20260913/package/code), [입력](experiments/locomo_ablation300_20260913/package/input), [프로토콜](experiments/locomo_ablation300_20260913/PROTOCOL.md) |
| 1540문항 binding | [잠긴 실행본](experiments/locomo_binding1540_20260913/package/code), [입력](experiments/locomo_binding1540_20260913/package/input) |

## GPU로 새 답변 생성

이 저장소는 여러 시점의 실제 실행본을 보존합니다. 모든 실험을 한 환경에서 실행하는 통합 설치 패키지는 아닙니다. 각 run의 모델 revision, 임베딩, 프롬프트, 예산, source SHA256 및 requirements를 사용하세요. 구형 후보의 vLLM API와 최신 실행본의 API는 다를 수 있습니다.

대표 저장 환경은 `torch 2.10.0+cu129`, `transformers 5.5.3`, `sentence-transformers 5.2.0`, `vllm 0.19.1`, `rank-bm25 0.2.2`입니다. 이 값은 모든 이력에 적용되는 단일 환경을 뜻하지 않습니다. 실험별 환경·의존성 기록을 우선하세요. GPU 생성에는 모델 가중치와 충분한 VRAM이 필요하고 포함돼 있지 않습니다.

두 ablation은 원래 원격 `/workspace/<experiment>/{code,input,run}` 구조였습니다. 저장소의 `package/code`와 `package/input`이 그 동결 입력입니다. 예를 들어 300문항 코드를 검증하고 새 출력에 protocol을 생성하는 명령은 다음과 같습니다.

```sh
python experiments/locomo_ablation300_20260913/package/code/run_ablation.py freeze --shard 0 --out reproduced/new_ablation300
```

이후 같은 `--out`으로 `prepare`, `evaluate` 단계를 각각 `--shard 0`, `--shard 1`에서 실행합니다. 실제 추론 환경과 모델 경로는 보존된 `run_gpu.sh`와 vendor runtime을 기준으로 구성해야 합니다. 입력을 새로 만드는 `prepare_inputs.py`는 이전 작업폴더를 참조하므로, 이 아카이브에서는 이미 동결한 `package/input` 재사용이 출발점입니다. 위 설명은 GPU 전 과정의 새 독립 재현 완료를 주장하지 않습니다.

LongMemEval `prepare_inputs.py`에는 과거 `D:/MemoryData` 절대경로와 원본 데이터 해시가 있습니다. 현재 보존된 `inputs.json`, `requests.json`, `selection.json`, 원시 judge 응답을 이용하면 기존 결과를 검증할 수 있습니다. 전체 cleaned 500문항 데이터 및 모든 중간 생성 캐시는 포함하지 않았습니다. 새 judge 호출에는 별도 API 키와 비용이 필요합니다.

## 보존 규칙

- `MANIFEST.json`의 `sha256`은 업로드 파일, `source_sha256`은 원본 파일 해시입니다. Markdown 로컬 링크를 고친 파일만 두 값이 다를 수 있습니다.
- `.gitattributes`의 `-text`는 입력·소스 줄바꿈 자동 변환을 막습니다. 동결 manifest가 코드와 입력 바이트를 검증하기 때문입니다.
- 과거 코드 아카이브의 `SOURCE_MANIFEST.json`, `UPLOAD_VALIDATION.json` 등은 당시 전체 수집본에 대한 기록입니다. 이 선별 저장소의 포함 파일은 루트 `MANIFEST.json`을 기준으로 합니다.
- 로컬에서 없는 과거 원격 원본은 경로·해시·점수 기록만 보존합니다. 모든 실패 실행이나 원격 캐시를 복원했다는 뜻은 아닙니다.
- 공개용 서비스, 모델 가중치, 가상환경, 접속 키, GPU 임대·서버 운영 폴더는 이 저장소의 범위가 아닙니다.
