# LoCoMo 결과를 만든 실행 코드 — 2026-09-09

이 폴더는 아래 비교표를 분석하기 위한 **실제 실행 소스·설정·채점 코드의 보관본**입니다. 현재 개발 checkout을 실행 당시 코드로 간주하지 않도록 버전을 분리했습니다. 모델 추론을 새로 돌린 결과가 아닙니다.

## 최신 점수와 코드 위치

전체 1,540문항(category 1–4), 10개 대화의 공식 F1 ×100입니다. Qwen3.5-9B FP16과, 임베딩이 필요한 방법은 MiniLM-L6-v2를 사용했습니다. 방법별 프롬프트·검색 예산·실패 복구·실행 환경은 서로 다릅니다.

| 방법 | F1 | 먼저 볼 실행본 |
|---|---:|---|
| E-Mem | 56.96 | [r4 구축/실행](snapshots/server_snapshot/workspace/MemoryData-finish-r4), [r11 저장 메모리 복구](snapshots/server_snapshot/workspace/MemoryData-finish-r11) |
| 우리 Seed | 56.26 | [MiniLM 실행본](snapshots/recursive_minilm/run_transfer.py), [기억 구축](snapshots/recursive_minilm/source/refine.py) |
| 우리 Refined | 55.53 | [MiniLM 실행본](snapshots/recursive_minilm/run_transfer.py), [보강 로직](snapshots/recursive_minilm/source/parent_evidence.py) |
| 우리 r40 | 55.52 | [MiniLM 실행본](snapshots/recursive_minilm/run_transfer.py), [부모 레시피](snapshots/recursive_minilm/source/portable_parent.py) |
| Naive / Full-context | 55.17 | [r1 full_raw](snapshots/certmem_r1/scripts/run_system_v15.py), [r2 나머지 문항](snapshots/certmem_r2/scripts/run_system_v15.py) |
| LightMem 기존 pipeline | 46.60 | [공통 실행본](snapshots/primary_pinned/MemoryData), [최종 서버 코드](snapshots/server_snapshot/workspace/MemoryData) |
| 우리 CertMem v15 | 42.23 | [r1](snapshots/certmem_r1), [r2](snapshots/certmem_r2) |
| HiGMem | 40.33 | [과거 실행 소스](snapshots/higmem_historical/HiGMem), [후속 서버 코드](snapshots/server_snapshot/workspace/HiGMem) |
| LightMem direct | 38.20 | [direct 실행본](snapshots/server_snapshot/workspace/MemoryData-lightmem-direct-r1) |
| SimpleMem | 38.19 | [r13](snapshots/server_snapshot/workspace/MemoryData-finish-r13), [r17](snapshots/server_snapshot/workspace/MemoryData-finish-r17), [r21](snapshots/server_snapshot/workspace/MemoryData-simplemem-qa-repair-r21), [최종 합성/채점](snapshots/server_snapshot/workspace/MemoryData-simplemem-finalizer-r25b) |
| A-MEM | 36.85 | [r26 원본 실패 정책 복원](snapshots/server_snapshot/workspace/MemoryData-amem-original-policy-r26), [r27 후처리](snapshots/server_snapshot/workspace/MemoryData-amem-postrun-recovery-r27) |
| Mem0 | 36.50 | [r3](snapshots/server_snapshot/workspace/MemoryData-finish-r3), [r5](snapshots/server_snapshot/workspace/MemoryData-finish-r5) |
| LangMem | 29.93 | [공통 실행본](snapshots/primary_pinned/MemoryData), [복구/최종 채점](snapshots/server_snapshot/workspace/MemoryData/scripts) |

점수 원본은 [reports](reports)에 있습니다. [Seed/r40/Refined 점수](reports/seed_r40_refined_comparison.json), [CertMem/Naive 공식 점수](reports/certmem_official_scores.json), [전체 비교 기록과 한계](reports/comparison_record.md)를 함께 보세요. 비교 기록에는 과거 시점의 진행 상황도 남아 있으므로 최신 완료 항목을 우선합니다.

## Naive 55점부터 분석할 때

1. `certmem_r1/scripts/run_system_v15.py`의 `full_raw` 분기를 읽습니다. 세션별 원문 턴을 모두 연결해서 reader에 전달합니다. 질문마다 전체 원문을 주는 방식입니다.
2. `certmem_r1/scripts/run_read_v9.py`의 `READER_SYS`와 `reader_prompt`에서 최종 답변 프롬프트를 확인합니다. v15 reader의 출력 한도는 32토큰입니다.
3. 같은 v15 파일의 `certified` 경로와 원문 선택을 비교합니다. 질문의 gold/evidence metadata와 실제 모델에 들어간 prompt를 구별해야 합니다.
4. `certmem_r1/scripts/score_certmem_full.py`와 `recursive_minilm/score_refined_full.py`에서 공식 scorer 연결을 확인합니다.
5. Seed/r40/Refined는 `recursive_minilm/run_transfer.py` → `source/refine.py`, `source/portable_parent.py`, `source/parent_evidence.py` → `transfer_runtime.py` 순으로 읽습니다. Reader 기억 예산은 2,048토큰, 출력 한도는 96토큰입니다.

Naive와 CertMem은 r1 첫 885문항과 r2 나머지 655문항을 합친 결과입니다. r1의 37개 정책 평균이나 사후 최고 정책 점수가 아닙니다. Refined와 r40의 반올림 전 차이는 약 0.0054점으로, 유의한 개선이라고 단정할 수 없습니다.

## 보관본 구조와 실행 조건

- `snapshots/server_snapshot`: 최종 서버에서 보존한 버전별 코드. `workspace/<실행본>` 경로를 유지했습니다. 각 방법의 코드·프롬프트·런처·기존 테스트·계측/채점 도구를 포함합니다.
- `snapshots/server_snapshot/etc/supervisor/conf.d`: 실제 런처 명령과 환경변수. 자동 시작은 하지 마세요. 당시 서버 경로를 사용합니다.
- `snapshots/primary_pinned`: 초기 공통 baseline의 핀된 소스. 후속 복구는 해당 r버전 실행본과 함께 봅니다.
- `snapshots/higmem_historical`: HiGMem 과거 실행 소스. 현재 서버 사본과 차이가 있어 별도로 보존합니다.
- `snapshots/certmem_r1`, `snapshots/certmem_r2`: 두 번의 CertMem/Naive 실행에 배포한 코드 압축본에서 추출한 소스.
- `snapshots/recursive_minilm`: 최신 1,540문항 Seed/r40/Refined 실행 소스와 `launch_source.sha256`. `run_baseline.sh`, `PROTOCOL.md`, `prepare_protocol.py`에 실행 단계가 있습니다.
- `snapshots/legacy_qwen_embedding`: 이전 377문항 Qwen 임베딩 실험 관련 로컬 소스. 최신 MiniLM 점수의 실행본과 구별해야 합니다. 다른 모델/데이터셋 연구용 파일도 일부 포함되어 있으며, 전부가 LoCoMo 최종 실행에 사용된 것은 아닙니다.
- `snapshots/extra_variants`: 추가 변형의 백업에서 찾은 코드/설정.
- `SOURCE_MANIFEST.json`: 각 파일의 SHA-256, 원본 압축파일/멤버 또는 로컬 출처. 보관본의 코드를 고쳐서 새 구현으로 만들지 않았습니다.

이것은 **코드 분석용 아카이브**이며, 깨끗한 환경에서 한 명령으로 재현 검증한 패키지는 아닙니다. `/workspace/...`, `/venv/...` 같은 절대 경로는 원본 그대로입니다. 데이터·모델·의존성 설치 및 해당 경로 배치가 필요합니다. 원시 대화 데이터, 질문별 답변, 메모리 캐시, 모델 가중치, 가상환경, 접속 키는 이번 업로드에 넣지 않았습니다. 원래 해시 목록은 제외된 데이터도 참조하므로 데이터 없이 `sha256sum -c launch_source.sha256` 전체가 통과하지 않습니다.

공통 원본 데이터 SHA-256: `cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993`.
공식 scorer SHA-256: `8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd`.

`recursive_minilm/environment.json`의 packages는 원래 소스 환경의 기록이며 실제 target runtime을 뜻하지 않습니다. 최신 MiniLM 실행은 vLLM 0.19.1 / NLL chunked prefill 512를 사용했습니다. 모델 revision, launch 설정과 보고서를 함께 확인하세요.

토큰 총량의 집계 범위도 다릅니다. Seed/r40/Refined는 캐시를 공유했고, CertMem/Naive는 선택된 실행 경로 비용입니다. E-Mem과 HiGMem은 계측 한계가 남아 있습니다. 전체 코드를 제공한다는 사실이 baseline 구현의 공정성·통계적 유의성·독립 재현 성공을 보증하지는 않습니다.

## 라이선스

기존 upstream 코드의 저작권과 라이선스가 유지됩니다. 이 아카이브에 별도의 일괄 재라이선스를 적용하지 않습니다. 저장소의 원래 `methods/` 구현 및 보관본에 포함된 LICENSE/NOTICE도 확인하세요.
## 업로드 검증

[UPLOAD_VALIDATION.json](UPLOAD_VALIDATION.json)에 파일 검증 결과를 기록했습니다. 출처 파일 13,256개(서로 다른 내용 862개)의 해시가 일치합니다. MiniLM 실행 시점의 125개 핀 중 데이터셋 1개를 제외한 124개 파일을 모두 포함했고 해시가 일치합니다. `additional_pins`에는 다른 보고서가 참조하는 과거 코드도 해시로 찾아 보충했습니다.

`COPIED_SOURCE.json`이 참조하는 초기 복사 시점의 `runtime_meter.py`, `test_refined_runtime.py` 두 버전은 찾지 못했습니다. 최종 실행에 사용한 두 파일은 `recursive_minilm`에 있으며 `launch_source.sha256`와 일치합니다. 초기 복사 기록을 최종 실행 코드의 핀으로 해석하지 마세요.