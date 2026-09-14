# LongMemEval-S baseline 실행 기록

**STOP incident correction:** The assistant prematurely stopped still-needed instance 50468468 while awaiting source approval. The user then reported indefinite scheduling on restart. Vast officially documents this GPU-reassignment risk. That decision was incorrect. Source permissions are now granted, but GPU access must be recovered. Existing automatic STOP-on-failure code is on deployment hold; no further STOP/START requests will be issued as a recovery shortcut. Verified HiGMem/LightMem smoke and SimpleMem failure backups remain local.

**최신 사용자 승인:** 수정본 전송·적용 및 남은 실행 설정에 대한 승인을 받았습니다. 이전 전송 승인 차단은 해소됐습니다. 현재는 중지한 Vast 인스턴스 50468468의 재시작과 SSH 연결 확인이 필요합니다. 브라우저 제어 도구 시작 오류로 사용자에게 Vast Start 후 현재 SSH 명령 확인을 요청했습니다.

현재 이 작업은 **HiGMem·LightMem·SimpleMem 각 500문항**을 담당합니다. 완료된 결과를 검증하고 무료 보조 F1과 백업을 만든 뒤 Vast 인스턴스를 중지하는 것이 목표입니다.

**2026-09-10 16:17 UTC 확인:** HiGMem·LightMem은 전체 대화 사전 검증 각 1문항만 완료했습니다. 500문항 전체를 완료한 방법은 없습니다. SimpleMem v1은 실제 실행에서 계측 구간 중복 때문에 실패했습니다(11초, 논리 오류 45회, 실제 Qwen 요청 0건). 실패 기록과 원본 파일은 로컬까지 해시 검증해 보존했습니다. 원본 병렬 처리와 알고리즘을 유지하는 v2 계측 수정본은 독립 테스트 17개를 통과했습니다. 수정본 `simplemem_metering_fix_v2.tgz`(72,393바이트, 28파일, SHA256 8ed7785f7e5d97c75742d0dc13f16bbda79efec9844df098220db998209f2893)의 실제 전송이 자동 승인 검토에서 거절되어, 이 파일과 서버 목적지의 명시적 승인을 요청했습니다. 아직 v2를 전송·적용하지 않았습니다.

**비용 보호 조치(16:29:47 UTC):** 실행 중인 baseline이 없는 상태에서 Vast instance 50468468의 수동 STOP 요청이 HTTP200/success=true로 수락됐습니다. 결과 백업은 로컬 해시까지 확인했습니다. 실제 플랫폼 stopped 상태는 아직 독립 확인하지 못했으므로 요청 수락과 구분합니다. 전송 승인 후 Vast 플랫폼 계정에서 재시작해야 남은 실험을 진행할 수 있습니다.

무인 실행기와 자동 중지 감시기는 아직 서버에서 시작하지 않았습니다. 이전 `native3_autonomous_ops_v1.tgz` 전송은 자동 승인 검토에서 정확한 파일 묶음·목적지 승인이 필요하다는 이유로 거절됐습니다. 그 계획은 실패가 확인된 SimpleMem v1 경로를 사용하므로 그대로 실행하지 않습니다. v2 실제 검증 뒤 새 경로·해시를 반영한 계획이 필요합니다.

현재 상태의 기준은 `STATUS.json`, 담당 범위는 `METHOD_OWNERSHIP.json`입니다. 아래 내용에는 이전 준비·승인·실행 시점의 이력이 포함되어 있으며 현재 상태와 구분해야 합니다.

이 폴더는 전체 7개 baseline 비교의 실행 기록입니다. 최신 사용자 지정에 따라 **이 작업과 현재 GPU는 HiGMem·LightMem·SimpleMem만 각각 500문항(총 1,500답변)**을 담당합니다. E-Mem·A-MEM·Mem0·LangMem은 다른 GPU 작업 범위이며 이 작업에서 실행하지 않습니다. 질문을 250개씩 나누지 않습니다. 담당 기준은 `METHOD_OWNERSHIP.json`, 세 방법의 기존 native 명령 참조는 `THREE_METHOD_EXECUTION.json`입니다. 현재 서버는 `root@ssh1.vast.ai:28469`, 경로는 `/workspace/longmemeval_s_native7_20260910`입니다.

**최근 확인(2026-09-10 14:56 UTC): 현재 담당 중 HiGMem·LightMem의 동일 전체 이력 사전 실행 각 1문항을 검증·백업했습니다. HiGMem은 550턴 전부 처리해 종료 코드 0으로 끝났고, 원본 모델 호출 1,730회와 실제 요청 1,731회가 모두 정상 종료됐습니다. 답변은 원본 검색 결과에 따른 abstention이며 정답률을 검증했다는 뜻은 아닙니다. SimpleMem의 원본 LanceDB 환경은 검증됐고, native Dialogue 후보 코드의 전송·사전 실행이 남았습니다. full500 완료 방법은 0개, 유료 judge 호출은 0건입니다.** baseline 작업은 현재 실행 중이 아닙니다. 예전 `native7-launcher`·`native7-resume`·7개 전용 `run_all.py` queue를 다시 시작하지 않습니다.

**원본 충실도 복구:** 기존 4096자 통합 입력은 SimpleMem의 turn별 bulk API 및 Mem0 공식 예제의 메시지 두 개 입력과 다릅니다. A-MEM은 별도 agent library가 아니라 논문 재현 저장소 WujiangXu/A-mem을 기준으로 turn별 입력, 1000토큰, 질문 키워드 생성, k10 검색, 원래 QA를 복원한 별도 후보를 만들었습니다. 원본과 후보·기존 실패는 서로 다른 source/protocol로 보존합니다. LightMem의 user_only·원래 짝 처리 규칙으로 550개 source turn 중 546개가 native에 전달되고 4개가 제외됐으며, 이를 숨기지 않고 기록했습니다.

**현재 소스 준비 범위:** 최신 담당 지정에 맞춘 `native_recovery_simplemem_only_v1.tgz`(72,963바이트, SHA256 `30997c322916bfd3425a3c5ca0eca203449b059d2727f2537180646d546246ed`)의 독립 검토를 완료했습니다. 일반 파일 28개, runtime binding 44개이며 원본 18개와 동결 후보·검증기 26개는 그대로입니다. 이 정확한 묶음의 전송·적용 승인을 요청했고 아직 전송하지 않았습니다. 이전 `native_recovery_four_03.tgz`는 과거 제안으로 보존하며 현재 작업에서 전송하지 않습니다. 원본 runtime manifest는 18개 binding이며, 과거 기록의 161개는 원본 수치가 아니라 중간 제안 수치였습니다. 새 묶음은 원본 18개를 보존합니다. 앞선 A-MEM 소스 전송은 자동 승인 검토가 파일 묶음별 승인을 요구해 거절했으므로 새 묶음에도 해당 전송 제한을 적용합니다. 이미 검증한 `.venv-simplemem-native0253` 환경을 재사용하며 원본 location 배열 변환은 적용하지 않습니다.

**실제 환경에서 추가 확인한 장애(2026-09-10 11:49 UTC):** 공용 client의 LanceDB0.38은 원본 SimpleMem이 사용하는 Tantivy FTS를 제거하여 실제 인덱스 생성이 ValueError로 실패했습니다. 공식 LanceDB0.25.3을 쓰는 별도 Python3.11 SimpleMem 환경 설치와 실제 검증을 완료했습니다. 55개 공개 wheel 해시·의존성, 원본 Tantivy 의미/키워드/조건 검색과 reopen, 고정 MiniLM CPU FP32·384차원·256토큰 및 HTTP 수치 일치가 통과했습니다. 설치·검증 기록은 로컬 백업 SHA까지 확인했으며 baseline 후보 자체의 full-history smoke 통과를 뜻하지 않습니다. 복구 source와 최종 archive는 변경하지 않으며, 이후 새 실제 launch plan을 만들 때 SimpleMem의 Python 경로를 검증된 별도 환경으로 정정해야 합니다. 공용 환경의 pip check 통과는 이 기능 호환성을 증명하지 않습니다.

**Mem0 환경 검증 완료(2026-09-10 13:49 UTC):** 새 Python3.11.16 환경에 원본 OSS0.1.94와 고정 공개 wheel 62개를 설치했습니다. 원본 Qdrant1.19 adapter의 저장·검색·조건 필터·수정·재연결·삭제, 고정 Qwen 토크나이저, 로컬 MiniLM 384차원·256토큰 및 SDK/HTTP 오차 1e-6 이하, 원래 2000/.1/.1 설정의 로컬 Qwen SDK text/JSON 호출이 통과했습니다. 설치된 Mem0 Python93개+METADATA1개와 Qwen metadata9개를 원문 해시와 정확히 대조했습니다. 검증 증거 626,934바이트의 서버·로컬 SHA와 내부 파일 해시도 독립 확인했습니다. Memory.add나 후보 full-history smoke를 수행했다는 뜻은 아닙니다.

## 무엇을 같은 조건으로 맞추나요?

모든 방법은 **Qwen3.5-9B FP16**과 **MiniLM-L6-v2, 384차원, 원래 입력 한도 256토큰**을 사용합니다. 데이터는 같은 LongMemEval-S cleaned 500문항이며, 원래 질문 ID와 답변 불가능 문항도 유지합니다.

메모리 생성·갱신·검색·내부 추론·답변 생성은 각 방법의 구현을 유지합니다. 샘플링 설정, 검색 개수, 답변 토큰 한도 등은 해당 방법의 참조 설정을 따릅니다. **공통 답변 생성기나 공통 reader 예산으로 바꾸지 않으므로, 같은 모델을 쓴다는 것이 방법별 계산량까지 같다는 뜻은 아닙니다.**

| 방법 | 현재 실행 기록과 준비 코드 | 원본 처리와 연동 범위 |
|---|---|---|
| E-Mem | `native_five.py --method e_mem`; shared smoke 검증 완료 | `EMemAdapter.manager.chat`의 도구 호출, block 추론, 집계, 최종 QA 유지 |
| SimpleMem | 기존 실패 보존; `official_recovery/simplemem_native_dialogues_v1/runner.py` 로컬 검토 완료·미전송 | 실제 공식 source의 turn별 Dialogue를 bulk add_dialogues로 전달. window 40/overlap 2, 병렬 처리 및 native ask 사용. location 배열 변환 미적용 |
| LangMem | 기존 시도 중단·보존; `official_recovery/langmem_native_sessions_v1/runner.py` 로컬 검토 완료·미전송 | 공식 memory manager에 역할·내용·순서를 보존한 세션 messages 전달. 날짜 전달과 benchmark QA는 LongMemEval 연결 정책으로 명시 |
| Mem0 | 기존 실패 보존; `official_recovery/mem0_native_pairs_v1/runner.py` 로컬 검토 완료·미전송 | 공식 OSS v0.1.94 core에 세션 내 원래 발화 두 개씩 전달. QA는 기존 benchmark 연동이며 논문의 Cloud backend와 같다고 주장하지 않음 |
| A-MEM | 기존 실패 보존; `official_recovery/a_mem_paper_v1/runner.py` 로컬 검토 완료·미전송 | 논문 재현 저장소 WujiangXu/A-mem의 turn별 메모리, 1000토큰, query keyword 생성, k10 검색 및 native QA 사용 |
| LightMem | `native_lightmem.py`; shared smoke 검증 완료 | 공식 LongMemEval 짝 처리, user_only, LLMLingua2 압축, 추출·갱신·검색·reader 유지. native에서 제외된 4개 turn을 기록 |
| HiGMem | `native_higmem.py`; shared smoke 검증·백업 완료 | 공개 README의 paper 설정: no-profile, event-metadata-only, no-link, event top-k 10. query rewriting, 계층 검색, LLM 필터, 최종 JSON 답변 유지 |

`official_recovery`의 새 버전은 검증 완료된 기존 결과와 섞지 않습니다. 새 source·환경·runtime 검증 및 동일한 전체 이력 smoke를 통과해야 실제 full500 실행에 사용합니다. Mem0·LangMem의 benchmark QA와 각 방법의 LongMemEval 날짜·입력 변환은 논문 자체의 기본값과 구별해 기록합니다.

HiGMem의 위 옵션 이름에 `ablation`이 들어가더라도, 이 공개 버전의 README가 논문 재현 설정으로 지정한 조합입니다. 임의로 모든 기능을 켜지 않습니다. LongMemEval에는 LoCoMo category가 없으므로 원래 일반 질문용 prompt를 쓰고 질문 날짜를 전달합니다.

LightMem에는 고정된 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` 압축 모델을 추가로 사용합니다. 전체 source를 입력으로 제공하되, 공식 짝 처리 규칙 때문에 native API에 전달되지 않은 turn 수는 관측값으로 기록합니다.

## 실행과 검증 파일

- [launch_native7.sh](launch_native7.sh): 과거 7개 queue 시작점입니다. 현재 3개 담당 작업에서 재실행하지 않습니다.
- [run_all.py](run_all.py): 과거 7개 전용 queue입니다. 현재 담당 3개 실행에는 사용하지 않습니다.
- [launch_plan.template.json](launch_plan.template.json): 검증 receipt와 실제 경로를 넣기 전의 실행 계획입니다. 검증 후 생성되는 `launch_plan.json`이 실제 입력입니다.
- [model_integrity.json](model_integrity.json): 고정 model revision의 weight·설정·tokenizer 해시와 HF 출처입니다. 기대 해시가 있다는 사실만으로 서버 파일이 검증된 것은 아닙니다.
- [verify_runtime.py](verify_runtime.py): 서버에서 모델·runtime 파일과 `source/source_snapshot.json`에 기록된 코드 해시, CUDA 및 실제 endpoint를 확인하고 `runtime_receipt.json`을 만듭니다.
- [export_official.py](export_official.py): 500개 유효 예측과 완료·출처를 확인한 뒤 공식 평가기용 JSONL을 내보냅니다. 이 파일 자체는 judge를 호출하지 않습니다.

모델 준비 파일, source snapshot, CPU 테스트 통과는 실행 준비의 근거입니다. **이 작업의 HiGMem·LightMem·SimpleMem 각각 500개, 총 1,500개 질문 ID와 유효한 답변·실패 목록·provenance를 확인한 뒤에만 담당 full500 생성을 완료로 표시합니다. 전체 7개 연구의 완료 여부는 다른 GPU의 4개 결과까지 별도로 확인해야 합니다.** 실패·빈 답변·누락을 성공으로 세지 않으며, 실패 artifact와 이전 attempt는 보존합니다.

메모리 builder에는 source session ID·날짜와 각 turn의 `role/content`만 전달합니다. 정답, 질문 유형, evidence 표시와 `has_answer` 같은 turn annotation을 제외합니다. 평가 질문과 질문 날짜는 답변 단계에서 사용합니다.

## 점수와 API 비용

답변 생성 완료와 **공식 LongMemEval judge 평가 완료는 별개**입니다. 사용자의 API 비용 우려로 공식 judge 실행은 보류 중이며, 현재 공식 accuracy 점수는 없습니다.

실제 OpenAI API 키를 입력·저장·사용한 건수는 **0건**입니다. 실행 스크립트의 `EMPTY`는 로컬 OpenAI 호환 endpoint용 문자열입니다. 공식 judge API는 호출하지 않았습니다.

저장된 답변으로 **추가 API 비용 없이 로컬 token F1을 보조 계산할 수 있습니다.** 이 값은 공식 LongMemEval accuracy가 아닙니다. 로컬 평가에 API 비용이 없다는 설명은 GPU 서버 임대비까지 없다는 뜻은 아닙니다.

## 이전 계획과의 관계

아래 시각별 기록은 당시 상태를 보존한 이력입니다. 현재 담당과 실행 상태는 문서 상단 및 STATUS.json을 따릅니다.

`experiments/longmemeval_qwen35_20260908/PLAN.md`와 이전 5-method controlled-reader 준비물은 **이 작업의 실행 계획으로 대체되었습니다(superseded)**. 과거 준비·일부 실험 기록으로 보존하며, 이번 native 7-method full500 실행이나 완료 결과로 합산하지 않습니다.

고정 데이터 SHA-256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`

전송 승인 대기: `runners_bundle.tgz` (30,939 bytes, SHA-256 `59a6f7707f6c1b7a7a7e295de86c61376e055f77b910d397784ccdf9b00634f9`)에는 실행기·설정·공식 평가 코드만 있으며 API 키 검사 결과 0건입니다. 2026-09-10 자동 승인 검토가 정확한 payload의 외부 반출 승인 필요를 이유로 지정 서버 전송을 거절하여 이 묶음은 아직 전송되지 않았습니다. GPU 설치 작업은 계속되지만 native7 launcher와 baseline 추론은 시작되지 않았습니다.
목표 재개 후 갱신: 사용자 목표 파일의 추가 승인으로 전송이 허용되었습니다. LICENSE 보완을 포함한 `runners_bundle.tgz` (77,337 bytes, SHA256 `e590265b4246a054c56210649b6e8054603d51b79b2fd1a89056b0a21ee3258b`)의 서버 해시 일치를 확인하고 적용했습니다. LangMem 설치를 재개했고 `native7-launcher`를 Supervisor로 시작했습니다. launcher 시작은 실험 완료를 뜻하지 않습니다. 실제 검증과 shared smoke를 모두 통과한 뒤 full500으로 넘어갑니다. 위 전송 승인 대기 기록은 이전 상태입니다.
2026-09-10 08:49 UTC 갱신: 사용자가 `startup_fix_01.tgz` (6,073 bytes, SHA256 `7e9381f2e8ccd7914a07140f81c933b45f77ee71ef5760faf28a325e9526a119`)의 전송·적용을 명시적으로 승인했습니다. 서버 해시 일치를 확인하고 기존 세 파일을 `startup_fix_01.before.tgz`에 보존한 뒤 적용했습니다. Qwen GPU 할당을 0.78로 조정하고 FP16·65,536 context를 유지했으며, 외부 OpenRouter 환경 우회와 localhost proxy 경유를 차단했습니다. Supervisor launcher와 Qwen이 시작됐고 실제 endpoint 검증은 진행 중입니다. LightMem 설치는 기존 프로세스를 유지합니다. baseline 추론 및 유료 judge는 아직 시작되지 않았습니다.
2026-09-10 09:02 UTC 갱신: Qwen의 실제 답변 생성·스트리밍·도구 호출과 LightMem Python3.11/Torch2.8의 native import·실제 GPU FP16 연산이 통과했습니다. NLTK3.10.3이 proxy 다운로드를 거부하는 문제는 다운로드 자식 프로세스만 proxy 환경을 제거하고 `/root/.cache/longmemeval_s_native7_20260910/nltk_data`의 private cache를 쓰도록 해결했습니다. NLTK 보안 검사와 원본 method는 유지했습니다. `nltk_fix_02.tgz` (5,857 bytes, SHA256 `d403e3f2fafa8122c7d7536243c51534b396b75fadab92e2ccb85f0b69081462`)를 해시 확인·이전 파일 보존 후 적용했습니다. 전체 runtime 검증이 통과해 서버에 `runtime_receipt.json`, `launch_plan.json`이 생성되었고 shared full-history native smoke가 시작됐습니다. 3,500개 생성과 평가 완료를 뜻하지 않습니다.
2026-09-10 09:39 UTC 복구 기록: E-Mem 실제 산출물 독립 감사 26개 항목이 통과했습니다(53개 세션,550개 발화,170개 입력 청크 보존,has_answer12개 제거,코드48개 해시 일치). SimpleMem은 원본 파서와 동일한 location:str|None 규칙에 모델이 list를 반환하여 각3회 내부 재시도 후 실패했고, 동일 조건의 새 history attempt에서도 재현됐습니다. 두 실패는 `remote_verification/smoke_simplemem_failed_attempts12_20260910T0936Z.tgz`에 서버·로컬 동일 SHA로 보존했습니다. 원본과 다른 location 목록 정규화 후보는 별도 폴더에서 테스트8개+subtest13개를 통과했지만 **미적용**이며, 원본 유지와 별도 호환 변형 허용 중 사용자 선택이 대기 중입니다. 후보는 symbolic 검색·reader 입력에 영향을 줄 수 있어 원본과 동일하다고 보고하지 않습니다.

현재 `native7-launcher`, `native7-resume`은 SimpleMem 실패로 종료됐고, **`native7-smoke-langmem`(PID10696)이 기존 launch_plan의 LangMem 명령과 같은 smoke ID로 실행 중**입니다. 원본 queue/status.json은 앞선 SimpleMem 실패를 나타내므로 전체 작업이 멈췄다고 해석하면 안 됩니다. 실제 후속 smoke 증거는 `queue/manual_smoke_langmem_20260910T0936Z.json`, 해당 로그와 runs/langmem 아래 기록입니다. 이 독립 smoke가 실행 중일 때 native7-resume을 중복 시작하지 않습니다. 전체7개 shared smoke를 통과하기 전 full500을 시작하지 않습니다.