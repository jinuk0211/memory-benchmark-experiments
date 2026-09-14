> 정정: 이 문서의 F1는 정답률이 아닙니다. 공식 의미 채점 준비와 현재 미실행 상태는 [재채점 문서](../longmemeval_semantic_dev12_20260912/README_KO.md)를 참조하세요.

**LongMemEval 예전 DEV12와 Full-context·LangMem·SimpleMem·LightMem 비교**

2026-09-12 보존 스냅샷과 기존 자체 방법의 예측·계측 기록을 대조했다. 새 LLM/GPU 추론과 공식 judge 호출 없이, 기존 답변의 진단 F1만 CPU로 계산했다. 원본 결과·실험 코드는 변경하지 않았다.

**비교 가능한 범위**

사용자 표의 원본은 `efficiency_summary.json` (historical source: `backups/50577514/metrics_partial_20260912_1445/efficiency_summary.json`), 문항별 원본은 `efficiency_per_sample.csv` (historical source: `backups/50577514/metrics_partial_20260912_1445/efficiency_per_sample.csv`)다. 현재 서버의 실시간 상태가 아니라 해당 스냅샷의 집계다.

| 비교 방법 | 사용자 표의 완료·시간 집계 | 예전 12문항과 겹침 |
|---|---:|---:|
| Full-context | 497답변 + 승인된 스킵3 | 12/12 |
| LangMem | 시간·사용량 갖춘104 | 1/12 |
| SimpleMem | 시간·사용량 갖춘52 | 1/12 |
| LightMem | 구축·답변 갖춘111 | 1/12 |

세 메모리 baseline 모두 공통 문항은 118b2229 하나다. LangMem 답변은 이후 112개 체크포인트에서 확인했으며 비용 표의 104개 집계와 구별한다. 서로 다른 문항 수의 총 토큰이나 서로 다른 부분집합의 점수로 전체 방법 순위를 매기지 않는다.

**사용자 표에 자체 방법을 추가한 효율 비교**

LLM 토큰은 입력+출력이다. 임베딩 토큰은 별도이며 아래 LLM 합계에 합산하지 않는다. 시간은 서로 다른 하드웨어·큐·캐시·측정 경계를 가진 관측값이다.

| 방법 | 집계 문항 | 평균 구축 wall 초 | 평균 질의 초 | 보고된 누적 LLM 토큰 M | 토큰 범위 |
|---|---:|---:|---:|---:|---|
| Full-context | 497답변/500계상 | 0 | 33.5322 | 63.821008 | 스킵·재시도 포함 관측분; 사용량 누락3호출 |
| LangMem | 시간104, 비용 행108 | 420.1187 | 0.4421 | 27.917618 | 해당 비용 스냅샷의 보고된 호출 전체 |
| SimpleMem | 시간52, 비용 행53 | 1767.6670 | 36.9526 | 12.536006 | 사용량 미기록503호출이 있어 하한 |
| LightMem | 111 | 816.1129 | 4.0292 | 6.230967 | 사용량 미기록2호출이 있어 하한 |
| 자체 Seed, 9/10 MiniLM | 12 | 미계측 | 7.7903 | 3.374134 | 공통 기본 구축을 포함해 귀속한 비용 |
| 자체 r40, 9/10 MiniLM | 12 | 미계측 | 4.8015 | 3.374111 | 같은 공통 구축을 귀속한 비용 |
| 자체 Paper Refined | 12 | 미계측 | 4.1342 | 15.543197 | 기본 구축+source probe+utility+새 reader 귀속 |
| 자체 Seed-parent | 12 | 미계측 | 7.5111 | 15.549853 | 가져온 source 구축 비용+신규 reader를 귀속한 계산 |

M=100만. 자체 방법 네 행은 독립적으로 모두 처음부터 다시 실행한 cold-run 비용이 아니다. 공통 작업을 각 방법에 귀속한 것이므로 네 행을 합산하면 공통 비용을 중복 계산한다. 원래 세 arm을 함께 실행한 실제 LLM 총량은 15.595792M이고, Seed-parent는 기존 구축 결과를 가져온 뒤 새 reader0.026305M를 소비했다.

단순히 총 토큰을 완료104/52로 나누면 아직 완료되지 않은 행의 작업과 분모가 섞일 수 있다. Full-context·SimpleMem·LightMem의 미기록 호출은 소비0으로 처리하지 않는다. 해당 missing 호출 수를 모두 GPU 오류 횟수라고 해석하지 않는다.

자체 질의시간은 evaluate_sample 범위로 검색·임베딩·packing·생성·계측·진단F1을 포함하지만 모델 초기화와 외부 저장을 제외한다. baseline 질의시간은 자체 QA wall 기록이며 LightMem은 prediction 파일 mtime−construction 파일 mtime로 근사했다. 공유 GPU 큐 대기가 wall에 들어갈 수 있다. Refined는 12답변 중 새 native 생성9회라 캐시 영향이 있다.

`자체 12문항 비용 보고` (historical source: `migration_20260910/pilot_runs/paper_frozen_lme12_qwen35_minilm_date_v2_r1/RESULT_AND_COST_STATUS.md`) · `Seed-parent 비용` (historical source: `migration_20260910/pilot_runs/seed_parent_lme12_qwen35_minilm_r1/collected/reports/NATIVE_USAGE.json`)

**완전히 같은 12문항: Full-context와 자체 방법**

모두 Qwen3.5-9B이며 자체 네 방법은 9/10 MiniLM/date-v2 조건이다. Full-context는 embedding 없이 전체 원문을 입력한다. 데이터 SHA256이 동일하고 Full-context 요청 12개에 기존 질문 문자열이 그대로 포함됨을 확인했다. 다만 프롬프트·출력 상한·GPU 병렬화·캐시는 통제되지 않았다.

| 방법 | 진단 F1×100 | 평균 질의 초 | 신규 reader LLM 12문항 M | 구축 포함 귀속 LLM M | 구축 포함 LLM/문항 |
|---|---:|---:|---:|---:|---:|
| Full-context | 13.7840 | 37.6962 | 1.504181 | 1.504181 | 125,348 |
| Seed | 70.7872 | 7.7903 | 0.026309 | 3.374134 | 281,178 |
| r40 | 44.8447 | 4.8015 | 0.026286 | 3.374111 | 281,176 |
| Paper Refined | 44.8447 | 4.1342 | 0.019649 | 15.543197 | 1,295,266 |
| Seed-parent | 70.7872 | 7.5111 | 0.026305 | 15.549853 | 1,295,821 |

Full-context의 37.70초는 같은12개 평균으로, 전체497개 평균33.53초와 다른 분모다. 공통12개는 사용량 미기록0이고, 1.504181M은 보존된 재시도를 포함한다. 최종 답변 호출만의 토큰은 1.388837M이다.

구축이 끝난 뒤 reader만 보면 Seed는 Full-context 관측 토큰의 약1.75%를 썼다. 그러나 각 이력에 질문 하나씩만 한 이번 비교에서는 기본 구축까지 귀속하면 Seed는 약2.24배, Paper Refined는 약10.33배의 LLM 토큰이다. 이는 해당 기록의 회계 비교이며 통제된 독립 실행의 효율 배수로 일반화하지 않는다.

**진단 F1를 공식 정확도로 읽으면 안 되는 이유**

이번 CPU 재채점은 기존 generic_f1를 그대로 사용했다: 소문자 ASCII 영숫자, a/an/the 제거, 중복 횟수를 포함한 토큰 겹침 F1. 공식 judge는 실행하지 않았다. Full-context 답변의 평균 정규화 단어 수는70.25, 정답은6.33이다. 긴 설명이 짧은답 기반 F1에 불리하며, abstention 여부나 의미상 정답을 충분히 평가하지 않는다.

Full-context는 context131072, truncation=false, summarization=false, 기본 출력256토큰이며 공통12 중 cc6d1ec1은 출력1024 재시도가 있다. 자체 방법은 read2048/answer96. 따라서 F1 13.78 대70.79를 일반 정확도의 확정적 차이로 해석하지 않는다. 기존12개는 노출된 DEV이며 새 일반화 holdout도 아니다.

[Full-context·LangMem 재채점 근거](baseline_matched_scores.json) · `기존 평가 함수` (historical source: `experiments/longmemeval_s_native7_20260910/score_diagnostic_f1.py:19`)

**네 baseline이 모두 겹치는 1문항**

118b2229: 출근 통근시간 질문. 정답은 편도45분이다. 아래는 1문항 관측이며 방법 전체의 성능 순위가 아니다.

| 방법 | 진단 F1×100 | 답변의 의미 | 구축 wall 초 | 질의 초 | 보고된 총 LLM 토큰 |
|---|---:|---|---:|---:|---:|
| Full-context | 6.25 | 45분 이야기를 언급하지만 다른 사람 정보라며 모른다고 답함 | 0 | 95.4526 | 113,669 |
| LangMem | 0 | Unknown | 196.8245 | 0.3065 | 233,872 |
| SimpleMem | 100 | 편도45분 | 3571.0404 | 44.5968 | ≥250,789 |
| LightMem | 19.0476 | 편도45분과 출처 설명 | 715.7630 | 2.4500 | 50,756 |
| 자체 Seed | 100 | 편도45분 | 미계측 | 7.4229 | 전체 구축 포함 문항별 비용 미집계 |
| 자체 r40 | 100 | 편도45분 | 미계측 | 4.6806 | 전체 구축 포함 문항별 비용 미집계 |
| 자체 Paper Refined | 100 | 편도45분 | 미계측 | 3.6052 | 전체 구축 포함 문항별 비용 미집계 |
| 자체 Seed-parent | 100 | 편도45분 | 미계측 | 7.3083 | 전체 구축 포함 문항별 비용 미집계 |

SimpleMem 해당 문항에는 사용량 미기록28호출이 있어 ≥ 표시다. Seed/r40의 이 문항 신규 reader 토큰은 각각2184/2174로 확인됐지만 전체 구축 포함 비용과 혼합하지 않았다. Paper는 r40와 같은 문맥의 캐시 답을 썼고 신규 generation은0회다.

LightMem과 SimpleMem은 같은 45분을 답했다. LightMem은 정규화 예측38토큰, 정답4토큰이라 F1=8/42=19.0476이 된다. 낮은 F1만으로 LightMem이 틀렸다고 할 수 없다. Full-context의6.25 역시 정답으로 인정된다는 의미가 아니라 단어 겹침이다.

SimpleMem/LightMem 출처는 `중간 백업 manifest` (historical source: `backups/50577514/intermediate_20260912_1000/backup_manifest.json`)의52/111개 예측이다. 보존 시각2026-09-12 09:58:40 KST, 압축324,499,258bytes, 로컬 SHA256 검증 기록이 있다.

**이전 9/8 Qwen/Gemma 결과까지 포함한 성능 기록**

모든 행은 같은12개 ID다. 9/8의 embedding은 Qwen3-Embedding-0.6B, 9/10은 MiniLM이며 Gemma는 모델도 다르다.

| 과거 실행 | n | 진단 F1×100 | 비용·시간 상태 |
|---|---:|---:|---|
| 9/8 Qwen+Qwen embedding · Seed | 12 | 56.8983 | 완전한 구축·실제 소비량·전체 질의시간 미확인 |
| 9/8 Qwen+Qwen embedding · r40 | 12 | 40.6781 | 완전한 구축·실제 소비량·전체 질의시간 미확인 |
| 9/8 Qwen+Qwen embedding · Refined | 12 | 40.6781 | 완전한 구축·실제 소비량·전체 질의시간 미확인 |
| 9/8 Gemma+Qwen embedding · Seed | 12 | 42.0306 | 완전한 구축·실제 소비량·전체 질의시간 미확인 |
| 9/8 Gemma+Qwen embedding · r40 | 12 | 37.5035 | 완전한 구축·실제 소비량·전체 질의시간 미확인 |
| 9/8 Gemma+Qwen embedding · Refined | 12 | 43.0590 | 완전한 구축·실제 소비량·전체 질의시간 미확인 |
| 9/10 Qwen+MiniLM · Seed | 12 | 70.7872 | 별도 native 비용 보고 있음 |
| 9/10 Qwen+MiniLM · r40 | 12 | 44.8447 | 별도 native 비용 보고 있음 |
| 9/10 Qwen+MiniLM · Refined | 12 | 44.8447 | 별도 native 비용 보고 있음 |
| 9/10 Qwen+MiniLM · Seed-parent | 12 | 70.7872 | 별도 native 비용 보고 있음 |

9/8 JSONL에는 input/output 토큰과 generation_batch_seconds가 있지만, 전자는 캐시 결과에서도 읽히는 논리적 답변 길이이고 후자는 검색·임베딩·packing을 제외하는 좁은 구간이다. 이를 최신 baseline의 실제 전체 토큰·질의시간으로 옮겨 적지 않았다.

**자체 방법의 구축시간을 보충해서 설명하면**

완전한 구축 wall time은 없다. 다만 native LLM 호출 내부시간 합은 아래처럼 남아 있다. GPU 대기·CPU·메모리 가공·모델 로드 등 전체 범위를 포괄하는 baseline 구축 wall과 같은 열에 놓으면 안 된다.

| 구축 단계 | LLM 토큰 M | native LLM 호출시간 합 | 12이력당 평균 native 초 |
|---|---:|---:|---:|
| Seed/r40 공통 기본 구축 | 3.347825 | 865.397874초 | 72.116490 |
| source probe 생성 | 2.337593 | 1309.094520초 | 109.091210 |
| source utility scoring | 9.838130 | 1352.463196초 | 112.705266 |
| Refined 구성 합 | 15.523548 | 3526.955591초 | 293.912966 |

Seed-parent가 가져온 메모리 위에서 수행한 CPU augmentation·저장·검사는 총12.959699초다. 이를 전체 구축1.08초/이력이라고 표기하면 앞선 구축·probe·scoring을 빠뜨리므로 그렇게 표시하지 않았다.

**같은 12문항의 개별 점수**

| question_id | Full-context | Seed MiniLM | r40 MiniLM | Refined MiniLM | Seed-parent |
|---|---:|---:|---:|---:|---:|
| cc6d1ec1 | 1.02 | 40.00 | 40.00 | 40.00 | 40.00 |
| 157a136e | 0.00 | 100.00 | 0.00 | 100.00 | 100.00 |
| e4e14d04 | 11.76 | 100.00 | 50.00 | 50.00 | 100.00 |
| e6041065 | 4.44 | 100.00 | 100.00 | 0.00 | 100.00 |
| d6062bb9 | 2.82 | 0.00 | 0.00 | 0.00 | 0.00 |
| gpt4_b5700ca9 | 5.63 | 66.67 | 0.00 | 0.00 | 66.67 |
| 118b2229 | 6.25 | 100.00 | 100.00 | 100.00 | 100.00 |
| 27016adc | 0.00 | 100.00 | 0.00 | 0.00 | 100.00 |
| 8cf51dda | 54.21 | 63.41 | 63.41 | 63.41 | 63.41 |
| 41275add | 53.16 | 57.14 | 62.50 | 62.50 | 57.14 |
| 4baee567 | 20.78 | 22.22 | 22.22 | 22.22 | 22.22 |
| gpt4_7de946e7 | 5.33 | 100.00 | 100.00 | 100.00 | 100.00 |

[모든 이전 모델·방법을 포함한 문항별 CSV](MATCHED_SCORES.csv)

**판단 가능한 범위**

기존12개에서는 자체 Seed가 r40/Refined보다 높은 진단 F1를 냈고, 검색 메모리 방식은 답변 단계 토큰을 크게 줄였다. 반면 원문 기반 구축·probe·utility가 비싸서 전체 비용은 다른 결론이 된다. LangMem·SimpleMem·LightMem까지12개 평균으로 비교하려면 현재 스냅샷에 없는 나머지11개 대응 예측이 필요하다. 현재1개 결과와 서로 다른 전체 부분집합 평균만으로 네 방식의 정확도·효율 우위를 확정하지 않는다.
