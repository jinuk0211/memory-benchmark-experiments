# Decision-Aware Memory Cards 본문 비교 감사

2026-09-08 검토. 대상은 **Decision-Aware Memory Cards: Counterfactual-Inspired Context Selection and Compression for Tool-Using LLM Agents**, Xinyu Guan · Qianyang Zhao · Yuming Deng. arXiv 최초 제출은2026-06-06, 확인한 최신판은 **v3, 2026-09-04**이다. 공식 **HTML 본문과15쪽 PDF에 접근 성공**했다. [서지·버전 이력](https://arxiv.org/abs/2606.08151), [버전 고정 HTML](https://arxiv.org/html/2606.08151v3).

이 문서는 기존 `METHOD_CONTRIBUTION.md`의 초록 수준 비교를 보완한다. 공식 본문 §3–§6을 직접 확인했으며, 아래 페이지는 확인한v3 PDF 기준이다. GitHub 코드·별도 supplement·이전 버전과 다른 논문은 검토하지 않았다.

## 본문에서 확인한 중복과 차이

| 쟁점 | 논문에서 확인한 내용과 위치 | 현재 고정 구현과의 비교 |
| --- | --- | --- |
| 선택 시점·입력 | 현재task의 graph 후보를 inference-time에 점수화한다. Gold context ID는 선택에 쓰지 않는다. [§3.1·3.3·3.5, pp.3–5](https://arxiv.org/pdf/2606.08151v3#page=4) | 현재는 benchmark 질문을 보기 전에 source 대화로 QA를 생성해 메모리를 구성한다. Gold를 선택에서 배제한다는 원칙은 겹치지만, 구성 시점과 utility 질의의 출처가 다르다. |
| Utility 측정 | 행동변화·필요성·성과·위험의 judge/simulator/ranker 추정치에 cost를 더해 고정 heuristic 가중합을 사용한다(Eq.3). [§3.2, p.4](https://arxiv.org/pdf/2606.08151v3#page=4) | 현재는 source 정답 토큰의 직접 조건부 평균 logprob를 full/empty/single 문맥에서 측정한다. 아래 정규화식을 사용하며, 별도 행동변화·위험 judge 점수는 없다. |
| 메모리 표현 | 선택한 근거를 다섯 필드의 typed card로 압축한다. [§3.4, p.5](https://arxiv.org/pdf/2606.08151v3#page=5) | 현재는 필요한 원문 ID를 덮는 기존 r40 parent 단위를 통째로 재사용하고 source 질문을 검색용 index text로 붙인다. 이 메모리 형식과 변환 방식은 다르다. |
| 예산 | 고정 context 예산으로 선택·압축하며, 압축 전후의 선택 변화를 구분한다. [§3.5, p.5; §5.4, pp.9–10](https://arxiv.org/pdf/2606.08151v3#page=5) | 효용·예산에 따른 선택이라는 큰 틀은 겹친다. 현재는 원래 parent를 유지하고 **추가 저장2000토큰**에 multiple-choice knapsack을 적용한다. Reader2048토큰은 별도 제약이다. |
| 모델 비교의 범위 | Qwen3.5-9B QLoRA는 judge 선택 일치도용 surrogate다. 주요 실험은 파일 검색·압축이다. [§3.6, pp.5–6; §5.5, pp.10–11](https://arxiv.org/pdf/2606.08151v3#page=6) | 현재의 frozen recipe를 writer·likelihood scorer·reader 전체에 적용하는 Qwen3.5/Gemma 및 LoCoMo/LongMemEval 전이와 실험 단위가 다르다. 모델 이름이 같다는 사실은 같은 방법이나 검증을 뜻하지 않는다. |

현재 구현의 비교 기준은 이미 완료한 코드 감사다. 여기서 `ell(S)`는 source 정답의 teacher-forced 평균 logprob다.

```text
admit: source full-answer F1 >= 0.8 and ell(full) - ell(empty) >= 0.1
gain(S) = clip((ell(S) - ell(empty)) / (ell(full) - ell(empty)), 0, 1)
```

실제 parent_single은 full/single 후보를 남기고 pair 후보를 제외한다. 후보의 원문 ID를 기존 parent payload로 매핑한 뒤, probe당 최대 하나를 선택한다. 비용은 payload와 별도 index text의 합이며8토큰 단위로 올림한다. 이 구체적인 구현 설명은 [기존 방법 감사](METHOD_CLAIM_AUDIT.md)에 근거하며 이번에 코드를 변경하거나 재감사하지 않았다.

## 미확인 사항과 독창성 주장의 한계

본문에서는 **source-only QA → full/empty/single answer-NLL 정규화 → 기존 parent mapping → 추가2000토큰 knapsack**이라는 동일 조합을 명시적으로 확인하지 못했다. §3.5에는 정확한 knapsack 알고리즘도 명시돼 있지 않다. 이는 검토한 본문에 대한 관찰이며, 공개 코드나 다른 선행연구에 그런 구현이 없다는 증명은 아니다. [§3.2–3.5, pp.4–5](https://arxiv.org/pdf/2606.08151v3#page=4).

논문도 utility를 정식으로 식별된 causal effect로 주장하지 않는다. [§6, p.11](https://arxiv.org/pdf/2606.08151v3#page=11). 현재 구현 역시 likelihood 차이나 parent 매핑을 causal identification 또는 검증된 NLL 보존으로 표현할 근거가 없다.

따라서 넓은 **counterfactual-inspired utility + 예산 내 memory 선택** 자체를 현재 작업의 새로운 발상이라고 주장하기는 어렵다. 이번 한 논문과 구분되는 구체적 구현 지점은 **benchmark 질문 이전의 source QA 신호, 직접 답변 likelihood 정규화, 기존 parent payload 재사용, 별도 추가 저장 예산의 결합**이다. 이는 비교 가능한 차이이지 독창성이나 유효성이 확립됐다는 결론은 아니다. 동일 저장 예산의 선택 우월성이나 전체 모델·데이터 일반성도 이 문헌 검토로 입증되지 않는다.

새 arm, threshold, 프롬프트, 실험 또는 코드를 제안·실행하지 않았다. 새 로컬 파일 `RELATED_WORK_FULLTEXT_AUDIT.md`만 작성했으며, 기존 문서·remote·서비스는 수정하거나 업로드하지 않았다.
