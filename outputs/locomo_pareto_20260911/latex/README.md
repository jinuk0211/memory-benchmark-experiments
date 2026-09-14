# Native LaTeX / PGFPlots figures

이 묶음은 그래프를 LaTeX에서 직접 그립니다. PNG/PDF 이미지 삽입, shell escape 또는 외부 CSV 파일이 필요하지 않습니다. 좌표는 각 `.tikz.tex` 안에 내장되어 있습니다.

## Overleaf

1. 이 ZIP을 새 프로젝트로 업로드합니다.
2. Main document를 `main.tex`로 선택합니다.
3. pdfLaTeX으로 컴파일합니다. 총 세 그림이 각각 한 페이지에 배치됩니다.

로컬에 TeX 컴파일러가 없어 실제 컴파일 및 컴파일 결과의 레이블 배치는 확인하지 못했습니다. 원자료와 좌표의 일치, 괄호/환경 구조와 설정을 점검했습니다. PGFPlots 1.18 이상이 필요합니다.

## 기존 논문에 넣기

아래 파일을 논문의 `.tex` 파일과 같은 폴더에 둡니다. Preamble에 한 번만 추가합니다.

```latex
\input{locomo_style.tex}
```

본문에서 다음처럼 사용합니다. 두 단 논문에는 `figure*`, 한 단 논문에는 `figure`를 사용합니다.

```latex
\begin{figure*}[t]
  \centering
  \input{tokens_all.tikz.tex}
  \caption{Quality--token trade-off across 15 LoCoMo configurations.
  Tokens include the selected build/preparation and QA path, excluding
  embeddings. $\dagger$ denotes reconstructed costs; $\geq$ denotes
  recorded lower bounds.}
  \label{fig:locomo-pareto}
\end{figure*}
```

- `tokens_all.tikz.tex`: 전체 15개 구성, 경계 R40--Seed--E-Mem.
- `tokens_main.tikz.tex`: 메인 표 9개 구성, 경계 LightMem--Our--E-Mem.
- `latency.tikz.tex`: baseline 6개 + 내부 구성 4개, 측정 범위별 두 패널.
- `locomo_style.tex`: 공통 패키지, 색상, 축, 레이블 글꼴.
- `main.tex`: 세 그림과 상세한 영문 캡션을 포함한 전체 문서.

제공된 색상은 밝은 배경의 논문용입니다. 색상은 `locomo_style.tex`의 `\definecolor`, 크기는 각 axis의 `width`와 `height`, 레이블 위치는 `xshift`와 `yshift`로 수정합니다. 15개 점의 가독성을 위해 한 단 논문 폭 또는 두 단 논문의 전체 폭을 권장합니다.

토큰 합계의 재구성/계측 하한과 latency 측정 범위는 기존 결과와 동일합니다. Seed/R40/Our/Recursive 준비 비용을 모두 같게 처리하지 않았으며, 캐시로 재사용한 답변의 logical reader 토큰도 포함했습니다. Latency는 서로 다른 시간 정의를 가로질러 frontier를 연결하지 않습니다. 두 패널의 F1 축 범위도 다릅니다.

코드 참고: [PGFPlots 공식 문서](https://tikz.dev/pgfplots/), [로그축과 역방향 축](https://tikz.dev/pgfplots/reference-scaling).