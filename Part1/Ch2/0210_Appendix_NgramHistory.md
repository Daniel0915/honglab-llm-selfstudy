# 2-10. [부록] N-그램 언어 모델의 역사

> 원본: [`_lecture_notes/Ch2-11_부록_N그램의역사.md`](../_lecture_notes/Ch2-11_부록_N그램의역사.md)

---

## 한 줄

> **80년 된 N-그램은 여전히 현역이다.**
> 주류 생성 모델의 자리에서는 물러났지만,
> 현대 LLM 세계 안에서도 평가 · 검색 · 전처리 · 디코딩 · 엣지 환경에 자리 잡고 있다.

파트1이 N-그램부터 시작하는 이유이기도 하다.

---

## 1940~50년대 · 창시기 — 정보이론과 언어가 만나다

| 연도 | 사람 | 일 |
|---|---|---|
| **1913** | A. A. Markov | 푸시킨 〈예브게니 오네긴〉 약 2만 글자를 자음·모음으로 나누고 **전이확률**을 손으로 계산. 마르코프 체인이 태어난 순간이자 N-그램의 수학적 뿌리 |
| **1948** ★ | Claude Shannon (Bell Labs) | *A Mathematical Theory of Communication*. 영어를 N-그램 확률 모델로 보고 글자·단어 근사를 **직접 생성해 보였다** |
| **1951** | Claude Shannon | *Prediction and Entropy of Printed English*. 사람에게 다음 글자를 맞히게 하는 **"Shannon game"** 으로 영어 엔트로피를 **약 1 bit/letter** 로 추정 |

> 섀넌의 2차 단어 근사 샘플:
> **"THE HEAD AND IN FRONTAL ATTACK ON AN ENGLISH WRITER…"**
> LLM 문장의 80년 전 선조.

1 bit/letter 라는 수치는 지금도 **언어 모델 품질의 궁극적 기준선**으로 남아 있다.

---

## 1970~80년대 · 음성인식 시대 — 규칙에서 통계로

| 연도 | 사람 | 일 |
|---|---|---|
| **1971~76** | Fred Jelinek (IBM) | IBM 음성인식 팀을 이끌며 N-그램 LM을 시스템의 핵심 엔진으로. **HMM + N-그램 LM** 조합이 이후 30년간 음성인식의 표준이 된다 |
| **1987** | Slava M. Katz | **Katz Backoff** — 데이터에 없는 N-그램의 확률을 낮은 차수로 되돌려 추정. **제로 카운트 문제를 처음 실용적으로 푼 작업** |

> "Every time I fire a linguist, the performance of the speech recognizer goes up."
> — Jelinek. 통계적 접근의 승리를 상징하는 한마디.

Katz Backoff 가 [`0208` 6절](0208_Morpheme.md) 의 백오프다.
강의 실습에서 중복 문장을 없앤 그 기법의 원전.

---

## 1990년대 · 통계 NLP의 전성기

| 연도 | 사람 | 일 |
|---|---|---|
| **1993** | Brown et al. (IBM) | *The Mathematics of Statistical Machine Translation*. 번역을 노이즈 채널 모델로 풀고 **번역문의 자연스러움을 N-그램 LM으로 채점**. 2015년 무렵까지 구글·MS 번역의 기반 |
| **1995** ★명작 | Kneser & Ney (RWTH Aachen) | **Kneser–Ney Smoothing**. 단순 빈도 대신 **「이 단어가 얼마나 다양한 문맥에서 등장하는가」**를 스무딩에 반영 |
| **1999** | Chen & Goodman | **Modified Kneser–Ney**. 이후 약 15년간 N-그램 LM의 절대 강자. 오늘날 KenLM에도 기본 옵션으로 살아 있다 |

---

## 2000년대~2010년대 초 · 대규모 통계 시대

| 연도 | 사람 | 일 |
|---|---|---|
| **2003** | Bengio et al. | *A Neural Probabilistic Language Model*. 당시엔 틈새였지만 **2010년대 딥러닝 혁명의 씨앗** |
| **2007** | Brants et al. (Google) | 1조 토큰 코퍼스에서 N-그램 LM + **"Stupid Backoff"** (EMNLP 2007). **Web 1T 5-gram** 데이터셋이 LDC로 공개되어 세계 표준 자원이 됨 |
| **2011** | Kenneth Heafield | **KenLM** — 빠르고 메모리를 적게 쓰는 N-그램 질의 라이브러리. **지금도 현역인 사실상 표준 구현체** |

> 2003년 Bengio 가 [Part2/Ch4/0401 NPLM](../../Part2/Ch4/0401_NPLM.ipynb) 의 그 논문이다.
> 파트1의 역사 연표와 파트2의 첫 노트북이 여기서 맞물린다.

---

## 2010년대 중~후반 · 신경망 부상

| 연도 | 사람 | 일 |
|---|---|---|
| **2013** | Mikolov et al. (Google) | **Word2Vec** — N-그램이 못 잡던 **의미적 일반화**를 신경망이 잡기 시작 |
| **2017** ★전환 | Vaswani et al. (Google) | *Attention Is All You Need*. 이 시점부터 주요 벤치마크에서 N-그램은 **베이스라인** 자리로 |
| **2018~20** | Google · OpenAI | BERT · GPT-2 · GPT-3. 대부분 벤치마크에서 N-그램은 성능 경쟁에서 사실상 은퇴 |

---

## 2020년대~현재 · 그래도 살아남은 N-그램

주류 생성 모델 자리는 Transformer에게 내줬지만,
**가볍고 빠르고 해석이 쉽다**는 강점 때문에 LLM 시스템 곳곳에서 현역이다.

| # | 자리 | 내용 |
|---|---|---|
| 01 | **평가 지표** | **BLEU · ROUGE · chrF** — 기계번역·요약의 사실상 표준 지표가 지금도 N-그램 일치 기반. LLM 논문들도 이 지표로 점수를 보고한다 |
| 02 | **검색 · RAG** | **BM25**의 끈질긴 생명력. N-그램·TF-IDF 같은 전통 IR 기법이 RAG 파이프라인의 **1단계 리트리버**로 쓰인다. 의미 검색과 하이브리드로 묶이는 경우가 많다 |
| 03 | **반복 억제 (디코딩)** | **`no_repeat_ngram_size`** — LLM 디코딩의 대표 옵션. 생성이 같은 구절을 맴도는 반복 루프를 막는 표준 도구 |
| 04 | **음성·번역 리스코어링** | 작은 기기나 빠른 응답이 필요한 환경에서, 신경망 후보에 **KenLM으로 점수를 다시 매기는** 하이브리드가 아직 널리 쓰인다 |
| 05 | **입력기(IME) · 키보드** | 스마트폰 자동완성, 한영 변환 — 반응 속도와 메모리가 중요한 곳에서 N-그램이 여전히 합리적 선택 |
| 06 | **데이터 중복 제거** | LLM 사전학습 데이터 중복 제거에 쓰는 **MinHash**의 본질은 문서를 N개 단어씩 자른(**shingling**) 겹침 측정. **LLM을 만들려면 먼저 N-그램이 필요한 셈** |
| 07 | **Infini-gram (2024)** | AI2가 초대규모 N-그램 LM을 만들어 LLM과 비교·보완한 논문 (Liu 외). 「LLM 시대에도 N-그램이 할 수 있는 일」을 2024년에 다시 묻는 작업 |
| 08 | **교육의 출발점** | 스탠퍼드 CS224N, Jurafsky의 SLP, 그리고 이 강의. **조건부 확률 · 스무딩 · 희소성 문제의 원형**이 여기 모여 있다 |

**03번은 직접 써 본 사람이 많을 것이다** — `no_repeat_ngram_size` 는
[`0203` 2절](0203_CharFrequency.md)의 「다다다다」 문제에 대한
2026년식 처방이고, 처방의 도구가 여전히 N-그램이다.

---

## 참고 자료 (강의 제시)

- Shannon (1948), *A Mathematical Theory of Communication* — N-그램 언어 모델의 원년
- Shannon (1951), *Prediction and Entropy of Printed English* — Shannon game, 영문 엔트로피
- Jurafsky & Martin, **SLP3 Ch.3** — N-gram Language Models — 표준 레퍼런스
- Heafield (2011), **KenLM**
- Chen & Goodman (1999), *Smoothing Techniques* — Modified Kneser–Ney
- Liu 외 (2024), **Infini-gram**
