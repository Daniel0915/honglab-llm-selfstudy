# LLM 바닥부터 만들기 — 자습 자료

[홍랩 AI](https://www.honglab.ai/) **LLM 바닥부터 만들기** 시리즈 학습 저장소입니다.
성격이 다른 두 묶음이 들어 있습니다.

| | 성격 |
|---|---|
| **Part1** | 강의를 **수강한 뒤 정리한 학습 노트** (마크다운) |
| **Part2 · Part3 · Part4** | **공개된 목차와 텍스트 보충 자료만** 참고하여 코드와 설명을 처음부터 새로 쓴 **독립 자습 자료** (노트북) |

> ⚠️ **Part1** — 수강 후 정리한 **개인 학습 노트**입니다.
> 영상 설명을 그대로 옮긴 것이 아니라, 각 강의 페이지의 **텍스트 보충 자료**와
> **퀴즈 해설**을 바탕으로 요점과 연결 고리를 제 언어로 다시 적었습니다.
> 강의의 실습 코드와 데이터는 저작권 자료이므로 포함하지 않았습니다.
>
> ⚠️ **Part2 · Part3 · Part4** — **강의 영상의 내용이 아닙니다.** 영상은 보지 않았습니다.
> 공개된 **강의 목차**와, 파트4의 경우 강의 페이지에 붙어 있는 **텍스트 보충 자료**
> (핵심 개념 · 자주 나오는 질문 · 출처)만 읽고 작성했습니다.
>
> 어느 쪽이든 강의를 대체하지 않으며, 원 강의의 설명·실습 코드와 다를 수 있습니다.
> **다르면 강의 쪽이 맞습니다.**
>
> 강의 자체는 [홍랩 AI](https://www.honglab.ai/)에서, 공식 실습 자료는
> [HongLabInc/HongLabAI](https://github.com/HongLabInc/HongLabAI)에 있습니다.
> 이 저장소는 공식 자료를 포함하지 않습니다.

노트북 14개 + 문서 4편 + Part1 학습 노트 15편.
각 장은 **앞 장이 못 푼 문제에서 시작**하도록 엮었습니다.

---

## 학습 순서

### Part1 — 통계적 언어 모델 ([llmpt1](https://www.honglab.ai/courses/llmpt1)) · 수강 노트

| 노트 | 주제 | 핵심 질문 |
|---|---|---|
| [Ch1/0101](Part1/Ch1/0101_AIAgentStructure.md) | AI 에이전트의 구조 | 에이전트는 왜 지금에서야 가능해졌나 |
| [Ch1/0102](Part1/Ch1/0102_UsingAIWell.md) | AI 제대로 알고 쓰자 | 기억 없는 주사위를 어떻게 다루나 |
| [Ch1/0103](Part1/Ch1/0103_LLMHistory.md) | LLM의 발전 과정 | 1948년 섀넌에서 지금까지 |
| [Ch2/0203](Part1/Ch2/0203_CharFrequency.md) | 글자 빈도 통계 | 빈도 → 확률, 왜 최고값만 고르면 안 되나 |
| [Ch2/0204](Part1/Ch2/0204_Bigram.md) | 바이그램 | 조건부 확률 한 칸이 만드는 차이 |
| [Ch2/0206](Part1/Ch2/0206_Ngram.md) | 엔그램 | **일반화 vs 암기** — n의 스위트 스폿 |
| [Ch2/0208](Part1/Ch2/0208_Morpheme.md) | 형태소 분석 | 교착어 · 토큰 · 백오프 · 과적합 |
| [Ch2/0210](Part1/Ch2/0210_Appendix_NgramHistory.md) | [부록] N-그램의 역사 | 1913 마르코프 → 2024 Infini-gram |

전체 15편(퀴즈 메모·부록 포함)은 [`Part1/README.md`](Part1/README.md).
**파트1은 C로 구현하는 과정이라 이 저장소에는 노트만 있습니다.**

### Part2 — 뉴럴 언어모델 ([llmpt2](https://www.honglab.ai/courses/llmpt2))

| 노트북 | 주제 | 핵심 질문 |
|---|---|---|
| [Ch3/0301](Part2/Ch3/0301_NgramCurse.ipynb) | N-gram의 저주 | 문맥을 길게 보면 왜 오히려 망가지는가 |
| [Ch3/0302](Part2/Ch3/0302_Perplexity.ipynb) | 혼란도 | 모델의 좋고 나쁨을 숫자 하나로 말하는 법 |
| [Ch3/0303](Part2/Ch3/0303_WordEmbedding.ipynb) | 단어 임베딩 | 벡터가 어떻게 희소성을 푸는가 |
| [Ch4/0401](Part2/Ch4/0401_NPLM.ipynb) | NPLM 구조 | Bengio 2003은 무엇을 바꿨고 무엇을 못 풀었나 |
| [Ch4/0402](Part2/Ch4/0402_Tokenization.ipynb) | 토크나이징 | 한국어에서 "단어"를 무엇으로 셀 것인가 |
| [Ch4/0403](Part2/Ch4/0403_TrainAndGenerate.ipynb) | 학습 최적화 · 생성 | 실무 학습 루프와 생성 전략 |

### Part3 — 트랜스포머와 사전학습 ([llmpt3](https://www.honglab.ai/courses/llmpt3))

| 노트북 | 주제 | 핵심 질문 |
|---|---|---|
| [Ch5/0501](Part3/Ch5/0501_Attention.ipynb) | 어텐션 | 고정 문맥에서 어떻게 탈출하는가 |
| [Ch5/0502](Part3/Ch5/0502_TransformerBlock.ipynb) | 트랜스포머 블록 | 잔차 · 정규화 · FFN은 각각 왜 필요한가 |
| [Ch5/0503](Part3/Ch5/0503_ModernRecipe.ipynb) | 현대적 레시피 | 2017년과 지금의 차이 (RoPE · GQA · SwiGLU) |
| [Ch6/0601](Part3/Ch6/0601_MiniGPT.ipynb) | 미니 GPT | 부품을 모아 작동하는 GPT 만들기 |
| [Ch6/0602](Part3/Ch6/0602_ScaleUp_RunPod.md) | 클라우드 스케일업 | M1을 벗어나 진짜 GPU에서 돌리기 |
| [Ch6/0603](Part3/Ch6/0603_InContextLearning.ipynb) | 인컨텍스트 러닝 | 가중치를 안 고치고 어떻게 배우는가 |

### Part4 — 파인튜닝과 대화형 AI ([llmpt4](https://www.honglab.ai/courses/llmpt4))

| 파일 | 주제 | 핵심 질문 |
|---|---|---|
| [Ch7/0701](Part4/Ch7/0701_TrainingPipeline.ipynb) | 훈련 파이프라인 | 일곱 단계를 **손실 함수**로 가르면 |
| [Ch7/0702](Part4/Ch7/0702_InstructionTuning.ipynb) | 지시 튜닝 | 대화 한 건이 **모델 입력**이 되기까지 |
| [Ch7/0703](Part4/Ch7/0703_FullFineTuning.ipynb) | 전체 미세 조정 | 미세 조정이 바꾸지 **못하는** 것 |
| [Ch7/0704](Part4/Ch7/0704_SFT_RunPod.md) | 실전 실행 | 24GB GPU에서 진짜로 돌리기 |
| [Ch7/0705](Part4/Ch7/0705_Ch7_QuizNotes.md) | 퀴즈 자습 메모 | 25문항을 노트북 어디서 확인했는가 |
| [Ch8](Part4/Ch8/README.md) | 페르소나 만들기 | ⏳ **강의 미공개** — 예습 포인트만 |

**순서대로 보세요.** 자세한 것은 [`Part4/README.md`](Part4/README.md).

---

## 전체 흐름

```
Part1  빈도 세기 → 조건부 확률 → n을 올리면 외워버린다 (일반화 vs 암기)
  └→ Part2  N-gram의 한계 → 임베딩으로 희소성 해결 → NPLM (그래도 문맥 길이가 고정)
        └→ Part3  어텐션으로 가변 문맥 → 트랜스포머 블록 → 미니 GPT → 사전학습
              └→ 사전학습의 부산물: 인컨텍스트 러닝
                    └→ 그 한계가 파인튜닝의 이유
                          └→ Part4  SFT(형식 가르치기) → 전체 미세 조정
                                └→ Ch8  PEFT/LoRA · 페르소나 (강의 미공개)
```

한 줄로 줄이면 **「다음 단어를 맞히는 기계」가 「대화하는 조수」가 되기까지**입니다.
파트4의 결론 문장이 그 경계를 말해 줍니다 — **형식은 미세 조정으로, 능력은 사전훈련으로.**

---

## 환경 설정

두 개의 가상환경을 씁니다. **파트4부터는 3.11이 필요합니다**
(최신 `transformers` 가 Python 3.9 지원을 끊었습니다).

| 환경 | 파이썬 | 쓰는 곳 |
|---|---|---|
| `.venv` | 3.9+ | Part2 · Part3 (torch · numpy · matplotlib 만) |
| `.venv311` | 3.11 | Part4 (+ transformers · datasets · peft · trl) |

```bash
cd ~/git/honglab-llm-selfstudy

# Part2 · Part3
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/jupyter lab .

# Part4
python3.11 -m venv .venv311
./.venv311/bin/pip install -r Part4/requirements.txt
./.venv311/bin/jupyter lab Part4
```

가상환경과 모델 체크포인트는 `.gitignore` 에 들어 있어 커밋되지 않습니다.

### 맥에서 돌리기

- [`common.py`](common.py) 의 `get_device()` 가 **cuda → mps → cpu** 순으로 고릅니다.
  인터넷 코드 대부분이 `"cuda"` 로 하드코딩돼 있어서 그냥 쓰면 맥에서 터집니다.
- **Part2 · Part3 전부 M1 Air 8GB에서 돌아갑니다.** 가장 무거운 `0601_MiniGPT` 도 MPS로 2분 이내.
- **Part4** 는 `0703` 2절(초소형 모델 배선 확인)까지 맥에서 2~3분.
  실제 베이스 모델 훈련은 [`Part4/Ch7/0704_SFT_RunPod.md`](Part4/Ch7/0704_SFT_RunPod.md) 를 따라 GPU에서.
- Part4는 **`transformers` 없이도 전부 돌아갑니다** — [`Part4/chat.py`](Part4/chat.py) 의
  `ByteTokenizer` 로 자동 대체되고, 손실 마스킹·배치 구성의 동작은 똑같이 확인됩니다.
  토큰 수만 실제보다 많아집니다.

---

## 공용 모듈

| 파일 | 제공하는 것 |
|---|---|
| [`common.py`](common.py) | `get_device()` · `set_seed()` · `count_params()` · 내장 말뭉치 |
| [`Part4/chat.py`](Part4/chat.py) | ChatML 렌더링 · **손실 마스킹** · 패딩 · 긴 대화 쪼개기 · 토크나이저 |

```python
# Part4 핵심 네 줄
render_chatml(messages, add_generation_prompt)   # 대화 → ChatML 문자열
build_sft_example(tok, messages, mask_prompt)    # → (input_ids, labels)
preview_mask(tok, ex)                            # 가린 토큰을 ░ 로 그려 보기
collate(batch, pad_id)                           # 패딩 라벨 = IGNORE_INDEX
```

---

## 설계 원칙

**1. 모든 주장은 실행해서 확인합니다**

말로 넘어가지 않습니다. 셀을 돌리면 숫자가 나옵니다.

- 「잔차가 없으면 기울기가 사라진다」 → 층별 그래디언트 크기를 출력
- 「RoPE의 내적은 상대 거리에만 의존한다」 → 직접 계산해서 대조
- 「가린 토큰도 모델은 읽는다」 → 인과 어텐션을 달고 가린 위치의 그래디언트를 출력
- 「마스킹 없이 배우면 질문까지 지어낸다」 → 두 모델을 훈련해 **0/10 vs 8/10** 으로 셈
- 「종료 토큰을 빼면 안 멈춘다」 → **17토큰에 멈춤 vs 상한 80토큰에서 잘림**

**2. 실패와 한계를 숨기지 않습니다**

내장 말뭉치는 50문장뿐이라 모델이 계속 **과적합합니다.** 감추는 대신 전면에 놓고
"과적합을 알아보는 눈"을 기르는 재료로 씁니다.
파트4의 온도 실험도 마찬가지입니다 — 장난감 모델이 과적합해서 T=3.0까지 올려야 흔들리고,
**셀이 그 사실을 직접 출력합니다.**

**3. 각 장은 앞 장이 못 푼 문제에서 시작합니다**

**4. 작게 먼저 돌립니다**

`0703` 2절은 랜덤 초기화한 초소형 모델로 3절과 **완전히 같은 코드**를 검증합니다.
큰 모델에서 바로 시작하면 오타 하나 찾는 데 수십 분이 듭니다. 실무 순서 그대로입니다.

---

## 말뭉치 키우기 (중요)

[`common.py`](common.py) 의 내장 말뭉치는 **개념 확인용**입니다.
이걸로는 진짜 언어 모델이 되지 않습니다.

```python
from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
with open("ko_wiki.txt", "w", encoding="utf-8") as f:
    for row in ds.select(range(20000)):
        f.write(row["text"].replace("\n", " ") + "\n")
```

각 노트북에서 `load_corpus("ko_wiki.txt")` 또는 `CORPUS_PATH = "ko_wiki.txt"` 로 바꾸면
**코드 수정 없이 그대로 돌아갑니다.** 자세한 것은
[Part2 4-2 노트북 6절](Part2/Ch4/0402_Tokenization.ipynb).

> M1 8GB에서는 위키 전체를 올리지 마세요. 수십 MB면 충분히 다른 세상이 보입니다.

---

## 강의 공개 현황 (2026-10-05 확인)

| 파트 | 상태 |
|---|---|
| LLM 파트1. 통계적 언어 모델 | 공개 · **수강 완료** (챕터1 · 챕터2) |
| LLM 파트2. 뉴럴 언어모델 | 공개 |
| LLM 파트3. 트랜스포머와 사전학습 | 공개 |
| LLM 파트4 — **챕터 7 (파인튜닝)** | 공개 · 인트로 · 훈련 파이프라인 · 지시 튜닝 · 전체 미세 조정 · 퀴즈 |
| LLM 파트4 — **챕터 8 (페르소나 만들기)** | ⏳ **미공개** (「얼리버드 안내」 텍스트만) |

챕터 8이 공개되면 [`Part4/Ch8/`](Part4/Ch8/) 에 같은 번호 체계로 채웁니다.

---

## 파일 구조

```
.
├── README.md
├── requirements.txt          Part2 · Part3 용
├── common.py                 디바이스 선택 · 말뭉치 로딩 · 유틸
├── Part1/                    수강 노트 (마크다운)
│   ├── README.md
│   ├── Ch1/  0101 0102 0103 0104
│   ├── Ch2/  0201 … 0209 · 0210 0211(부록)
│   └── _lecture_notes/       강의 페이지 보충 자료 원본
├── Part2/
│   ├── Ch3/  0301 0302 0303
│   └── Ch4/  0401 0402 0403
├── Part3/
│   ├── Ch5/  0501 0502 0503
│   └── Ch6/  0601 0602(md) 0603
└── Part4/
    ├── README.md  requirements.txt  chat.py
    ├── Ch7/  0701 0702 0703 0704(md) 0705(md)
    ├── Ch8/  README.md  (강의 미공개)
    └── _lecture_notes/       강의 페이지 보충 자료 원본
```
