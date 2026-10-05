# 2-1. 코딩 AI 준비

> 수강 후 정리한 학습 노트입니다. 원본은
> [`_lecture_notes/Ch2-02_코딩AI준비.md`](../_lecture_notes/Ch2-02_코딩AI준비.md) ·
> [`Ch2-01_안내_무료코딩AI.md`](../_lecture_notes/Ch2-01_안내_무료코딩AI.md).

---

## 한 줄

이 챕터는 **N-그램을 배우는 동시에 코딩 AI 쓰는 법을 배우는** 이중 과정이다.
C 코드를 손으로 쓰는 것이 아니라 **AI에게 시켜서 만드는 과정 자체가 교재**다.

---

## 1. 실습 환경

| 항목 | 선택 |
|---|---|
| 주 언어 | **C언어** (전처리·도구 작업은 파이썬 보조) |
| 에디터 | Visual Studio Code |
| 코딩 AI | **Claude Code의 Opus 모델** 기준. Gemini · ChatGPT 도 가능 |

> 모델마다 특성이 달라 진행 과정이 다를 수 있다.
> 여러 모델을 써 보는 것 자체가 AI 활용 연습으로 권장된다.

**왜 C인가** — 바닥부터 만든다는 이 시리즈의 전제에 맞는다.
해시테이블·비트 연산·메모리를 직접 다루게 되고,
그 덕에 「왜 자료구조가 필요한가」가 피부로 온다.
(참고: 이 저장소의 Part2~Part4는 파이썬/PyTorch 기반이다.)

---

## 2. AI 사용 팁 — 강의가 실제로 겪은 것

| 관찰 | 내용 |
|---|---|
| **같은 입력에 다른 응답** | 두 세션에 똑같이 입력했는데 다른 답이 나왔다. 확률적 샘플링 → 본질적으로 비결정적 |
| **모델 성능 차이** | Haiku는 부족, Sonnet은 틀이 잡혀 있을 때 괜찮음, **Opus가 가장 자율적**. 컴파일러가 없을 때 Opus는 알아서 대안을 찾았지만 하위 모델은 사용자에게 떠넘겼다 |
| **컨텍스트의 점진적 구축** | 한 번 지시하면 세션 내내 그 방식으로 작성한다 |
| **시행착오 기록** | AI에게 시행착오를 마크다운에 기록하라고 하면 자동 정리해준다. **사람이 보려는 게 아니라 나중에 AI가 참고하도록** 남기는 것 |
| **권한 설정 자동화** | `.claude/settings.local.json` 에 읽기/쓰기/실행을 허용으로 두면 매번 묻지 않는다. 이 설정 파일도 AI에게 만들어달라고 하면 된다 |

**가장 실무적인 줄은 「시행착오 기록」이다.**
문서의 독자를 사람이 아니라 **다음 세션의 AI**로 잡는 발상 —
무상태([`0101` 3절](../Ch1/0101_AIAgentStructure.md))를 바깥에서 푸는 가장 싼 방법이다.

---

## 3. 무료로 따라 하려면

| 도구 | 특징 | 비고 |
|---|---|---|
| **Google Antigravity** | Gemini · Claude 등 모델 선택 · 주간 사용량 제한 | 신용카드 불필요 · VSCode와 별개 프로그램 |
| **GitHub Copilot Free** | 자동완성 월 2,000회 · 채팅 월 50회 | 모델 선택 불가, 작은 모델 배정 |
| **ChatGPT Free** | GPT-5 채팅 + Codex 에이전트 | 5시간 단위 사용량 제한 |

유료: Google AI Pro $20 · Claude Pro $20 (Claude Code 포함) ·
Claude Max $100~ · ChatGPT Plus $20.

> Gemini Code Assist 확장은 2026-06-18부터 개인 계정 요청을 받지 않는다.
> Google이 Antigravity로 통합했고 개인용 Gemini CLI도 같은 날 종료되었다.

### 윈도우에서 빌드가 안 될 때

AI가 코드는 써 놓고 「터미널에서 직접 실행해달라」며 멈추는 경우가 있다.
**컴파일러 경로가 안 잡힌 것**이다.

Visual Studio 2022 Community에서 「C++를 사용한 데스크톱 개발」을 설치해도
일반 터미널에서 `cl` 이 바로 돌지는 않는다. **같은 세션에서** `vcvars64.bat` 을
먼저 실행해야 한다 — 두 명령을 따로 시키면 세션이 끊겨 실패하므로
하나의 `.bat` 으로 묶는다.

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
chcp 65001 >nul
cl /nologo hello.c /Fe:hello.exe
hello.exe
```

`chcp 65001` 은 한글이 깨지지 않게 하는 설정.
시작 메뉴의 "Developer Command Prompt for VS 2022" 를 쓰면 경로가 미리 잡혀 있다.

> 무료 플랜은 작은 모델이 배정되어 이런 환경 문제를 스스로 못 풀고
> 사용자에게 넘기는 일이 잦다. 위 `.bat` 을 만들어두고 실행하라고 알려주면 된다.

---

## 4. 제공 데이터

강의 페이지의 다운로드에 `data.zip` 과 `korean_public_domain.zip` 이 붙어 있다.
(저작권 자료이므로 이 저장소에는 포함하지 않는다.)

---

## 다음

[`0202_DictionaryData.md`](0202_DictionaryData.md) — 통계는 결국 「개수 세기」다.
