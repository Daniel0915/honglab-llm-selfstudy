# 6-2. GPU 클라우드 스케일업 (RunPod)

> **이 문서의 목적**
> 6-1의 MiniGPT를 **M1 맥에서 벗어나 진짜 GPU에서 돌리는 것.**
> 그리고 그 과정을 **AI 에이전트에게 시켜 보는 것.**

노트북이 아니라 문서인 이유: 이 장의 실습은 코드 실행이 아니라 **원격 장비 운용**입니다.

---

## 1. 왜 클라우드가 필요한가

6-1 마지막 표를 다시 봅시다.

| 구성 | 파라미터 | fp16 가중치 | AdamW 학습 메모리 |
|---|---|---|---|
| 우리 MiniGPT | 0.8M | 0.002 GB | 0.01 GB |
| GPT-2 small | 124M | 0.25 GB | 2.0 GB |
| GPT-2 XL | 1.56B | 3.1 GB | 24.9 GB |
| Llama-2 7B | 6.6B | 13.2 GB | 105.4 GB |

**M1 Air 8GB의 한계선은 GPT-2 small 근처**입니다. 그마저도 느립니다.

학습 메모리가 가중치의 약 8배인 이유:

```
파라미터 1개당
  가중치 fp32        4 바이트
  기울기             4 바이트
  Adam 1차 모멘텀 m  4 바이트
  Adam 2차 모멘텀 v  4 바이트
  ────────────────────────────
  합계              16 바이트
```

여기에 **활성값(activation)** 이 더해집니다. 배치와 문맥 길이에 비례해서요.
그래서 "파라미터 × 16"은 하한선이지 실제 사용량이 아닙니다.

---

## 2. RunPod의 구조 — 무엇을 빌리는 것인가

**내 맥이 빨라지는 게 아닙니다.** 인터넷 저편의 리눅스 서버를 시간 단위로 빌리는 것입니다.

```
  M1 Air                       RunPod 서버 (예: RTX 4090 24GB)
┌──────────────┐              ┌────────────────────────────────┐
│ 터미널/브라우저 │ ──SSH/HTTP──▶│ Ubuntu + CUDA + PyTorch         │
│ Claude Code   │ ◀────────────│ 내 코드, 데이터셋, 체크포인트      │
└──────────────┘              └────────────────────────────────┘
  리모컨 역할                      여기서 실제 연산이 일어남
```

### 과금 단위

| 항목 | 과금 | 끄면? |
|---|---|---|
| **GPU 시간** | 파드가 **실행 중일 때만** | 멈춤 |
| **컨테이너 디스크** | 파드가 존재하는 동안 | 파드 삭제해야 멈춤 |
| **네트워크 볼륨** | 항상 | **볼륨을 삭제해야 멈춤** |

> ⚠️ **가장 흔한 실수**: 파드를 껐으니 과금이 끝난 줄 알았는데 볼륨 요금이 계속 나가는 것.
> 실습이 끝나면 **파드와 볼륨을 모두 삭제**하세요.

### GPU 고르기

| GPU | VRAM | 용도 |
|---|---|---|
| RTX 4090 | 24GB | **이 강의 실습의 기본값.** 7B LoRA 파인튜닝 가능 |
| RTX 3090 | 24GB | 4090보다 싸고 느림. 예산 우선이면 |
| A100 40/80GB | 40/80GB | 7B 전체 파인튜닝, 13B급 |
| H100 | 80GB | 필요 없습니다. 비쌉니다 |

**24GB가 개인 실습의 실질적 기준선**입니다. 그 아래로 내려가면 배치를 줄이느라 시간이 더 듭니다.

### Secure Cloud vs Community Cloud

| | 가격 | 중단 위험 |
|---|---|---|
| Secure Cloud | 비쌈 | 없음 |
| Community Cloud | 쌈 | **호스트 사정으로 회수될 수 있음** |

처음에는 **Secure Cloud**를 쓰세요. 학습이 두 시간 돌다 날아가면 아낀 돈보다 손해가 큽니다.
체크포인트 저장에 익숙해진 뒤에 Community로 내려가면 됩니다.

---

## 3. 실습 절차

### 3-1. 계정 준비

1. RunPod 가입
2. **Billing → Auto-Recharge 를 끕니다** ← 가장 중요
3. 2만 원 정도 충전

> Auto-Recharge가 켜져 있으면 잔액 소진 시 카드에서 자동으로 더 빠져나갑니다.
> 이걸 꺼 두면 **최대 손실이 충전액으로 고정**됩니다.

### 3-2. 파드 생성

- Template: **RunPod PyTorch** (CUDA·PyTorch가 미리 설치된 이미지)
- GPU: RTX 4090 × 1
- Container Disk: 20~50GB
- Expose: SSH (22), Jupyter (8888)

### 3-3. 접속

**방법 A — 브라우저 Jupyter**: 파드 카드의 `Connect` → `Jupyter Lab`. 가장 쉽습니다.

**방법 B — SSH** (권장):

```bash
# 맥 터미널에서
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

SSH 키가 없다면 먼저 만들고 RunPod 설정에 공개키를 등록하세요.

```bash
ssh-keygen -t ed25519 -C "ufo9363@educo.co.kr"
cat ~/.ssh/id_ed25519.pub        # 이 내용을 RunPod > Settings > SSH Public Keys 에 붙여넣기
```

### 3-4. 코드 올리기

```bash
# 맥에서 SelfStudy 폴더를 통째로 전송
rsync -avz -e "ssh -p <PORT>" ~/git/honglab-llm-selfstudy root@<POD_IP>:/workspace/

# 또는 git 을 쓴다면 (더 권장)
# 서버에서:
cd /workspace && git clone <내 저장소 주소>
```

### 3-5. 환경 확인

```bash
nvidia-smi                       # GPU가 보이는지
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`common.py` 의 `get_device()` 는 CUDA를 자동으로 잡습니다. **코드를 고칠 필요가 없습니다.**

```python
def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")   # ← 서버에서 여기로
    elif torch.backends.mps.is_available(): return torch.device("mps")   # ← 맥에서 여기로
    else: return torch.device("cpu")
```

이것이 `device = "cuda"` 로 하드코딩하지 않은 이유입니다.

### 3-6. 학습 실행

터미널을 닫아도 계속 돌게 하세요. **SSH가 끊기면 프로세스가 죽습니다.**

```bash
# tmux 안에서 실행하면 접속이 끊겨도 살아 있습니다
tmux new -s train
python train.py 2>&1 | tee train.log
# Ctrl+B 누른 뒤 D  -> 빠져나오기
# 다시 들어갈 때: tmux attach -t train
```

### 3-7. 결과 회수 후 삭제

```bash
# 맥에서
rsync -avz -e "ssh -p <PORT>" root@<POD_IP>:/workspace/checkpoints ./
```

그 다음 **파드 삭제 + 볼륨 삭제**. 이 순서를 몸에 익히세요.

---

## 4. 이 장의 진짜 과제 — Claude에게 시켜 보기

위 절차를 손으로 한 번 해 보셨다면, **두 번째부터는 AI에게 시키세요.**

맥 터미널에서 Claude Code를 켜고 이렇게 말하면 됩니다.

```
RunPod 파드에 SSH로 붙어서 (접속 정보: root@1.2.3.4 -p 40022),
/workspace 에 있는 SelfStudy/Part3/Ch6/0601_MiniGPT.ipynb 를
스크립트로 바꿔서 GPU로 학습시켜줘.
n_layer=8, n_embd=512, block_size=256 으로 키우고,
VRAM 24GB 안에 들어가는 최대 배치 크기를 찾아서 써줘.
학습 로그는 파일로 남기고, 끝나면 체크포인트 크기를 알려줘.
```

**왜 이걸 꼭 해 봐야 하는가**

지금 하는 일이 바로 "원격 GPU 클러스터를 에이전트로 제어하는 일"의 축소판이기 때문입니다.
데이터센터가 완공되면 사람이 GPU를 한 장씩 만지지 않습니다.
**스크립트와 에이전트로 수천 장을 동시에 조종합니다.** 그 일의 입구가 여기입니다.

---

## 5. 스케일업 체크리스트

GPU에서 처음 돌릴 때 반드시 확인할 것들입니다.

### 배치 크기 찾기

```python
# OOM(메모리 부족)이 날 때까지 키워 보고 한 단계 내려서 씁니다
for bs in (8, 16, 32, 64, 128):
    try:
        x = torch.randint(0, vocab_size, (bs, block_size), device="cuda")
        model(x, x)[1].backward()
        print(f"batch {bs} OK, 사용 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"batch {bs} OOM")
        break
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
```

### GPU에서만 쓸 수 있는 가속

| 기법 | 코드 | 효과 |
|---|---|---|
| **혼합 정밀도** | `torch.autocast("cuda", dtype=torch.bfloat16)` | 메모리 절반, 속도 2배 |
| **TF32** | `torch.set_float32_matmul_precision("high")` | 행렬곱 가속, 한 줄 |
| **컴파일** | `model = torch.compile(model)` | 커널 융합, 1.3~2배 |
| **gradient accumulation** | 여러 스텝 모아 `opt.step()` | 메모리 없이 유효 배치 ↑ |
| **gradient checkpointing** | 활성값을 다시 계산 | 메모리 ↓, 속도 ↓ |

M1의 `mps` 에서는 이 중 상당수가 제한적이거나 효과가 없습니다.
**GPU에서 비로소 제대로 쓸 수 있는 도구들**입니다.

```python
# 전형적인 GPU 학습 루프
torch.set_float32_matmul_precision("high")
model = torch.compile(model).cuda()

for step in range(max_steps):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
```

### 체크포인트는 반드시

```python
if step % 1000 == 0:
    torch.save({"model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step}, f"ckpt_{step}.pt")
```

Community Cloud를 쓰거나 크레딧이 떨어지면 **파드가 통째로 사라집니다.**
저장하지 않은 학습은 없던 일이 됩니다.

---

## 6. 비용 감각

| 작업 | 대략의 GPU 시간 | 4090 기준 비용 |
|---|---|---|
| MiniGPT를 키워서 한국어 위키 학습 | 2~5시간 | 수천 원 |
| 7B 모델 LoRA 파인튜닝 1회 | 1~3시간 | 수천 원 |
| 7B 전체 파인튜닝 | A100 여러 장 × 수 시간 | 수만 원~ |
| GPT-3급 사전학습 | 수천 GPU × 수개월 | **수백만 달러** |

> 시간당 단가는 수시로 바뀌고 지역·수급에 따라 다릅니다. **콘솔에서 직접 확인**하세요.
> 위 표는 "실습은 만 원대, 사전학습은 개인이 못 함"이라는 **자릿수 감각**을 위한 것입니다.

**핵심 결론**: 개인이 사전학습을 처음부터 하는 건 불가능합니다.
그래서 실무는 **공개된 사전학습 모델을 가져와 파인튜닝**합니다. 그게 다음 단계입니다.

---

## 7. 자주 겪는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `CUDA out of memory` | 배치/문맥이 큼 | 배치 축소 → gradient accumulation |
| SSH 끊기면 학습 중단 | 포그라운드 실행 | `tmux` 사용 |
| GPU 사용률이 낮음 | 데이터 로딩 병목 | `num_workers`, `pin_memory=True` |
| 첫 스텝이 매우 느림 | `torch.compile` 컴파일 중 | 정상. 이후 빨라짐 |
| 재접속하니 파일이 없음 | 컨테이너 디스크에 저장 | `/workspace`(볼륨)에 저장 |
| 잔액이 0이 되며 전부 삭제 | 알림 무시 | 체크포인트를 주기적으로 회수 |

---

## 정리

- 클라우드 GPU는 **연산을 대신해 주는 원격 장비**. 내 맥은 리모컨
- **Auto-Recharge를 끄면** 최대 손실이 충전액으로 고정
- 파드만 지우지 말고 **볼륨까지** 지워야 과금이 끝남
- `get_device()` 패턴 덕분에 **코드 수정 없이** 맥 ↔ GPU 이동 가능
- `tmux` + 체크포인트는 선택이 아니라 필수
- **이 과정을 Claude에게 시켜 보는 것이 이 장의 진짜 실습**

## 다음

→ `0603_InContextLearning.ipynb` : 사전학습된 모델이 왜 예시 몇 개로 새 일을 하는가
