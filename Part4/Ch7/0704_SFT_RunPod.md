# 7-4. 실제 GPU에서 SFT 돌리기 (RunPod)

`0703_FullFineTuning.ipynb` 2절까지는 맥에서 돌아갑니다. 3절(진짜 베이스 모델)부터는
GPU가 필요합니다. 파트3의 [`Ch6/0602_ScaleUp_RunPod.md`](../../Part3/Ch6/0602_ScaleUp_RunPod.md)
에서 만든 환경을 그대로 씁니다.

---

## 1. GPU 고르기 — VRAM 계산이 먼저

`0703` 1절의 계산을 다시 적으면:

| 모델 | 전체 미세 조정 필요 VRAM | 최소 GPU |
|---|---|---|
| Qwen3-0.6B-Base | 약 8 GB | RTX 3090 / 4090 (24GB) 여유 |
| kanana-2-1.3b-base | 약 16 GB | 3090 / 4090 빠듯하게 가능 |
| Qwen3-4B-Base | 약 49 GB | A100 80GB, 또는 **LoRA** |
| 8B 급 | 약 98 GB | 다중 GPU, 또는 **LoRA/QLoRA** |

> 위 수치는 gradient checkpointing 을 켠 기준입니다. 끄면 활성값이 3배 가까이 늘어납니다.

**비용 감각** — 0.6B~1.3B 를 수만 건 지시 데이터로 2~3 에폭 돌리는 데
A40/A100 한 장으로 **1~3시간**, 대략 **1~2만원** 수준입니다.
(시세는 수시로 바뀝니다. 직접 확인하세요. 특정 서비스와 무관합니다.)

---

## 2. Pod 설정

- **템플릿**: `RunPod PyTorch 2.x` (CUDA 12.x)
- **GPU**: A40 48GB 또는 RTX 4090 24GB
- **Container Disk**: 40GB 이상 — 모델 체크포인트가 생각보다 큽니다
- **Volume**: 50GB 이상, `/workspace` 에 마운트
  모델·데이터셋 캐시를 여기 두면 Pod 을 껐다 켜도 다시 안 받습니다

```bash
# Pod 접속 후
export HF_HOME=/workspace/hf           # 캐시를 볼륨에 둔다 (중요)
export TOKENIZERS_PARALLELISM=false

pip install -U transformers datasets accelerate peft trl
nvidia-smi                             # VRAM 확인
```

---

## 3. 파일 올리기

```bash
# 로컬에서
cd ~/git/honglab-llm-selfstudy
scp -P <포트> -r Part4 root@<호스트>:/workspace/
```

또는 Pod 의 JupyterLab 에서 `Part4` 폴더째 드래그.

---

## 4. 훈련 스크립트

노트북 대신 스크립트로 돌리는 편이 안전합니다 — 브라우저가 끊겨도 훈련은 계속됩니다.

```bash
cd /workspace/Part4
nohup python train_sft.py > train.log 2>&1 &
tail -f train.log
```

`train_sft.py` 는 `0703` 의 `sft_train()` 과 **같은 함수**를 씁니다.
아래가 최소 형태입니다.

```python
import sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from chat import load_tokenizer

MODEL_ID = "Qwen/Qwen3-0.6B-Base"      # 또는 kakaocorp/kanana-2-1.3b-base
OUT      = "/workspace/out/sft-qwen3-0.6b"

tok = load_tokenizer(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable()
model.config.use_cache = False

# --- 데이터: 하나만 쓰지 말고 섞는다 (7-2 6절) ---
ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
data = [[{"role": "user", "content": r["instruction"]},
         {"role": "assistant", "content": r["output"]}] for r in ds.select(range(20000))]

# 멀티턴을 꼭 섞으세요 — 단일턴만 배우면 이어지는 질문에서 흐름을 놓칩니다
# ds2 = load_dataset("nayohan/141_korean_multi_session_dialogue", split="train")
# data += [...]

from Ch7_train import sft_train          # 0703 의 함수를 모듈로 빼서 재사용
sft_train(model, tok, data, epochs=3, lr=2e-5, micro_bs=4, accum=16, max_len=1024)

model.save_pretrained(OUT); tok.save_pretrained(OUT)
```

### 설정값 (0.6B~1.3B, 지시 데이터 수만 건)

```
learning_rate   1e-5 ~ 2e-5     크게 잡으면 사전훈련을 잊습니다 (catastrophic forgetting)
epochs          2 ~ 3           더 돌리면 데이터를 외웁니다
warmup_ratio    0.03
lr_scheduler    cosine
max_length      1024 ~ 2048
effective_bs    64 ~ 128        micro_bs × accum × GPU수
dtype           bfloat16        fp16 은 손실이 NaN 으로 터지는 경우가 있습니다
```

---

## 5. 훈련 중에 볼 것

| 신호 | 정상 | 이상하면 |
|---|---|---|
| train loss | 완만히 하강 | 급락 → 외우는 중(에폭 줄이기) / 발산 → LR 낮추기 |
| grad norm | 0.3 ~ 2 정도에서 안정 | 계속 clip 에 걸리면 LR 이 큼 |
| VRAM | 90% 근처 | 100% 치면 micro_bs 낮추고 accum 올리기 |
| 생성 샘플 | **스스로 멈추는가** | 안 멈추면 종료 토큰이 손실에서 빠졌는지 확인 |

**매 에폭 끝에 샘플을 생성해 보세요.** loss 숫자보다 이게 훨씬 많은 것을 말해 줍니다.

---

## 6. 끝난 뒤

```bash
# 결과만 내려받기 (0.6B bf16 기준 약 1.2GB)
scp -P <포트> -r root@<호스트>:/workspace/out/sft-qwen3-0.6b ./
```

**Pod 을 반드시 종료하세요.** 켜 둔 시간만큼 과금됩니다.
볼륨은 유지되므로 다음에 다시 붙이면 캐시가 그대로 있습니다.

---

## 7. 흔한 함정

| 증상 | 원인 |
|---|---|
| 답을 다 쓰고도 안 멈춤 | 종료 토큰을 손실에서 뺐음 |
| 모델이 질문까지 지어냄 | 손실 마스킹을 안 했음 |
| 손실이 비정상적으로 낮음 | 패딩 라벨을 IGNORE 로 안 두고 0 으로 둠 |
| 처음부터 NaN | fp16 사용 → bf16 으로 |
| 사전훈련 지식이 다 날아감 | 학습률이 큼 (1e-4 이상은 거의 항상 과함) |
| OOM 인데 배치는 작음 | `use_cache=True` 와 gradient checkpointing 충돌 |
| 토크나이저 불일치 오류 | 모델과 다른 토크나이저를 씀 — 반드시 짝으로 |

---

> **다음** → 챕터 8. 전체 미세 조정은 비쌉니다. [`../Ch8/`](../Ch8/) 에서
> **PEFT/LoRA** 로 같은 일을 개인 GPU에서 하는 법과, **페르소나**를 입히는 법을 다룹니다.
