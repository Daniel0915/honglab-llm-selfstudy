"""Part4 공용 유틸리티 — 대화 형식, 손실 마스킹, 배치 만들기.

Part2/Part3의 `common.py` 와 같은 역할입니다. 노트북 첫 셀에서
sys.path 에 Part4 폴더를 추가한 뒤 `from chat import ...` 로 씁니다.

여기 있는 것은 전부 "SFT 데이터 한 건이 모델 입력이 되기까지"에 필요한 것들입니다.
핵심은 세 가지뿐입니다.

    1. 대화를 ChatML 문자열로 감싼다          -> render_chatml()
    2. assistant 응답 + 종료 토큰에만 손실    -> build_sft_example()
    3. 패딩 자리는 라벨을 IGNORE 로 둔다      -> collate()
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch

# 손실에서 제외할 라벨 값. PyTorch cross_entropy 의 기본 ignore_index 가 -100 입니다.
IGNORE_INDEX = -100

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# ------------------------------------------------------------------ 대화 형식

def render_message(role: str, content: str) -> str:
    """메시지 한 개를 ChatML 한 블록으로 감쌉니다.

    <|im_start|>role\n내용<|im_end|>\n

    역할마다 틀이 다르지 않습니다 — system/user/assistant 전부 같은 모양이고
    역할 이름만 바뀝니다. (챕터7 퀴즈 20번)
    """
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def render_chatml(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """대화 전체를 ChatML 문자열로 만듭니다.

    add_generation_prompt=True 면 끝에 `<|im_start|>assistant\n` 까지만 붙입니다.
    대화할 때 이 부분은 **우리가 붙여 주고** 모델은 그 뒤부터 씁니다.
    그래서 훈련할 때도 이 머리 부분은 손실에서 뺍니다. (퀴즈 12번)
    """
    out = "".join(render_message(m["role"], m["content"]) for m in messages)
    if add_generation_prompt:
        out += f"{IM_START}assistant\n"
    return out


# ------------------------------------------------------------------ 손실 마스킹

@dataclass
class SFTExample:
    input_ids: List[int]
    labels: List[int]

    def __len__(self):
        return len(self.input_ids)

    @property
    def n_supervised(self) -> int:
        """실제로 손실이 걸리는 토큰 수."""
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def build_sft_example(tokenizer, messages: List[Dict[str, str]],
                      mask_prompt: bool = True) -> SFTExample:
    """대화 한 건을 (input_ids, labels) 로 바꿉니다.

    mask_prompt=True  : assistant 응답 + 종료 토큰에만 손실    <- SFT 의 기본
    mask_prompt=False : 전체 토큰에 손실                       <- 비교용

    가린다는 것은 「못 맞혔다고 혼내지 않는다」는 뜻이지
    「보여주지 않는다」가 아닙니다. input_ids 에는 전부 그대로 들어갑니다.
    """
    input_ids: List[int] = []
    labels: List[int] = []

    def extend(text: str, supervised: bool):
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        labels.extend(ids if supervised else [IGNORE_INDEX] * len(ids))

    for m in messages:
        role, content = m["role"], m["content"]
        if role == "assistant" and mask_prompt:
            # 머리(<|im_start|>assistant\n)는 가리고, 내용 + 종료 토큰에만 손실.
            extend(f"{IM_START}assistant\n", supervised=False)
            extend(f"{content}{IM_END}\n", supervised=True)
        else:
            extend(render_message(role, content), supervised=not mask_prompt)

    return SFTExample(input_ids=input_ids, labels=labels)


def preview_mask(tokenizer, ex: SFTExample, width: int = 88) -> str:
    """손실이 걸리는 토큰만 글자로, 가린 토큰은 ░ 로 그려 봅니다.

    토큰 하나씩 디코딩하면 한글이 깨지므로(UTF-8 한 글자 = 여러 바이트/토큰),
    같은 상태가 이어지는 구간을 **묶어서** 디코딩합니다.
    """
    pieces = []
    run, run_sup = [], None
    def flush():
        if not run:
            return
        text = tokenizer.decode(run)
        pieces.append(text if run_sup else "░" * len(text.strip() or " "))
    for tid, lab in zip(ex.input_ids, ex.labels):
        sup = lab != IGNORE_INDEX
        if sup is not run_sup and run:
            flush(); run.clear()
        run_sup = sup
        run.append(tid)
    flush()
    text = "".join(pieces).replace("\n", "⏎")
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


# ------------------------------------------------------------------ 배치

def collate(batch: List[SFTExample], pad_id: int, max_len: Optional[int] = None):
    """여러 샘플을 하나의 배치로 묶습니다.

    패딩 자리는 input_ids 는 pad_id, **labels 는 IGNORE_INDEX** 입니다. (퀴즈 22번)
    0 으로 두면 0번 토큰을 맞히라고 가르치는 셈이 됩니다.
    """
    lens = [len(e) for e in batch]
    n = max_len or max(lens)
    ids, labs, attn = [], [], []
    for e in batch:
        pad = n - len(e)
        if pad < 0:
            ids.append(e.input_ids[:n]); labs.append(e.labels[:n]); attn.append([1] * n)
            continue
        ids.append(e.input_ids + [pad_id] * pad)
        labs.append(e.labels + [IGNORE_INDEX] * pad)
        attn.append([1] * len(e) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def split_long_conversation(messages: List[Dict[str, str]], tokenizer,
                            max_len: int) -> List[List[Dict[str, str]]]:
    """컨텍스트 상한을 넘는 긴 대화를 **문답 쌍이 완결된 경계**에서 나눕니다. (퀴즈 21번)

    길이에서 그냥 자르면 답이 중간에 끊긴 채 「여기서 멈춰라」를 가르치게 됩니다.
    """
    system = [m for m in messages if m["role"] == "system"][:1]
    turns = [m for m in messages if m["role"] != "system"]

    chunks, cur = [], list(system)
    for i in range(0, len(turns), 2):          # user, assistant 를 한 쌍으로
        pair = turns[i:i + 2]
        trial = cur + pair
        n = len(tokenizer.encode(render_chatml(trial), add_special_tokens=False))
        if n > max_len and len(cur) > len(system):
            chunks.append(cur)
            cur = list(system) + pair
        else:
            cur = trial
    if len(cur) > len(system):
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------ 토크나이저

class ByteTokenizer:
    """transformers 없이도 노트북이 돌아가도록 만든 **대역용** 토크나이저.

    UTF-8 바이트 하나를 토큰 하나로 씁니다(0~255). 그 위에 ChatML 특수 토큰을
    256번부터 붙였습니다. 진짜 BPE가 아니라서 토큰 수가 훨씬 많지만,
    **손실 마스킹과 배치 구성의 동작은 똑같이 확인할 수 있습니다.**

    실제 훈련에서는 반드시 베이스 모델에 딸린 토크나이저를 쓰세요.
    모델과 토크나이저가 어긋나면 임베딩 표의 번호가 전부 틀어집니다.
    """

    SPECIALS = [IM_START, IM_END, "<|pad|>"]

    def __init__(self):
        self.specials = {s: 256 + i for i, s in enumerate(self.SPECIALS)}
        self.inv = {v: k for k, v in self.specials.items()}
        self.vocab_size = 256 + len(self.SPECIALS)
        self.pad_token_id = self.specials["<|pad|>"]
        self.eos_token_id = self.specials[IM_END]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids: List[int] = []
        i = 0
        while i < len(text):
            for s, sid in self.specials.items():
                if text.startswith(s, i):
                    ids.append(sid); i += len(s); break
            else:
                ids.extend(text[i].encode("utf-8")); i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        out, buf = [], bytearray()
        for i in ids:
            if i in self.inv:
                if buf: out.append(buf.decode("utf-8", "replace")); buf = bytearray()
                out.append(self.inv[i])
            else:
                buf.append(i)
        if buf: out.append(buf.decode("utf-8", "replace"))
        return "".join(out)


def load_tokenizer(model_id: str = "Qwen/Qwen3-0.6B-Base", quiet: bool = False):
    """있으면 진짜 토크나이저를, 없으면 ByteTokenizer 를 돌려줍니다."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if not quiet:
            print(f"tokenizer = {model_id}  (vocab {tok.vocab_size:,})")
        return tok
    except Exception as e:
        if not quiet:
            print(f"transformers 토크나이저를 못 불러와 ByteTokenizer 로 대체합니다. ({type(e).__name__})")
        return ByteTokenizer()
