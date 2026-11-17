___
<a href='https://honglab.ai'><p style="text-align:center;"><img src='https://lh3.googleusercontent.com/lY3ySXooSmwsq5r-mRi7uiypbo0Vez6pmNoQxMFhl9fmZJkRHu5lO2vo7se_0YOzgmDyJif9fi4_z0o3ZFdwd8NVSWG6Ea80uWaf3pOHpR4GHGDV7kaFeuHR3yAjIJjDgfXMxsvw=w2400'  class="center" width="50%" height="50%"/></p></a>
___
<center><em>Content Copyright by HongLab, Inc.</em></center>

# BPE 토크나이저

**해당 강의:** 파트3 챕터6 「파이토치 구현」

자주 붙어 나오는 바이트 쌍을 합쳐 가며 어휘를 만드는 BPE를 직접 구현합니다.
그다음 실습 자료에 들어 있는 한국어 동화용 BPE 토크나이저(어휘 8천 개)를 불러와, 강의에서 본 문장을 토큰으로 나눠 봅니다.

- 실행 시간: 몇 초 (CPU)
- 설치할 것: regex (`pip install regex`)
- 준비: 실습 자료의 `exp02_tinystories_Korean` 폴더 (「실습 자료의 토크나이저」부터 씁니다)

## 여섯 단어로 BPE 배우기

강의와 같은 여섯 단어가 데이터셋 전부라고 생각합니다. 모든 단어를 바이트 단위로 쪼개 놓고 시작합니다.
영어는 한 글자가 한 바이트입니다.

```python
import re
from collections import Counter

def piece_str(t):
    # 바이트 조각을 글자로 보여 줍니다. 글자 중간에서 잘린 바이트는 <16진수>로 씁니다.
    return re.sub(r"\\x([0-9a-f]{2})", r"<\1>", t.decode("utf-8", errors="backslashreplace"))

def show(seq):
    return " ".join(piece_str(t) for t in seq)

def count_pairs(seqs):
    pairs = Counter()
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            pairs[(a, b)] += 1
    return pairs

def merge(seq, a, b):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out

words = ["newest", "newest", "widest", "widest", "goes", "goes"]
seqs = [[bytes([b]) for b in w.encode("utf-8")] for w in words]
for s in seqs:
    print(show(s))
```

```text
n e w e s t
n e w e s t
w i d e s t
w i d e s t
g o e s
g o e s
```

나란히 붙은 쌍의 개수를 세고, 가장 많은 쌍을 하나로 합칩니다. 이것을 정한 횟수만큼 반복합니다.

```python
def train_bpe(seqs, n_merges, verbose=True):
    merges = []
    for step in range(1, n_merges + 1):
        pairs = count_pairs(seqs)
        (a, b), cnt = pairs.most_common(1)[0]
        seqs = [merge(s, a, b) for s in seqs]
        merges.append((a, b))
        if verbose:
            print(f"{step}. {show([a])} + {show([b])} → {show([a + b])}  ({cnt}번)")
    return seqs, merges

seqs_after, _ = train_bpe(seqs, 3)
print()
for s in seqs_after:
    print(show(s))

print("===========================")

seqs_after, _ = train_bpe(seqs, 2)
print()
for s in seqs_after:
    print(show(s))
```

```text
1. e + s → es  (6번)
2. es + t → est  (4번)
3. n + e → ne  (2번)

ne w est
ne w est
w i d est
w i d est
g o es
g o es
===========================
1. e + s → es  (6번)
2. es + t → est  (4번)

n e w est
n e w est
w i d est
w i d est
g o es
g o es
```

강의 화면과 같이 처음에는 `e`와 `s`가 6번으로 가장 많아 `es`가 되고, 다음에는 `es`와 `t`가 4번으로 `est`가 됩니다.
병합을 더 하면 단어 전체가 토큰 하나가 됩니다.

```python
seqs_after, _ = train_bpe(seqs, 8)
print()
for s in seqs_after[::2]:
    print(show(s))
```

```text
1. e + s → es  (6번)
2. es + t → est  (4번)
3. n + e → ne  (2번)
4. ne + w → new  (2번)
5. new + est → newest  (2번)
6. w + i → wi  (2번)
7. wi + d → wid  (2번)
8. wid + est → widest  (2번)

newest
widest
g o es
```

## 한글 — 바이트 여섯이 토큰 하나로

한글은 한 글자가 3바이트입니다. 「친구」는 글자로는 둘이지만 바이트로는 여섯입니다.

```python
bs = "친구".encode("utf-8")
print(list(bs), f"→ {len(bs)}바이트")
```

```text
[236, 185, 156, 234, 181, 172] → 6바이트
```

「친구」가 자주 나오는 짧은 글로 BPE를 돌려 봅니다. 띄어쓰기로 나눈 단어를 바이트로 쪼개 놓고 시작합니다.
글자 중간에서 잘린 바이트 조각은 `<16진수>`로 표시합니다.

```python
corpus = '''
나는 친구와 놀았다
친구가 집에 왔다
친구를 만나서 기뻤다
나는 친구에게 편지를 썼다
친구와 야구를 했다
친구가 구름을 보았다
'''
ko_seqs = [[bytes([b]) for b in w.encode("utf-8")] for w in corpus.split()]
print("「친구와」의 처음 모습:", show(ko_seqs[1]))
print()
ko_after, _ = train_bpe(ko_seqs, 5)
print()
for s in ko_after[1:5]:
    print(show(s))
```

```text
「친구와」의 처음 모습: <ec> <b9> <9c> <ea> <b5> <ac> <ec> <99> <80>

1. <ea> + <b5> → <ea><b5>  (8번)
2. <ea><b5> + <ac> → 구  (8번)
3. <ec> + <b9> → <ec><b9>  (6번)
4. <ec><b9> + <9c> → 친  (6번)
5. 친 + 구 → 친구  (6번)

친구 <ec> <99> <80>
<eb> <86> <80> <ec> <95> <98> <eb> <8b> <a4>
친구 <ea> <b0> <80>
<ec> <a7> <91> <ec> <97> <90>
```

자주 붙어 다니는 바이트 쌍부터 합치다 보니 「구」와 「친」이 각각 조각이 되고, 「친구」도 자주 나오니 둘이 다시 붙어 토큰 하나가 됩니다.
BPE가 글자 경계를 알아서 그런 것이 아닙니다. 자주 붙어 다니는 바이트끼리 뭉쳤을 뿐입니다.

## 실습 자료의 토크나이저

실습 자료에는 한국어 동화 데이터로 학습한 BPE 토크나이저가 들어 있습니다. 어휘 8천 개짜리와 1만 6천 개짜리 두 가지입니다.
파일에는 바이트 256개와 특수 토큰, 그리고 학습한 병합 규칙이 순서대로 저장돼 있습니다.

```python
from pathlib import Path

# 「실습 자료 다운로드」 레슨에서 받은 실습 자료 중 exp02_tinystories_Korean 폴더의 위치입니다.
# 이 노트북 폴더 안에 옮겨 두었다면 그대로 두고, 다른 곳에 있다면 경로를 고치세요.
EXP02 = Path("exp02_tinystories_Korean")
assert (EXP02 / "tokenizer").exists(), f"{EXP02.resolve()}가 없습니다. 실습 자료의 exp02_tinystories_Korean 폴더 경로를 확인하세요."

import json

obj = json.load(open(EXP02 / "tokenizer" / "ko_tinystories_bpe_8k.json", encoding="utf-8"))
print("어휘 크기:", len(obj["vocab"]))
print("특수 토큰:", obj["special_tokens"])
print("병합 규칙 수:", len(obj["merges"]))
```

```text
어휘 크기: 8000
특수 토큰: ['<|endoftext|>']
병합 규칙 수: 7743
```

실제 토크나이저는 문장을 바로 바이트로 쪼개지 않고, 먼저 정규식으로 단어 조각을 나눕니다(GPT-2와 같은 규칙).
단어 앞의 띄어쓰기는 그 단어에 붙습니다(「 오늘」). 그다음 조각마다 학습한 병합 규칙을 먼저 배운 것부터 적용합니다.
`\p{L}`(글자) 같은 유니코드 분류를 쓰기 때문에 표준 `re` 대신 `regex` 모듈을 씁니다.

```python
import regex as re

PAT = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

class BPETokenizer:
    def __init__(self, path):
        obj = json.load(open(path, encoding="utf-8"))
        self.vocab = {int(i): bytes.fromhex(h) for i, h in obj["vocab"].items()}   # 번호 → 바이트
        self.token_to_id = {b: i for i, b in self.vocab.items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in obj["merges"]]
        self.rank = {pair: r for r, pair in enumerate(merges)}                      # 먼저 배운 병합일수록 작은 수
        self.cache = {}

    def encode_piece(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        parts = [bytes([b]) for b in piece.encode("utf-8")]
        while len(parts) > 1:
            # 이웃한 쌍 가운데 가장 먼저 배운 병합을 찾아 합칩니다
            ranks = [self.rank.get(pair, float("inf")) for pair in zip(parts, parts[1:])]
            best = min(range(len(ranks)), key=ranks.__getitem__)
            if ranks[best] == float("inf"):
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        ids = [self.token_to_id[p] for p in parts]
        self.cache[piece] = ids
        return ids

    def encode(self, text):
        return [i for piece in PAT.findall(text) for i in self.encode_piece(piece)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

tok8k = BPETokenizer(EXP02 / "tokenizer" / "ko_tinystories_bpe_8k.json")
```

강의 화면의 문장을 토큰으로 나눠 봅니다.

```python
text = "나는 오늘 친구와 영화를 보러 갔다"
print("사전 분할:", PAT.findall(text))

ids = tok8k.encode(text)
print("토큰:", [tok8k.decode([i]) for i in ids])
print("토큰 ID:", ids)
print("되돌리기:", tok8k.decode(ids))
```

```text
사전 분할: ['나는', ' 오늘', ' 친구와', ' 영화를', ' 보러', ' 갔다']
토큰: ['나는', ' 오늘', ' 친구와', ' 영', '화를', ' 보러', ' 갔다']
토큰 ID: [788, 3437, 1482, 4353, 7807, 4792, 1393]
되돌리기: 나는 오늘 친구와 영화를 보러 갔다
```

강의 화면과 같은 일곱 토큰, 같은 번호가 나옵니다. 「영화를」은 「 영」과 「화를」 두 조각으로 나뉩니다.
BPE는 문법을 보지 않고 데이터에서 자주 붙어 나온 바이트만 보기 때문에, 이렇게 우리 눈에 어색한 조각이 생기기도 합니다.

처음 보는 말도 아는 조각으로 나눠 받아들일 수 있습니다. 신조어와 영어를 넣어 봅니다.

```python
def pieces(tok, ids):
    # 토큰마다 바이트를 글자로 보여 줍니다. 글자 중간에서 잘린 조각은 <16진수>로 씁니다.
    return [piece_str(tok.vocab[i]) for i in ids]

for t in ["오늘은 갓생 살기", "Transformer는 어텐션을 쓴다"]:
    print(f"{t!r} → {pieces(tok8k, tok8k.encode(t))}")
```

```text
'오늘은 갓생 살기' → ['오늘', '은', ' <ea><b0>', '<93>', '생', ' 살', '기']
'Transformer는 어텐션을 쓴다' → ['T', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'e', 'r', '는', ' 어', '<ed><85>', '<90>', '<ec>', '<85>', '<98>', '을', ' <ec><93>', '<b4>다']
```

동화 데이터에 없던 「갓」은 바이트 조각으로 나뉘고, 영어는 거의 본 적이 없어서 알파벳 하나하나로 나뉩니다.
토큰 수는 늘어나지만, 바이트에서 시작하므로 모르는 글자 때문에 막히는 일은 없습니다.

## 병합 기록 들여다보기

어휘의 앞 256개는 바이트 하나씩이고, 그 뒤가 특수 토큰, 그 뒤가 학습한 병합 결과입니다.
처음 배운 병합과 마지막에 배운 병합을 봅니다.

```python
first = 257                                         # 바이트 256개 + 특수 토큰 1개 다음부터
print("처음 배운 병합:", pieces(tok8k, range(first, first + 15)))
print("마지막에 배운 병합:", pieces(tok8k, range(7985, 8000)))
```

```text
처음 배운 병합: [' <ec>', ' <eb>', '<ec><9d>', '<eb><8b>', '<eb><8a>', '는', '다', ' <ea>', ' <ed>', '<ec><97>', '이', '<96><b4>', '어', '<95><84>', '<88>다']
마지막에 배운 병합: [' 타기로', ' 가져가려', '진이는', ' 내일', ' 쥐었어요', ' 저물어', ' 가리키는', ' 사라지자', ' 삐약', ' 많다는', '<9f><81>', ' 날갯짓하는', ' 졸랐어요', ' 탄', ' 광경']
```

처음에는 한글 한 글자를 이루는 바이트 조각과 「는」, 「다」, 「이」 같은 자주 쓰는 글자가 만들어지고, 나중에는 「 가리키는」, 「 졸랐어요」 같은 긴 덩어리가 만들어집니다.

## 어휘 크기 — 8천 대 1만 6천

어휘를 키우면 같은 글이 더 적은 토큰으로 나뉘지만, 모델의 임베딩 표가 커집니다.
실습 자료의 검증용 동화(`valid.txt`) 앞부분 10만 자로 비교합니다. 임베딩 표 크기는 강의 화면처럼 벡터 크기 512로 계산합니다.

```python
tok16k = BPETokenizer(EXP02 / "tokenizer" / "ko_tinystories_bpe_16k.json")

sample = (EXP02 / "dataset" / "ko_tinystories" / "valid.txt").read_text(encoding="utf-8")[:100_000]
sample = sample.replace("<|endoftext|>", "")        # 이야기 경계 표시는 빼고 셉니다

for name, t in [("8천", tok8k), ("1만 6천", tok16k)]:
    n = len(t.encode(sample))
    print(f"어휘 {name:>5}: 토큰 {n:>7,}개 · 토큰당 {len(sample) / n:.2f}자 · 임베딩 표 {len(t.vocab) * 512 / 1e6:.1f}M개")

print()
print("1만 6천 어휘로 같은 문장:", pieces(tok16k, tok16k.encode(text)))
```

```text
어휘    8천: 토큰  32,725개 · 토큰당 2.93자 · 임베딩 표 4.1M개
어휘 1만 6천: 토큰  31,205개 · 토큰당 3.07자 · 임베딩 표 8.2M개

1만 6천 어휘로 같은 문장: ['나는', ' 오늘', ' 친구와', ' 영', '화를', ' 보러', ' 갔다']
```

이 동화 데이터에서는 어휘를 두 배로 늘려도 토큰 수가 5%쯤 줄어드는 데 그치고, 임베딩 표는 두 배가 됩니다.
쉬운 낱말이 되풀이되는 동화라서 8천 개로도 자주 쓰는 말이 대부분 토큰 하나가 되기 때문으로 보입니다.
어휘 크기는 데이터에 맞춰 정합니다. 여러 분야의 글을 다루는 큰 모델은 수만에서 수십만 개를 씁니다.

## 정리

- BPE는 바이트에서 시작해 가장 자주 붙어 나오는 쌍을 하나씩 합쳐 어휘를 만듭니다.
- 한글은 한 글자가 3바이트지만, 자주 쓰는 말은 여러 글자가 토큰 하나로 뭉칩니다.
- 문법을 보지 않으므로 「영 + 화를」처럼 어색한 조각도 생기지만, 처음 보는 말과 다른 언어도 같은 방식으로 처리합니다.
- 어휘가 크면 문장이 짧아지고 임베딩 표가 커집니다.

다음 노트북에서는 이 토크나이저로 작은 GPT를 만들어 훈련합니다.
