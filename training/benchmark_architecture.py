import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# TinyChat-F427 — LOCKED ARCHITECTURE
# ============================================================

VOCAB_SIZE = 1024
D_MODEL = 144
N_LAYERS = 6

N_Q_HEADS = 4
N_KV_HEADS = 1
HEAD_DIM = 36

D_FF = 288
CONTEXT = 128

torch.set_num_threads(6)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


# ============================================================
# RoPE
# ============================================================

def apply_rope(x):
    # x: [B, heads, T, head_dim]

    B, H, T, D = x.shape

    half = D // 2

    positions = torch.arange(
        T,
        device=x.device,
        dtype=x.dtype
    )

    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half,
            device=x.device,
            dtype=x.dtype
        )
        / half
    )

    angles = positions[:, None] * frequencies[None, :]

    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:]

    rotated = torch.cat(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ],
        dim=-1
    )

    return rotated


# ============================================================
# MULTI-QUERY ATTENTION
# ============================================================

class MQA(nn.Module):

    def __init__(self):
        super().__init__()

        self.wq = nn.Linear(
            D_MODEL,
            N_Q_HEADS * HEAD_DIM,
            bias=False
        )

        self.wk = nn.Linear(
            D_MODEL,
            N_KV_HEADS * HEAD_DIM,
            bias=False
        )

        self.wv = nn.Linear(
            D_MODEL,
            N_KV_HEADS * HEAD_DIM,
            bias=False
        )

        self.wo = nn.Linear(
            N_Q_HEADS * HEAD_DIM,
            D_MODEL,
            bias=False
        )


    def forward(self, x):

        B, T, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(
            B,
            T,
            N_Q_HEADS,
            HEAD_DIM
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            N_KV_HEADS,
            HEAD_DIM
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            N_KV_HEADS,
            HEAD_DIM
        ).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        # Shared K/V for all four query heads
        k = k.expand(
            B,
            N_Q_HEADS,
            T,
            HEAD_DIM
        )

        v = v.expand(
            B,
            N_Q_HEADS,
            T,
            HEAD_DIM
        )

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(HEAD_DIM)

        mask = torch.triu(
            torch.ones(
                T,
                T,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        out = attention @ v

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            T,
            N_Q_HEADS * HEAD_DIM
        )

        return self.wo(out)


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(self):
        super().__init__()

        self.w_gate = nn.Linear(
            D_MODEL,
            D_FF,
            bias=False
        )

        self.w_up = nn.Linear(
            D_MODEL,
            D_FF,
            bias=False
        )

        self.w_down = nn.Linear(
            D_FF,
            D_MODEL,
            bias=False
        )


    def forward(self, x):

        gate = F.silu(
            self.w_gate(x)
        )

        up = self.w_up(x)

        return self.w_down(
            gate * up
        )


# ============================================================
# Transformer Block
# ============================================================

class Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm1 = RMSNorm(D_MODEL)
        self.attn = MQA()

        self.norm2 = RMSNorm(D_MODEL)
        self.ffn = SwiGLU()


    def forward(self, x):

        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# TinyChat-F427
# ============================================================

class TinyChatF427(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL
        )

        self.blocks = nn.ModuleList(
            [
                Block()
                for _ in range(N_LAYERS)
            ]
        )

        self.final_norm = RMSNorm(
            D_MODEL
        )


    def forward(self, tokens):

        x = self.embedding(tokens)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # tied LM head
        logits = F.linear(
            x,
            self.embedding.weight
        )

        return logits


# ============================================================
# BUILD + VERIFY
# ============================================================

torch.manual_seed(42)

model = TinyChatF427().to(DEVICE)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print()
print("==============================================")
print(" TinyChat-F427 : ARCHITECTURE VERIFICATION")
print("==============================================")
print()
print("Vocabulary       :", VOCAB_SIZE)
print("d_model          :", D_MODEL)
print("Layers           :", N_LAYERS)
print("Query heads      :", N_Q_HEADS)
print("KV heads         :", N_KV_HEADS)
print("Head dimension   :", HEAD_DIM)
print("FFN dimension    :", D_FF)
print("Context          :", CONTEXT)
print()
print("PARAMETERS       :", parameter_count)
print("Expected         : 1206864")

if parameter_count != 1206864:
    raise RuntimeError(
        f"ARCHITECTURE PARAMETER ERROR: {parameter_count}"
    )

print("PARAMETER CHECK  : PASS")
print()


# ============================================================
# TRAINING BENCHMARK
# ============================================================

def benchmark_batch(batch_size):

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )

    x = torch.randint(
        0,
        VOCAB_SIZE,
        (
            batch_size,
            CONTEXT
        ),
        device=DEVICE
    )

    y = torch.randint(
        0,
        VOCAB_SIZE,
        (
            batch_size,
            CONTEXT
        ),
        device=DEVICE
    )


    # warmup
    for _ in range(2):

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE
            ),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()


    steps = 5

    start = time.perf_counter()

    for _ in range(steps):

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE
            ),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    tokens = (
        batch_size
        *
        CONTEXT
        *
        steps
    )

    tokens_per_second = (
        tokens / elapsed
    )

    return (
        elapsed / steps,
        tokens_per_second,
        float(loss.item())
    )


print("==============================================")
print(" CPU TRAINING BENCHMARK")
print("==============================================")
print()

results = []

for batch in [4, 8, 16, 32]:

    try:

        sec_step, tok_sec, loss = (
            benchmark_batch(batch)
        )

        results.append(
            (
                batch,
                sec_step,
                tok_sec
            )
        )

        print(
            f"Batch {batch:2d} : "
            f"{sec_step:.3f} s/step | "
            f"{tok_sec:.1f} tokens/s | "
            f"loss {loss:.4f}"
        )

    except RuntimeError as e:

        print(
            f"Batch {batch:2d} : FAILED | {e}"
        )


best = max(
    results,
    key=lambda x: x[2]
)

print()
print("----------------------------------------------")
print("BEST BATCH       :", best[0])
print("TOKENS/SECOND    :", f"{best[2]:.1f}")

for tokens in [
    1_000_000,
    5_000_000,
    10_000_000,
    20_000_000
]:

    seconds = (
        tokens / best[2]
    )

    print(
        f"{tokens:>10,d} tokens : "
        f"{seconds/60:.2f} min "
        f"({seconds/3600:.2f} h)"
    )

print()
print("ARCHITECTURE LOCK : PASS")
print("==============================================")
