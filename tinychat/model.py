import math
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 1024
D_MODEL = 144
N_LAYERS = 6

N_Q_HEADS = 4
N_KV_HEADS = 1
HEAD_DIM = 36

D_FF = 288
CONTEXT = 128


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


def apply_rope(x):
    _, _, T, D = x.shape
    half = D // 2

    positions = torch.arange(
        T, device=x.device, dtype=x.dtype
    )

    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half, device=x.device, dtype=x.dtype
        )
        / half
    )

    angles = positions[:, None] * frequencies[None, :]

    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat(
        (
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ),
        dim=-1,
    )


class MQA(nn.Module):
    def __init__(self):
        super().__init__()

        self.wq = nn.Linear(
            D_MODEL, N_Q_HEADS * HEAD_DIM, bias=False
        )

        self.wk = nn.Linear(
            D_MODEL, N_KV_HEADS * HEAD_DIM, bias=False
        )

        self.wv = nn.Linear(
            D_MODEL, N_KV_HEADS * HEAD_DIM, bias=False
        )

        self.wo = nn.Linear(
            N_Q_HEADS * HEAD_DIM, D_MODEL, bias=False
        )

    def forward(self, x):
        B, T, _ = x.shape

        q = self.wq(x).view(
            B, T, N_Q_HEADS, HEAD_DIM
        ).transpose(1, 2)

        k = self.wk(x).view(
            B, T, N_KV_HEADS, HEAD_DIM
        ).transpose(1, 2)

        v = self.wv(x).view(
            B, T, N_KV_HEADS, HEAD_DIM
        ).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        k = k.expand(B, N_Q_HEADS, T, HEAD_DIM)
        v = v.expand(B, N_Q_HEADS, T, HEAD_DIM)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(HEAD_DIM)

        mask = torch.triu(
            torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            mask, float("-inf")
        )

        attention = F.softmax(scores, dim=-1)

        out = attention @ v

        out = out.transpose(1, 2).contiguous()
        out = out.view(
            B, T, N_Q_HEADS * HEAD_DIM
        )

        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_gate = nn.Linear(
            D_MODEL, D_FF, bias=False
        )

        self.w_up = nn.Linear(
            D_MODEL, D_FF, bias=False
        )

        self.w_down = nn.Linear(
            D_FF, D_MODEL, bias=False
        )

    def forward(self, x):
        return self.w_down(
            F.silu(self.w_gate(x))
            * self.w_up(x)
        )


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = RMSNorm(D_MODEL)
        self.attn = MQA()

        self.norm2 = RMSNorm(D_MODEL)
        self.ffn = SwiGLU()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyChatF427(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE, D_MODEL
        )

        self.blocks = nn.ModuleList(
            Block() for _ in range(N_LAYERS)
        )

        self.final_norm = RMSNorm(D_MODEL)

    def forward(self, tokens):
        x = self.embedding(tokens)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # tied embedding / LM output head
        return F.linear(
            x,
            self.embedding.weight
        )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )
