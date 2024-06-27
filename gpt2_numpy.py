#%%
import numpy as np
import tiktoken  # pip install tiktoken
import pickle
from IPython.display import clear_output


encoder = tiktoken.get_encoding("gpt2")
tokens = encoder.encode("Water was flowing fast in the river. He sat near the river bank")  # edit to start with a different prompt
num_tokens_to_generate = 20


with open("gpt2-neural-net-weights.pkl", "rb") as f:
    weights = pickle.load(f)

#%%

temperature = 3
block_size: int = 1024  # context length
vocab_size: int = 50257  # number of words in the tokenized vocabulary
n_layer: int = 12  # number of transformer layers
n_head: int = 12  # number of attention heads
n_embd: int = 768  # embedding dimension


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


for step in range(num_tokens_to_generate):
    idx = np.array(tokens, dtype=np.int32)[None, :]  # first index is the batch axis
    B, T = idx.shape  # batch size and sequence length

    wte = weights["transformer.wte.weight"]  # token embeddings
    wbe = weights["transformer.wpe.weight"]  # positional embeddings
    attention_mask = np.triu(np.full((T, T), -np.inf), 1)

    pos = np.arange(0, T)
    pos_emb = wbe[pos]
    tok_emb = wte[idx]

    x = tok_emb + pos_emb

    for i in range(n_layer):  # apply all blocks
        # layer norm 1
        gamma = weights[f"transformer.h.{i}.ln_1.weight"]
        beta = weights[f"transformer.h.{i}.ln_1.bias"]
        epsilon = 1e-5
        x2 = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + epsilon) * gamma + beta

        # causal self-attention
        c_attn = weights[f"transformer.h.{i}.attn.c_attn.weight"]
        c_attn_bias = weights[f"transformer.h.{i}.attn.c_attn.bias"]
        c_proj = weights[f"transformer.h.{i}.attn.c_proj.weight"]
        c_proj_bias = weights[f"transformer.h.{i}.attn.c_proj.bias"]

        qkv = x2 @ c_attn.T + c_attn_bias
        q, k, v = np.split(qkv, 3, axis=-1)
        k = k.reshape((B, T, n_head, n_embd // n_head)).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        q = q.reshape((B, T, n_head, n_embd // n_head)).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape((B, T, n_head, n_embd // n_head)).transpose(0, 2, 1, 3)  # (B, nh, T, hs)

        scale_factor = 1 / np.sqrt(q.shape[-1])
        attn_weight = q @ k.transpose(0, 1, 3, 2) * scale_factor
        attn_weight += attention_mask
        attn_weight = softmax(attn_weight)
        y = attn_weight @ v
        y = y.transpose(0, 2, 1, 3).reshape((B, T, n_embd))
        y = y @ c_proj.T + c_proj_bias

        x3 = x + y

        # layer norm 2
        gamma2 = weights[f"transformer.h.{i}.ln_2.weight"]
        beta2 = weights[f"transformer.h.{i}.ln_2.bias"]
        x4 = (x3 - x3.mean(axis=-1, keepdims=True)) / np.sqrt(x3.var(axis=-1, keepdims=True) + epsilon) * gamma2 + beta2

        # Multi-layer perceptron
        c_fc = weights[f"transformer.h.{i}.mlp.c_fc.weight"]
        c_fc_bias = weights[f"transformer.h.{i}.mlp.c_fc.bias"]
        c_proj = weights[f"transformer.h.{i}.mlp.c_proj.weight"]
        c_proj_bias = weights[f"transformer.h.{i}.mlp.c_proj.bias"]

        x5 = x4 @ c_fc.T + c_fc_bias
        x5 = 0.5 * x5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x5 + 0.044715 * x5**3)))  # gelu non-linearity
        x6 = x5 @ c_proj.T + c_proj_bias

        x = x3 + x6

    # layer norm final
    gamma = weights["transformer.ln_f.weight"]
    beta = weights["transformer.ln_f.bias"]
    x = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + epsilon) * gamma + beta

    # output projection
    lm_head = weights["lm_head.weight"]
    logits = x @ lm_head.T

    # sample the next token
    logits = logits[0, -1, :]
    probs = softmax(logits / temperature)  # convert to probabilities, apply temperature
    probs[np.argsort(probs)[:-50]] = 0  # only keeps the top 50 probs to sample from
    probs /= probs.sum()
    idx = np.random.choice(np.arange(vocab_size), p=probs)  # sample
    tokens.append(idx)
    # clear_output(wait=True)
print(encoder.decode(tokens))
