import math
import time

import numpy as np

# BMASS Tiny Transformer
# One causal self-attention block, trained from scratch.
# NumPy only; designed for very small CPU/RAM systems.

TEXT = (
    "bmass is a small local machine. "
    "bmass runs models locally. "
    "a local model can learn from local examples. "
    "small systems can be useful, private, and efficient. "
    "the machine observes, acts, measures, and learns. "
    "we cultivate intelligence one careful step at a time. "
) * 40

SEED = 7
CONTEXT = 24
D_MODEL = 16
D_FF = 32
STEPS = 1200
LEARNING_RATE = 0.03
PRINT_EVERY = 100
GENERATE_CHARS = 220
WEIGHTS_FILE = "tiny_transformer_weights.npz"

rng = np.random.default_rng(SEED)
chars = sorted(set(TEXT))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = np.array([stoi[ch] for ch in TEXT], dtype=np.int64)

V = len(chars)
T = CONTEXT
D = D_MODEL
F = D_FF
scale = 0.08

# Trainable parameters
E = rng.normal(0, scale, (V, D))
P = rng.normal(0, scale, (T, D))
Wq = rng.normal(0, scale, (D, D))
Wk = rng.normal(0, scale, (D, D))
Wv = rng.normal(0, scale, (D, D))
Wo = rng.normal(0, scale, (D, D))
W1 = rng.normal(0, scale, (D, F))
b1 = np.zeros(F)
W2 = rng.normal(0, scale, (F, D))
b2 = np.zeros(D)
Wout = rng.normal(0, scale, (D, V))
bout = np.zeros(V)

params = [E, P, Wq, Wk, Wv, Wo, W1, b1, W2, b2, Wout, bout]
param_names = [
    "E", "P", "Wq", "Wk", "Wv", "Wo",
    "W1", "b1", "W2", "b2", "Wout", "bout"
]

causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def forward(x_ids, y_ids=None):
    n = len(x_ids)

    X = E[x_ids] + P[:n]

    Q = X @ Wq
    K = X @ Wk
    values = X @ Wv

    scores = (Q @ K.T) / math.sqrt(D)
    local_mask = causal_mask[:n, :n]
    scores = np.where(local_mask, -1e9, scores)

    attention = softmax(scores, axis=1)
    attended = attention @ values

    attention_output = attended @ Wo
    residual = X + attention_output

    hidden_pre = residual @ W1 + b1
    hidden = np.tanh(hidden_pre)
    feedforward = hidden @ W2 + b2

    block_output = residual + feedforward
    logits = block_output @ Wout + bout
    probabilities = softmax(logits, axis=1)

    cache = (
        x_ids, X, Q, K, values, attention, attended,
        residual, hidden, block_output, probabilities
    )

    if y_ids is None:
        return probabilities, cache

    loss = -np.mean(
        np.log(probabilities[np.arange(n), y_ids] + 1e-12)
    )
    return loss, cache


def backward(cache, y_ids):
    (
        x_ids, X, Q, K, values, attention, attended,
        residual, hidden, block_output, probabilities
    ) = cache

    n = len(x_ids)
    grads = [np.zeros_like(parameter) for parameter in params]

    (
        gE, gP, gWq, gWk, gWv, gWo,
        gW1, gb1, gW2, gb2, gWout, gbout
    ) = grads

    dlogits = probabilities.copy()
    dlogits[np.arange(n), y_ids] -= 1.0
    dlogits /= n

    gWout[:] = block_output.T @ dlogits
    gbout[:] = np.sum(dlogits, axis=0)
    dblock = dlogits @ Wout.T

    dresidual = dblock.copy()
    dfeedforward = dblock

    gW2[:] = hidden.T @ dfeedforward
    gb2[:] = np.sum(dfeedforward, axis=0)
    dhidden = dfeedforward @ W2.T

    dhidden_pre = dhidden * (1.0 - hidden * hidden)
    gW1[:] = residual.T @ dhidden_pre
    gb1[:] = np.sum(dhidden_pre, axis=0)
    dresidual += dhidden_pre @ W1.T

    dX = dresidual.copy()
    dattention_output = dresidual

    gWo[:] = attended.T @ dattention_output
    dattended = dattention_output @ Wo.T

    dattention = dattended @ values.T
    dvalues = attention.T @ dattended

    gWv[:] = X.T @ dvalues
    dX += dvalues @ Wv.T

    dscores = attention * (
        dattention
        - np.sum(dattention * attention, axis=1, keepdims=True)
    )

    local_mask = causal_mask[:n, :n]
    dscores[local_mask] = 0.0

    dQ = (dscores @ K) / math.sqrt(D)
    dK = (dscores.T @ Q) / math.sqrt(D)

    gWq[:] = X.T @ dQ
    gWk[:] = X.T @ dK

    dX += dQ @ Wq.T
    dX += dK @ Wk.T

    np.add.at(gE, x_ids, dX)
    gP[:n] = dX

    return grads


def clip_gradients(grads, max_norm=1.0):
    total_norm = math.sqrt(
        sum(float(np.sum(gradient * gradient)) for gradient in grads)
    )

    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + 1e-12)
        for gradient in grads:
            gradient *= scale_factor

    return total_norm


def sample_window():
    start = int(rng.integers(0, len(data) - T - 1))
    x = data[start:start + T]
    y = data[start + 1:start + T + 1]
    return x, y


def generate(seed="bmass ", n_chars=GENERATE_CHARS, temperature=0.8):
    ids = [stoi.get(character, 0) for character in seed]

    for _ in range(n_chars):
        context = ids[-T:]
        probabilities, _ = forward(
            np.array(context, dtype=np.int64)
        )

        next_probabilities = probabilities[-1]
        scaled_logits = (
            np.log(next_probabilities + 1e-12) / temperature
        )
        scaled_probabilities = np.exp(
            scaled_logits - np.max(scaled_logits)
        )
        scaled_probabilities /= scaled_probabilities.sum()

        next_id = int(rng.choice(V, p=scaled_probabilities))
        ids.append(next_id)

    return "".join(itos[index] for index in ids)


def save_weights():
    payload = {
        name: value
        for name, value in zip(param_names, params)
    }

    payload["chars"] = np.array(chars)
    payload["context"] = np.array([T])
    payload["d_model"] = np.array([D])
    payload["d_ff"] = np.array([F])

    np.savez(WEIGHTS_FILE, **payload)


print("BMASS Tiny Transformer")
print(f"Vocabulary size: {V}")
print(f"Context length: {T}")
print(f"Trainable parameters: {sum(p.size for p in params):,}")
print(f"Training steps: {STEPS}")

start_time = time.time()
ema_loss = None

for step in range(1, STEPS + 1):
    x, y = sample_window()

    loss, cache = forward(x, y)
    gradients = backward(cache, y)
    gradient_norm = clip_gradients(gradients)

    for parameter, gradient in zip(params, gradients):
        parameter -= LEARNING_RATE * gradient

    if ema_loss is None:
        ema_loss = float(loss)
    else:
        ema_loss = 0.95 * ema_loss + 0.05 * float(loss)

    if step == 1 or step % PRINT_EVERY == 0:
        print(
            f"step={step:4d} "
            f"loss={float(loss):.4f} "
            f"ema={ema_loss:.4f} "
            f"grad={gradient_norm:.3f}"
        )

elapsed = time.time() - start_time
save_weights()

print("\nTraining complete")
print(f"Elapsed time: {elapsed:.2f} seconds")
print(f"Weights saved to: {WEIGHTS_FILE}")
print("\nGenerated text:\n")
print(generate())
