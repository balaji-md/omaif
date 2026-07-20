#!/usr/bin/env python3

import time
from pathlib import Path

import numpy as np


TEXT = """
bmass is a small local artificial intelligence system.
bmass runs from a usb drive.
bmass works without cloud computing.
bmass keeps its models and information locally.
bmass can inspect its machine.
bmass can learn from local examples.
bmass can cultivate small models.
the machine observes acts measures and learns.
local intelligence belongs to the operator.
"""

HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 24
LEARNING_RATE = 0.08
TRAINING_STEPS = 3000
SAMPLE_LENGTH = 240

WEIGHTS_FILE = Path("seedling_weights.npz")


def softmax(values):
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


characters = sorted(set(TEXT))
vocabulary_size = len(characters)

char_to_index = {
    character: index
    for index, character in enumerate(characters)
}

index_to_char = {
    index: character
    for character, index in char_to_index.items()
}


rng = np.random.default_rng(7)

Wxh = rng.normal(
    0,
    0.01,
    (HIDDEN_SIZE, vocabulary_size),
)

Whh = rng.normal(
    0,
    0.01,
    (HIDDEN_SIZE, HIDDEN_SIZE),
)

Why = rng.normal(
    0,
    0.01,
    (vocabulary_size, HIDDEN_SIZE),
)

bh = np.zeros((HIDDEN_SIZE, 1))
by = np.zeros((vocabulary_size, 1))


def loss_and_gradients(inputs, targets, previous_hidden):
    xs = {}
    hs = {-1: previous_hidden.copy()}
    probabilities = {}

    loss = 0.0

    for step, input_index in enumerate(inputs):
        xs[step] = np.zeros((vocabulary_size, 1))
        xs[step][input_index] = 1

        hs[step] = np.tanh(
            Wxh @ xs[step]
            + Whh @ hs[step - 1]
            + bh
        )

        logits = Why @ hs[step] + by
        probabilities[step] = softmax(logits)

        loss += -np.log(
            probabilities[step][targets[step], 0] + 1e-12
        )

    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    next_hidden_gradient = np.zeros_like(previous_hidden)

    for step in reversed(range(len(inputs))):
        output_gradient = probabilities[step].copy()
        output_gradient[targets[step]] -= 1

        dWhy += output_gradient @ hs[step].T
        dby += output_gradient

        hidden_gradient = (
            Why.T @ output_gradient
            + next_hidden_gradient
        )

        raw_hidden_gradient = (
            1 - hs[step] * hs[step]
        ) * hidden_gradient

        dbh += raw_hidden_gradient
        dWxh += raw_hidden_gradient @ xs[step].T
        dWhh += raw_hidden_gradient @ hs[step - 1].T

        next_hidden_gradient = Whh.T @ raw_hidden_gradient

    for gradient in (
        dWxh,
        dWhh,
        dWhy,
        dbh,
        dby,
    ):
        np.clip(
            gradient,
            -5,
            5,
            out=gradient,
        )

    return (
        loss,
        dWxh,
        dWhh,
        dWhy,
        dbh,
        dby,
        hs[len(inputs) - 1],
    )


def generate(seed_character, length):
    hidden = np.zeros((HIDDEN_SIZE, 1))

    current = np.zeros((vocabulary_size, 1))
    current[char_to_index[seed_character]] = 1

    generated = [seed_character]

    for _ in range(length):
        hidden = np.tanh(
            Wxh @ current
            + Whh @ hidden
            + bh
        )

        probabilities = softmax(
            Why @ hidden + by
        ).ravel()

        next_index = rng.choice(
            vocabulary_size,
            p=probabilities,
        )

        next_character = index_to_char[next_index]
        generated.append(next_character)

        current = np.zeros((vocabulary_size, 1))
        current[next_index] = 1

    return "".join(generated)


def parameter_count():
    return (
        Wxh.size
        + Whh.size
        + Why.size
        + bh.size
        + by.size
    )


print("BMASS character-language seedling")
print("Vocabulary size:", vocabulary_size)
print("Trainable parameters:", parameter_count())
print("Training steps:", TRAINING_STEPS)
print()

start_time = time.time()

position = 0
hidden = np.zeros((HIDDEN_SIZE, 1))

for step in range(1, TRAINING_STEPS + 1):
    if position + SEQUENCE_LENGTH + 1 >= len(TEXT):
        position = 0
        hidden = np.zeros((HIDDEN_SIZE, 1))

    input_text = TEXT[
        position:
        position + SEQUENCE_LENGTH
    ]

    target_text = TEXT[
        position + 1:
        position + SEQUENCE_LENGTH + 1
    ]

    inputs = [
        char_to_index[character]
        for character in input_text
    ]

    targets = [
        char_to_index[character]
        for character in target_text
    ]

    (
        loss,
        dWxh,
        dWhh,
        dWhy,
        dbh,
        dby,
        hidden,
    ) = loss_and_gradients(
        inputs,
        targets,
        hidden,
    )

    Wxh[:] -= LEARNING_RATE * dWxh
    Whh[:] -= LEARNING_RATE * dWhh
    Why[:] -= LEARNING_RATE * dWhy
    bh[:] -= LEARNING_RATE * dbh
    by[:] -= LEARNING_RATE * dby

    position += SEQUENCE_LENGTH

    if step == 1 or step % 500 == 0:
        average_loss = loss / len(inputs)

        print(
            f"step={step:4d} "
            f"loss_per_character={average_loss:.4f}"
        )

elapsed = time.time() - start_time

np.savez(
    WEIGHTS_FILE,
    Wxh=Wxh,
    Whh=Whh,
    Why=Why,
    bh=bh,
    by=by,
    characters=np.array(characters),
)

print()
print("Training complete")
print(f"Elapsed time: {elapsed:.2f} seconds")
print("Weights saved to:", WEIGHTS_FILE)
print()
print("Generated text:")
print("-" * 50)
print(generate("b", SAMPLE_LENGTH))
print("-" * 50)
