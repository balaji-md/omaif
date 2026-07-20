# OMAIF

## Ol’ MacDonald’s AI Farm

**A minimal-hardware laboratory for cultivating, adapting, evaluating and deploying local artificial intelligence.**

OMAIF asks a deliberately sharp question:

> How little hardware is required before a machine can modify its own intelligence?

The project began on **BMASS**, a USB-booted Alpine Linux system running on a dual-core Intel Celeron N4500 with 4 GB RAM. The machine first ran a local Qwen model. It then crossed a more important boundary: it performed gradient descent, trained a neural network, trained a character-level language model, and trained a tiny causal transformer locally.

OMAIF treats model development as cultivation rather than consumption. A base model is seed stock. Data is soil. Training is selective pressure. Evaluation is crop inspection. An adapter is a cultivar. Deployment is harvest. Rollback is seed preservation.

## Current proof ladder

| Stage | Name | Mechanism | Status |
|---|---|---|---|
| 0 | Gradient | Single-parameter gradient descent | Complete |
| 1 | Seed | Small feed-forward neural network | Complete |
| 2 | Seedling | Character-level language model | Complete |
| 3 | Sapling | Tiny causal transformer with self-attention | Complete |
| 4 | Cultivar | Adapter training on a pretrained model | Next |
| 5 | Orchard | Repeatable local training, evaluation and promotion | Planned |

The completed stages prove mechanism, not industrial scale. OMAIF does **not** claim that a low-power Celeron can train a frontier foundation model from scratch. It demonstrates that training is not inherently a cloud-only act; the real constraint is scale.

## Design principles

1. **Local first** — inference, datasets, logs and learned artefacts should remain on the operator’s machine by default.
2. **Minimal sufficient intelligence** — use the smallest model that reliably completes the task.
3. **Separate behaviour from state** — teach durable operating patterns; inject volatile machine facts at runtime.
4. **Evidence before rhetoric** — preserve hardware manifests, source code, loss curves, timings, hashes and before/after outputs.
5. **Reversible cultivation** — retain the base model, version every adapter, evaluate before promotion and make rollback cheap.
6. **Operator control** — models propose and interpret; permissions, execution boundaries and promotion remain explicit.

## Repository map

```text
omaif/
├── README.md
├── LICENSE
├── INSTALL.md
├── FUTURE_DIRECTION.md
├── SYNC_CHECKLIST.md
├── .gitignore
├── assets/                 Images and diagrams safe for publication
├── docs/                   Architecture, evidence protocol and experiment log
├── evidence/               Hardware reports, timings, hashes and outputs
├── experiments/
│   ├── 00-gradient/        Single-weight learning proof
│   ├── 01-seed/            Feed-forward network
│   ├── 02-seedling/        Character language model
│   ├── 03-sapling/         Tiny transformer
│   └── 04-adapter/         Pretrained-model adaptation work
└── scripts/                Survey, validation and reproducibility utilities
```

## Quick start

Read [INSTALL.md](INSTALL.md), then copy the existing BMASS experiment files into the matching directories using [SYNC_CHECKLIST.md](SYNC_CHECKLIST.md).

For the first public release, publish the code and evidence for one rung at a time. Every experiment should answer five questions:

- What hardware ran it?
- What exactly was trained?
- What changed?
- How long did it take?
- Can another person reproduce it?

## Evidence standard

Each experiment directory should contain:

```text
README.md          purpose, architecture and exact command
source file        executable experiment
requirements.txt   only when dependencies exceed base Python/NumPy
before.txt         baseline behaviour where applicable
after.txt          post-training behaviour
metrics.txt        loss, elapsed time, memory and parameter count
SHA256SUMS         hashes of source, data and learned artefacts
```

Do not publish screenshots as the only evidence. Screenshots are supporting artefacts; machine-readable logs are the spine.

## Scope

OMAIF is intended for:

- minimal-hardware AI training experiments;
- offline and on-premises adaptation;
- small-model specialisation;
- transparent evaluation and rollback;
- educational reconstruction of the learning stack;
- practical local cultivars for tightly defined tasks.

It is not a safety wrapper for autonomous privileged execution. Any shell or systems integration must remain constrained by ordinary operating-system permissions and explicit allow-lists.

## Roadmap

The immediate engineering target is a locally trained adapter for a genuinely small pretrained model, followed by a controlled before/after evaluation. Longer-range directions are described in [FUTURE_DIRECTION.md](FUTURE_DIRECTION.md).

## Project status

**Experimental.** Interfaces, directory names and training recipes will change. Preserve evidence and hashes before refactoring.

## Licence

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Citation

Until a formal release is published, cite the repository name, commit hash and experiment directory used. A citable release and metadata file will be added once the adapter milestone is reproducible.
