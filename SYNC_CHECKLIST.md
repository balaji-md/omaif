# BMASS-to-GitHub Sync Checklist

Copy the existing BMASS files into these repository locations. Confirm names on the USB before copying; do not overwrite newer versions blindly.

| Existing BMASS file | Repository destination |
|---|---|
| `seed.py` or the renamed `tiny_nn.py` | `experiments/01-seed/seed.py` |
| `seedling.py` | `experiments/02-seedling/seedling.py` |
| `tiny_transformer.py` | `experiments/03-sapling/tiny_transformer.py` |
| `tiny_nn_weights.npz` | `evidence/01-seed/tiny_nn_weights.npz` |
| `seedling_weights.npz` | `evidence/02-seedling/seedling_weights.npz` |
| `tiny_transformer_weights.npz` | `evidence/03-sapling/tiny_transformer_weights.npz` |
| `bmass-hardware-survey.txt` | `evidence/hardware/bmass-hardware-survey.txt` |

Create evidence directories locally:

```sh
mkdir -p evidence/{hardware,01-seed,02-seedling,03-sapling}
```

For each experiment, also sync:

```text
run.log
metrics.txt
before.txt          when applicable
after.txt           when applicable
SHA256SUMS
```

## Hash before moving files

On BMASS:

```sh
cd /opt/bmass/training
sha256sum *.py *.npz > SHA256SUMS
```

## Git sequence

```sh
git status
git add README.md LICENSE INSTALL.md FUTURE_DIRECTION.md SYNC_CHECKLIST.md
git add scripts docs experiments evidence assets .gitignore
git commit -m "Establish OMAIF cultivation repository"
git push -u origin main
```

After copying real experiment files:

```sh
git add experiments evidence
git commit -m "Add BMASS local training proofs"
git push
```

## Never sync

- passwords, API keys or tokens;
- private datasets;
- shell history without review;
- machine identifiers that are not needed for reproducibility;
- proprietary model weights or restricted datasets;
- unredacted logs containing personal information.
