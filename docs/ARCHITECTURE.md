# Architecture

OMAIF separates the system into four planes:

```text
Inference plane     Runs base models and promoted cultivars
Cultivation plane   Trains candidates under explicit resource limits
Evidence plane      Stores datasets, logs, hashes, metrics and lineage
Control plane       Evaluates, promotes, quarantines and rolls back
```

The architecture is intentionally engine-agnostic. NumPy proves mechanisms at the smallest scale. Later stages may use llama.cpp-compatible adapters or another lightweight runtime, provided the complete training and evaluation path remains reproducible.
