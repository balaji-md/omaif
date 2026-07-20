# Future Direction

OMAIF should not become another pile of model demos. Its leverage lies in turning constrained machines into controlled local foundries for task-specific intelligence.

## 1. Complete the adapter bridge

The next hard milestone is to adapt a pretrained language model locally while freezing most of its parameters.

The experiment must demonstrate:

```text
unchanged base model
+ locally trained small adapter
+ fixed unseen evaluation set
= measurable behavioural improvement
```

Candidate methods include low-rank adaptation, prefix-like tuning or another parameter-efficient mechanism suitable for CPU execution. The first success need not use Qwen. The smallest viable pretrained transformer is the correct target.

The adapter milestone is complete only when the project records:

- base-model identity and licence;
- exact dataset;
- trainable parameter count;
- peak RAM;
- elapsed time;
- pre-training outputs;
- post-training outputs;
- held-out evaluation results;
- adapter hash;
- successful reload in a fresh process.

## 2. Build the cultivation controller

A future command should orchestrate the loop:

```text
bmass cultivate
```

Its internal pipeline:

```text
survey machine
→ choose feasible crop size
→ validate dataset
→ train in a constrained process
→ measure resource use
→ evaluate against a frozen test set
→ save and hash artefacts
→ compare with current production model
→ promote, quarantine or reject
```

The controller must never equate lower training loss with deployment fitness.

## 3. Separate weights, state and memory

Three layers must remain distinct:

### Learned behaviour
Durable patterns such as command discipline, Alpine conventions, concise responses and correct tool-selection logic.

### Live machine state
Volatile facts such as mounted disks, free RAM, installed packages, active services, network addresses and available models.

### Experience record
Requests, chosen actions, outputs, failures, corrections and final outcomes.

Training should alter durable behaviour. Runtime state should be injected. Experience should be filtered into candidate lessons. Mixing these layers creates stale knowledge and brittle models.

## 4. Establish the OMAIF evaluation harness

Every cultivar should face the same frozen obstacle course:

- command-format correctness;
- task completion rate;
- unsafe-action rate;
- hallucinated-result rate;
- number of attempts;
- generated tokens;
- time to first useful action;
- total task time;
- peak memory;
- energy use where measurable.

The primary metric should be:

> Useful tasks completed per minute, per gigabyte and—where measurable—per joule.

Tokens per second is a component metric, not the governing objective.

## 5. Add model lineage

Each trained artefact should carry a machine-readable birth certificate:

```json
{
  "base_model": "...",
  "base_hash": "...",
  "dataset_hash": "...",
  "training_code_hash": "...",
  "parent_adapter": null,
  "trainable_parameters": 0,
  "hardware": "...",
  "started_at": "...",
  "finished_at": "...",
  "evaluation": "...",
  "status": "candidate"
}
```

This turns a folder of weights into a traceable lineage. Promotion and rollback then become engineering operations rather than acts of faith.

## 6. Create safe automatic exploration

BMASS may eventually inspect its own operating environment, but raw exploration logs must never flow directly into training.

A safe discovery system should:

1. permit only read-only or explicitly harmless commands;
2. capture command, output, exit code and elapsed time;
3. redact secrets and personal data;
4. distinguish stable facts from transient state;
5. reject accidental failures and insecure workarounds;
6. require validation before any record becomes a training example.

The machine can generate experience. It must not automatically canonise every experience as truth.

## 7. Grow from a USB foundry to a local orchard

The long-range architecture is a portable, offline-capable system that can:

- run several small models;
- cultivate task-specific adapters;
- evaluate competing variants;
- preserve organisation-owned artefacts;
- serve models across a local network;
- connect to local documents under explicit permissions;
- promote and roll back versions;
- operate without mandatory cloud dependence.

Higher-end nodes may grow larger cultivars. Lower-end nodes may run, test, collect validated experience and cultivate micro-models. The farm becomes heterogeneous by design.

## 8. Research questions

OMAIF can turn several vague debates into measurable experiments:

- What is the minimum hardware needed to alter a language model meaningfully?
- When does a smaller cultivated model outperform a larger general model operationally?
- How much prompt overhead can be converted into learned behaviour?
- What is the energy cost of local adaptation versus repeated general-model inference?
- Which facts belong in weights, runtime state or retrieval memory?
- How small can an adapter become before behavioural gains disappear?
- Can a local model improve tool use without memorising fragile machine state?

## 9. Publication strategy

Publish the ladder, not merely the summit.

Each release should contain one reproducible rung:

```text
v0.1  Gradient and neural-network proof
v0.2  Character language model
v0.3  Tiny transformer
v0.4  Pretrained-model adapter
v0.5  Evaluation and lineage controller
v1.0  Repeatable local cultivation pipeline
```

Negative results belong in the record. A failed 100-million-parameter attempt on 4 GB RAM is not embarrassment; it maps the boundary of the field.

## End state

OMAIF succeeds when a modest local machine can inspect its constraints, choose a feasible adaptation strategy, cultivate a candidate model, test it against a frozen benchmark, preserve its lineage, and either promote or reject it without surrendering data or control to a remote platform.

That is the shift from owning an AI instance to owning the machinery of adaptation.
