# Installation

OMAIF is designed to keep the target machine lean. Development may occur on a comfortable workstation, but every claimed result must be rerun on the declared target hardware.

## 1. Baseline target

The first BMASS experiments used:

```text
Operating system: Alpine Linux 3.24
Architecture:     x86_64
CPU:              Intel Celeron N4500, 2 cores
Memory:           4 GB class; approximately 3.6 GB visible
Swap:             approximately 1.9 GB
Python:           Python 3
Numerics:         NumPy
GPU:              none required
```

These values describe the initial test bed, not a universal requirement.

## 2. Alpine preparation

Run package installation as root:

```sh
apk update
apk add python3 py3-numpy nano
```

If `py3-numpy` is unavailable, confirm that the Alpine `community` repository is enabled in `/etc/apk/repositories`, then run `apk update` again.

Create a non-root workspace:

```sh
mkdir -p /opt/bmass/training
chown bmass:bmass /opt/bmass/training
```

Switch to the unprivileged account:

```sh
su - bmass
cd /opt/bmass/training
```

Verify the numerical runtime:

```sh
python3 -c "import numpy; print(numpy.__version__)"
```

## 3. Resource isolation

On BMASS, the local llama server normally occupies memory. Stop it before training and verify that it is no longer consuming resources.

Find the process:

```sh
ps aux | grep '[l]lama-server'
```

Stop the identified process using its actual PID:

```sh
kill PID
```

Then inspect available memory:

```sh
free -h
```

Do not copy a PID from documentation. Process identifiers change at every boot.

## 4. Clone or copy the repository

On a networked development system:

```sh
git clone https://github.com/YOUR-GITHUB-USER/omaif.git
cd omaif
```

On the offline BMASS target, copy only the experiment and script files required for the run. Avoid installing Git merely for convenience if storage is tight.

Suggested target path:

```text
/opt/bmass/training/omaif
```

## 5. Run experiments

The canonical command pattern is:

```sh
cd /opt/bmass/training/omaif/experiments/EXPERIMENT
python3 PROGRAM.py 2>&1 | tee run.log
```

Capture resource use separately where possible:

```sh
/usr/bin/time -v python3 PROGRAM.py 2>&1 | tee run.log
```

BusyBox installations may not provide GNU `time -v`. In that case record at minimum:

```sh
date
free -h
df -h
```

before and after the run.

## 6. Preserve the artefact

After training:

```sh
sha256sum *.py *.npz *.txt > SHA256SUMS
```

Copy the following back into the Git repository:

- source code;
- dataset or dataset generator;
- saved weights or adapter, when licence and size permit;
- console log;
- metrics;
- hardware survey;
- SHA-256 hashes;
- exact commit identifier.

## 7. Reproducibility rule

A result belongs in the project only when:

1. the source is preserved;
2. the target hardware is declared;
3. the command is exact;
4. the learned artefact is saved;
5. the output is independently reloadable;
6. the result can be repeated from a clean start.

A falling loss curve alone is insufficient. The trained state must survive process termination and alter subsequent behaviour.

## 8. Development workflow

Use Debian or another full desktop environment to edit larger files:

```text
Edit and test syntax on Debian
→ copy source to the BMASS USB
→ boot Alpine BMASS
→ stop the inference server
→ train as the bmass user
→ preserve logs and weights
→ return artefacts to the Git repository
```

This is not cheating. Development convenience and execution provenance are separate layers. The training claim remains local only when the actual optimisation run occurs on the declared BMASS hardware.
