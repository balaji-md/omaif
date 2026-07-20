# Evidence Protocol

For every claimed training run:

1. Record date, operating system, CPU, visible RAM, swap and free storage.
2. Preserve the exact source and dataset.
3. Declare random seed, trainable parameter count and optimisation steps.
4. Capture starting and final loss.
5. Record elapsed time and peak memory when possible.
6. Save the learned artefact.
7. Terminate the process and reload the artefact in a new process.
8. Run a fixed evaluation that was not used for training.
9. Hash the evidence bundle.
10. Commit the code and logs together or reference the exact commit.

A screenshot can corroborate a run. It cannot replace raw logs, source code and hashes.
