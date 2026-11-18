# Unit 12 live session

DNA language models. A short look at Evo 2, then a demo of Carbon-X, a 14B model
that reads DNA and answers in plain English. The session ends with sparse
autoencoders (SAEs), a way to look inside a model and see what it has learned.

The live session needs a GPU. On molab, turn on the RTX PRO 6000 in the notebook
specs button before you run it. Carbon-X takes about 30 GB of GPU memory.

The exercise trains a small SAE on the hidden states of ESM2-8M and checks each
feature against known protein labels from UniProt. It runs on a CPU.

| Notebook | Open in molab |
| --- | --- |
| Live session (needs a GPU) | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/UNC-DATA-791/live-session-notebooks/blob/main/unit12-live-session/live-session.py) |
| Exercise: a sparse autoencoder on ESM2-8M | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/UNC-DATA-791/live-session-notebooks/blob/main/unit12-live-session/sae-exercise.py) |

`residues.csv` holds one row per residue for the 200 natural proteins from the
Unit 11 exercise, with the amino acid and the UniProt labels.
