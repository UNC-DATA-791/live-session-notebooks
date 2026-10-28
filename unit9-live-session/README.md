# Unit 9 live session

Autoregressive generative models for protein discovery. Encoder vs. decoder
(ESM2 vs. ProGen2), the autoregressive loop, and the sampling knobs that decide
whether a model recites or designs.

| Notebook | Open in molab |
| --- | --- |
| Live session | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/UNC-DATA-791/live-session-notebooks/blob/main/unit9-live-session/live-session.py) |
| Exercise: the protein Turing test | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/UNC-DATA-791/live-session-notebooks/blob/main/unit9-live-session/protein-turing-test-exercise.py) |

## The exercise

Students separate 30 natural globins (Pfam PF00042) from 30 that ProGen2 wrote,
by scoring every sequence with ESM2. ESM2 can judge the sequences because ESM2
did not write any of them.

A slider controls how many natural amino acids ProGen2 receives before it
writes the rest. That slider is the lesson. Measured on an RTX PRO 6000:

| Natural amino acids given to ProGen2 | ESM2 AUC |
| --- | --- |
| 5 | 0.953 |
| 15 | 0.941 |
| 30 | 0.907 |
| 60 | 0.758 |

The more natural sequence the model is given, the harder its output is to
detect.

Three tasks are left open, each with a hint in a `<details>` block: clean the
UniProt table, write `esm_score`, and choose a threshold.

## Requirements

Needs a GPU. On a GPU both notebooks load `progen2-large` (2.8B parameters,
~20 GB of VRAM in float32) and `esm2_t33_650M`; without one they fall back to
`progen2-small` and `esm2_t6_8M`, which has not been tested.

Generating the 30 sequences takes about two minutes on a GPU.
