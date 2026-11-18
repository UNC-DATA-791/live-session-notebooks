# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "numpy==2.4.6",
#     "polars==2.0.0",
#     "pyarrow==25.0.0",
#     "torch==2.11.0",
#     "transformers==5.20.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import altair as alt
    import numpy as np
    import polars as pl
    import torch

    return alt, np, pl, torch


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Exercise: what has a protein language model learned?

    In the live session, a sparse autoencoder (SAE) was a way to pull apart the ideas packed into a model's hidden state. Now you will train one yourself.

    We use a small protein model, **ESM2-8M**, because we . We can check each feature the SAE finds against real labels, such as helix, signal peptide and amino acid type.

    ## Before you start: read the abstract

    Read the abstract and look at the figures of [InterPLM: Discovering interpretable features in protein language models via sparse autoencoders](https://arxiv.org/abs/2412.12101) (Simon and Zou, 2024). Answer in your own words:

    - Which model did they look inside? What did they train the SAE on?
    - What did they compare the SAE features with? What did they find?
    - Why did they not just look at single neurons?
    """)
    return


@app.cell(hide_code=True)
def data_intro(mo):
    mo.md(r"""
    ## The data

    - **Proteins.** The 200 natural Swiss-Prot proteins from the Unit 11 exercise.
    - **Activations.** For each residue, the hidden state after one layer of **ESM2-8M**. The model has 6 layers. We start with layer 4. That is **35,897 residues**, each with **320 numbers**. You can pick another layer below.
    - **Labels.** The amino acid of each residue, plus tags from UniProt: helix, strand, turn, transmembrane, signal peptide, glycosylation, binding site, domain and disulfide bond.

    One caution. Helix, strand and turn tags only exist for proteins with a known 3D structure (`has_structure = 1`). In the other proteins, a residue without a tag is **unknown**, not "not a helix".
    """)
    return


@app.cell(hide_code=True)
def layer_picker(mo):
    layer = mo.ui.dropdown(
        options={str(_i): _i for _i in range(1, 7)},
        value="4",
        label="ESM2-8M layer",
    )
    layer
    return (layer,)


@app.cell
def load_data(mo, pl):
    residues = pl.read_csv(mo.notebook_dir() / "residues.csv")

    # Rebuild each protein's sequence from the residue table
    protein_sequences = (
        residues.group_by("accession", maintain_order=True)
        .agg(pl.col("aa").str.join("").alias("sequence"))["sequence"]
        .to_list()
    )

    label_names = [
        "helix",
        "strand",
        "turn",
        "transmembrane",
        "signal_peptide",
        "glycosylation",
        "binding_site",
        "domain",
        "disulfide",
    ]
    residues.head()
    return label_names, protein_sequences, residues


@app.cell
def esm_activations_fn(np, protein_sequences, torch):
    from functools import cache

    from transformers import AutoTokenizer, EsmModel

    @cache
    def load_esm():
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval()
        return tokenizer, model

    @cache
    def esm_activations(layer):
        """Hidden state after `layer` for every residue of every protein: one row per residue, 320 numbers each."""
        tokenizer, model = load_esm()
        rows = []
        with torch.no_grad():
            for sequence in protein_sequences:
                hidden = model(
                    **tokenizer(sequence, return_tensors="pt"),
                    output_hidden_states=True,
                ).hidden_states[layer][0]
                rows.append(
                    hidden[1:-1].numpy()
                )  # drop the start and end tokens
        return np.concatenate(rows)

    return (esm_activations,)


@app.cell
def activations_cell(esm_activations, layer, mo, residues):
    activations = esm_activations(layer.value)

    # Give every one of the 320 numbers a mean of 0 and a spread of 1
    X = (activations - activations.mean(axis=0)) / activations.std(axis=0)
    assert len(residues) == len(X)
    mo.md(
        f"Layer **{layer.value}**: **{X.shape[0]:,}** residues, **{X.shape[1]}** numbers each."
    )
    return (X,)


@app.cell(hide_code=True)
def sae_task_md(mo):
    mo.md(r"""
    ## 1. The sparse autoencoder

    The SAE has two parts, as in the diagram from the live session:

    - The **encoder** is one linear layer from 320 numbers to `n_features` numbers, followed by a **ReLU** (it sets negative values to 0).
    - The **top-k step** keeps only the `k` largest values for each residue and sets all the others to 0. The result is the **features**. At most `k` features are on for any residue. This is how we make the SAE *sparse*.
    - The **decoder** is one linear layer from `n_features` numbers back to 320 numbers. It rebuilds the hidden state.

    We use `n_features = 2560`, which is 8 times more than 320. (Goodfire's SAE for Evo 2 used a close cousin of top-k, called BatchTopK.)

    Write the class `SparseAutoencoder`. Its `forward(x)` must return **two** things: the features and the rebuilt input.

    /// details | Hint
    ```python
    class SparseAutoencoder(torch.nn.Module):
        def __init__(self, n_inputs, n_features, k):
            super().__init__()
            self.k = k
            self.encoder = torch.nn.Linear(n_inputs, n_features)
            self.decoder = torch.nn.Linear(n_features, n_inputs)

        def forward(self, x):
            activations = torch.relu(self.encoder(x))
            top = torch.topk(activations, self.k, dim=1)
            features = torch.zeros_like(activations).scatter(1, top.indices, top.values)
            return features, self.decoder(features)
    ```
    ///
    """)
    return


@app.cell
def sae_class(torch):
    class SparseAutoencoder(torch.nn.Module):
        def __init__(self, n_inputs, n_features, k):
            super().__init__()
            self.k = k
            self.encoder = torch.nn.Linear(n_inputs, n_features)
            self.decoder = torch.nn.Linear(n_features, n_inputs)

        def forward(self, x):
            activations = torch.relu(self.encoder(x))
            top = torch.topk(activations, self.k, dim=1)
            features = torch.zeros_like(activations).scatter(
                1, top.indices, top.values
            )
            return features, self.decoder(features)

    return (SparseAutoencoder,)


@app.cell
def sae_check(SparseAutoencoder, X, mo, torch):
    _features, _rebuilt = SparseAutoencoder(320, 2560, k=32)(
        torch.tensor(X[:5])
    )
    mo.md(
        f"Features: **{tuple(_features.shape)}**. Rebuilt: **{tuple(_rebuilt.shape)}**. "
        "You should see (5, 2560) and (5, 320). "
        f"Features on per residue: **{(_features > 0).sum(dim=1).tolist()}**. Each must be 32 or fewer."
    )
    return


@app.cell(hide_code=True)
def train_task_md(mo):
    mo.md(r"""
    ## 2. Train it

    The SAE learns by trying to rebuild its input. The loss is the **mean squared error** between the rebuilt hidden state and the original. The sparsity comes from the top-k step, so we do not need a penalty.

    Finish the training step in `train_sae`. For each batch:

    1. Run the SAE to get the features and the rebuilt input.
    2. Compute the loss.
    3. Clear the old gradients, run `backward()`, and take an optimizer step.

    /// details | Hint
    ```python
    features, rebuilt = sae(batch)
    loss = ((rebuilt - batch) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```
    ///
    """)
    return


@app.cell
def train_fn(SparseAutoencoder, torch):
    def train_sae(
        X, k, n_features=2560, epochs=40, batch_size=512, lr=1e-3, seed=0
    ):
        """Train a top-k SAE. Returns the trained SAE and the mean loss of each epoch."""
        torch.manual_seed(seed)
        data = torch.tensor(X)
        sae = SparseAutoencoder(data.shape[1], n_features, k)
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        epoch_losses = []
        for _ in range(epochs):
            order = torch.randperm(len(data))
            total = 0.0
            for i in range(0, len(data), batch_size):
                batch = data[order[i : i + batch_size]]
                features, rebuilt = sae(batch)
                loss = ((rebuilt - batch) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            epoch_losses.append(total / len(data))
        return sae, epoch_losses

    return (train_sae,)


@app.cell(hide_code=True)
def train_runs_md(mo):
    mo.md(r"""
    Train three SAEs, with `k = 8`, `32` and `128`. This takes about a minute on a CPU. Then compare them.
    """)
    return


@app.cell(hide_code=True)
def train_button(layer, mo):
    train_button = mo.ui.run_button(
        label=f"Train three SAEs on layer {layer.value}"
    )
    train_button
    return (train_button,)


@app.cell
def train_runs(X, mo, pl, torch, train_button, train_sae):
    mo.stop(not train_button.value, mo.md("Press the button to train."))
    ks = [8, 32, 128]
    trained = {k: train_sae(X, k) for k in ks}

    _rows = []
    for _k, (_sae, _losses) in trained.items():
        with torch.no_grad():
            _features, _ = _sae(torch.tensor(X))
        _rows.append(
            {
                "k": _k,
                "variance not rebuilt": round(_losses[-1], 3),
                "features never on": int(
                    ((_features > 0).sum(dim=0) == 0).sum()
                ),
            }
        )
    pl.DataFrame(_rows)
    return (trained,)


@app.cell(hide_code=True)
def train_curves(alt, pl, trained):
    _curves = pl.DataFrame(
        [
            {"epoch": _e + 1, "loss": _l, "k": f"k = {_k}"}
            for _k, (_, _losses) in trained.items()
            for _e, _l in enumerate(_losses)
        ]
    )
    alt.Chart(_curves).mark_line().encode(
        x=alt.X("epoch:Q", title="Epoch"),
        y=alt.Y(
            "loss:Q",
            title="Loss (variance not rebuilt)",
            scale=alt.Scale(type="log"),
        ),
        color=alt.Color("k:N", title=None),
    ).properties(
        width=460, height=280, title="Training loss for three values of k"
    )
    return


@app.cell(hide_code=True)
def train_after(mo):
    mo.md(r"""
    A smaller `k` is sparser, so each residue is described by fewer features. The cost is that less of the hidden state is rebuilt. The **variance not rebuilt** column is the share of the original hidden state that the SAE misses. Smaller is better.

    Notice the **features never on**. These are dead features. They waste space.
    """)
    return


@app.cell(hide_code=True)
def match_task_md(mo):
    mo.md(r"""
    ## 3. Which feature matches which label?

    Now the key question: **does a feature mean something?** We will use the SAE with `k = 32`.

    A feature is **on** for a residue if its value is above 0. For each label (for example "helix"), a good feature is on for the helix residues and off for the others. We score this with the **F1 score**:

    - **TP** (true positives): the feature is on and the label is true.
    - **FP** (false positives): the feature is on but the label is false.
    - **FN** (false negatives): the feature is off but the label is true.

    $$F1 = \frac{2\,TP}{2\,TP + FP + FN}$$

    F1 is 1 for a perfect match and 0 for no match.

    Write `f1_per_feature(on, y)`. `on` is a matrix of 0s and 1s with one row per residue and one column per feature. `y` is a true/false array with one value per residue. Return one F1 score for **each feature**.

    /// details | Hint
    `y.astype(np.float32) @ on` gives TP for every feature at once. The number of times a feature is on is `on.sum(axis=0)`. The number of true residues is `y.sum()`. Note that `2*TP + FP + FN` equals `on.sum(axis=0) + y.sum()`.

    ```python
    tp = y.astype(np.float32) @ on
    return 2 * tp / np.maximum(on.sum(axis=0) + y.sum(), 1)
    ```
    ///
    """)
    return


@app.cell
def f1_fn(np):
    def f1_per_feature(on, y):
        tp = y.astype(np.float32) @ on
        return 2 * tp / np.maximum(on.sum(axis=0) + y.sum(), 1)

    return (f1_per_feature,)


@app.cell
def match_setup(X, label_names, np, residues, torch, trained):
    main_k = 32
    _sae, _ = trained[main_k]
    with torch.no_grad():
        _features, _ = _sae(torch.tensor(X))
    on = (_features > 0).numpy().astype(np.float32)  # 1 where a feature is on

    # Every label as a true/false array, with one value per residue
    labels = {
        f"aa_{_a}": (residues["aa"] == _a).to_numpy()
        for _a in "ACDEFGHIKLMNPQRSTVWY"
    }
    labels.update(
        {_n: residues[_n].to_numpy().astype(bool) for _n in label_names}
    )

    # Helix, strand and turn are only known for proteins with a structure, so we score them there only
    _has_structure = residues["has_structure"].to_numpy().astype(bool)
    scored_on = {
        _n: _has_structure
        if _n in ("helix", "strand", "turn")
        else np.ones(len(residues), dtype=bool)
        for _n in labels
    }
    on.shape
    return labels, on, scored_on


@app.cell
def match_results(X, f1_per_feature, labels, np, on, pl, scored_on):
    def best_match(on_matrix):
        """For each label, the best F1 over all features, and which feature gave it."""
        _out = {}
        for _name, _y in labels.items():
            _mask = scored_on[_name]
            _scores = f1_per_feature(on_matrix[_mask], _y[_mask])
            _out[_name] = (float(_scores.max()), int(_scores.argmax()))
        return _out

    sae_best = best_match(on)

    # The same test on the raw 320 numbers. "On" means the number is above a cut-off. We try 4 cut-offs and keep the best.
    _raw = [best_match((X > _t).astype(np.float32)) for _t in (0, 0.5, 1, 2)]
    raw_best = {_n: max(_r[_n][0] for _r in _raw) for _n in labels}

    matches = pl.DataFrame(
        [
            {"label": _n, "feature": _f, "F1": round(_s, 2)}
            for _n, (_s, _f) in sae_best.items()
        ]
    ).sort("F1", descending=True)
    matches
    return raw_best, sae_best


@app.cell(hide_code=True)
def match_chart(alt, label_names, labels, np, pl, raw_best, sae_best):
    _rows = []
    for _n in label_names:
        _rows += [
            {"label": _n, "source": "Single raw number", "F1": raw_best[_n]},
            {"label": _n, "source": "SAE feature", "F1": sae_best[_n][0]},
        ]
    _aa = [_n for _n in labels if _n.startswith("aa_")]
    _rows += [
        {
            "label": "amino acid (mean of 20)",
            "source": "Single raw number",
            "F1": float(np.mean([raw_best[_n] for _n in _aa])),
        },
        {
            "label": "amino acid (mean of 20)",
            "source": "SAE feature",
            "F1": float(np.mean([sae_best[_n][0] for _n in _aa])),
        },
    ]
    alt.Chart(pl.DataFrame(_rows)).mark_bar().encode(
        y=alt.Y(
            "label:N",
            title=None,
            sort=label_names + ["amino acid (mean of 20)"],
        ),
        x=alt.X(
            "F1:Q",
            title="Best F1 of any single number or feature",
            scale=alt.Scale(domain=[0, 1]),
        ),
        yOffset=alt.YOffset("source:N"),
        color=alt.Color(
            "source:N",
            title=None,
            scale=alt.Scale(
                domain=["Single raw number", "SAE feature"],
                range=["#9aa3ad", "#2f6fdb"],
            ),
        ),
        tooltip=["label", "source", alt.Tooltip("F1:Q", format=".2f")],
    ).properties(
        width=460,
        height=380,
        title="Does one number or one feature match a label?",
    )
    return


if __name__ == "__main__":
    app.run()
