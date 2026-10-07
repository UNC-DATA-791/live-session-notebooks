# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "numpy==2.5.3",
#     "polars==1.44.2",
#     "pyarrow==25.0.1",
#     "scikit-learn==1.9.1",
#     "torch==2.14.0",
#     "transformers==5.17.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import io
    import urllib.parse
    import urllib.request

    import altair as alt
    import numpy as np
    import polars as pl
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
    from sklearn.multiclass import OneVsRestClassifier
    from transformers import AutoTokenizer, EsmModel

    return (
        AutoTokenizer,
        EsmModel,
        GroupShuffleSplit,
        LogisticRegression,
        OneVsRestClassifier,
        ShuffleSplit,
        alt,
        io,
        np,
        pl,
        torch,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Copy your neighbor's homework

    Biologists often annotate a new protein like this: find the most similar protein with a known function, and copy its GO terms. Today we test that idea with ESM2 embeddings.

    We compare two methods:

    - **Nearest neighbors (kNN)**: copy GO terms from the most similar training proteins.
    - **Logistic regression**: the model from the live session.

    And two splits: **random** and **superfamily**. Which method suffers more when close cousins are held out?

    The first part of this notebook is the same setup as the live session. Run it and skip down to **Your turn**.
    """)
    return


@app.cell
def _(torch):
    # on a GPU (e.g. molab), use the bigger model and more data;
    # on a laptop, keep it small so it runs in about a minute
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    model_name = "facebook/esm2_t33_650M_UR50D" if gpu else "facebook/esm2_t6_8M_UR50D"
    max_len = 1000 if gpu else 300
    n_terms = 50 if gpu else 12
    gpu, model_name, max_len, n_terms
    return device, max_len, model_name, n_terms


@app.cell
def _(io, max_len, pl, urllib):
    UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode({
        "format": "tsv",
        "query": f"reviewed:true AND organism_id:9606 AND length:[50 TO {max_len}] AND go:*",
        "fields": "accession,protein_name,protein_families,go_f,sequence",
    })

    with urllib.request.urlopen(UNIPROT_URL) as _resp:
        raw = pl.read_csv(io.BytesIO(_resp.read()), separator="\t", quote_char=None)

    proteins = (
        raw.rename({
            "Entry": "accession",
            "Protein names": "name",
            "Protein families": "family",
            "Gene Ontology (molecular function)": "go_mf",
            "Sequence": "sequence",
        })
        .drop_nulls(["family", "go_mf"])
        .with_columns(
            pl.col("go_mf").str.extract_all(r"GO:\d{7}").alias("go_terms"),
            # "Small GTPase superfamily, Rab family" -> "Small GTPase superfamily"
            pl.col("family").str.split(",").list.first().alias("superfamily"),
        )
    )
    return (proteins,)


@app.cell
def _(n_terms, pl, proteins):
    # most common MF terms, minus "protein binding" (GO:0005515), which is on almost everything
    top_terms = (
        proteins.explode("go_terms")
        .filter(pl.col("go_terms") != "GO:0005515")
        .group_by("go_terms")
        .len()
        .sort("len", descending=True)
        .head(n_terms)["go_terms"]
        .to_list()
    )

    # like CAFA, only score proteins that have at least one true label
    sample = proteins.filter(pl.col("go_terms").list.set_intersection(top_terms).list.len() > 0)
    sample.height, sample["superfamily"].n_unique()
    return sample, top_terms


@app.cell
def _(AutoTokenizer, EsmModel, device, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).eval().to(device)
    return model, tokenizer


@app.cell
def _(device, mo, model, np, sample, tokenizer, torch):
    _seqs = sample["sequence"].to_list()
    # sort by length so each batch has little padding; undo the sort at the end
    _order = np.argsort([len(s) for s in _seqs])
    _batches = []

    with torch.no_grad():
        for _i in mo.status.progress_bar(range(0, len(_seqs), 32), title="Embedding"):
            _batch = [_seqs[j] for j in _order[_i:_i + 32]]
            _inputs = tokenizer(_batch, return_tensors="pt", padding=True).to(device)
            _hidden = model(**_inputs).last_hidden_state
            # mean over real residues only, not the padding
            _mask = _inputs["attention_mask"].unsqueeze(-1)
            _batches.append(((_hidden * _mask).sum(1) / _mask.sum(1)).cpu().numpy())

    X = np.empty((len(_seqs), model.config.hidden_size), dtype=np.float32)
    X[_order] = np.concatenate(_batches)

    # one row per protein, one 0/1 column per GO term
    Y = np.array([[t in terms for t in top_terms] for terms in sample["go_terms"].to_list()], dtype=int)
    X.shape, Y.shape
    return X, Y


@app.cell
def _(GroupShuffleSplit, ShuffleSplit, X, sample):
    # (train_idx, test_idx) for each split
    splits = {
        "random": next(ShuffleSplit(1, test_size=0.2, random_state=0).split(X)),
        "superfamily": next(
            GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, groups=sample["superfamily"].to_numpy())
        ),
    }
    return (splits,)


@app.cell
def _(LogisticRegression, OneVsRestClassifier, X, Y, np):
    def lr_scores(train_idx, test_idx):
        """Logistic regression scores, shape (n_test, n_terms)."""
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
        clf.fit(X[train_idx], Y[train_idx])
        return clf.predict_proba(X[test_idx])

    def f_max(probs, y_true):
        """CAFA-style protein-centric F-max (no term weights)."""
        best = 0.0
        has_label = y_true.sum(axis=1) > 0
        for t in np.linspace(0.01, 0.99, 99):
            pred = probs >= t
            tp = (pred & (y_true == 1)).sum(axis=1)
            covered = pred.sum(axis=1) > 0
            if not covered.any():
                continue
            precision = (tp[covered] / pred[covered].sum(axis=1)).mean()
            recall = (tp[has_label] / y_true[has_label].sum(axis=1)).mean()
            if precision + recall > 0:
                best = max(best, 2 * precision * recall / (precision + recall))
        return best

    return f_max, lr_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your turn

    ### 1. Score with nearest neighbors

    Write `knn_scores(train_idx, test_idx, k=10)`. For each test protein:

    1. Find the `k` training proteins with the highest **cosine similarity** to it.
    2. For each GO term, the score is the similarity-weighted mean of those neighbors' labels.

    Return an array of shape `(n_test, n_terms)`, like `lr_scores`.

    <details>
    <summary>Hint</summary>

    Divide each row of `X` by its length (`np.linalg.norm(X, axis=1, keepdims=True)`). Then `X_test @ X_train.T` gives every cosine similarity at once. `np.argsort(-sims, axis=1)[:, :k]` gives the top `k` neighbors.

    </details>
    """)
    return


@app.cell
def _():
    # TODO: write knn_scores(train_idx, test_idx, k=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Compare

    For each split, compute F-max for kNN and for logistic regression. Show the result as a Polars table with columns `split`, `method`, `f_max`.

    <details>
    <summary>Hint</summary>

    `f_max(knn_scores(train, test), Y[test])`. Loop over `splits.items()`.

    </details>
    """)
    return


@app.cell
def _():
    # TODO: F-max table with columns split, method, f_max
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Think

    Which method loses more when you go from the random split to the superfamily split? Why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bonus: how many neighbors?

    Try `k` = 1, 5, 10, 25 and 50. Plot F-max against `k`, one line per split. Does the best `k` change between splits?
    """)
    return


@app.cell
def _():
    # TODO: F-max vs k, one line per split
    return


if __name__ == "__main__":
    app.run()
