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

__generated_with = "0.25.1"
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
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import NearestNeighbors
    from transformers import AutoTokenizer, EsmModel

    return (
        AutoTokenizer,
        EsmModel,
        GroupShuffleSplit,
        LogisticRegression,
        OneVsRestClassifier,
        io,
        np,
        pl,
        torch,
        train_test_split,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Annotation by homology

    Biologists often annotate a new protein like this: find the most similar protein with a known function, and copy its GO terms. Today we test that idea with ESM2 embeddings.

    We compare two methods:

    - **Nearest neighbors (kNN)**: copy GO terms from the most similar training proteins.
    - **Logistic regression**: the model from the live session.

    We'll use the same two train/test splits from the live session: **random**, and **family**, which holds out whole superfamilies. Is copying from neighbors as good as a trained model? Does it hold up when close cousins are held out?

    The first part of this notebook is the same setup as the live session.
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
    n_terms = 8
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
    # most common MF terms
    top_terms = (
        proteins.explode("go_terms")
        .group_by("go_terms")
        .len()
        .sort("len", descending=True)
        .head(n_terms)["go_terms"]
        .to_list()
    )

    # like CAFA, only score proteins that have at least one true label
    sample = proteins.filter(pl.col("go_terms").list.set_intersection(top_terms).list.len() > 0)

    # rows of sample line up with X and Y (made below):
    # - row 1084 of sample is a protein
    # - row 1084 of X is that protein's embedding
    # - row 1084 of Y is its 0/1 GO labels
    sample.height, sample["superfamily"].n_unique()
    return sample, top_terms


@app.cell
def _(AutoTokenizer, EsmModel, device, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).eval().to(device)
    return model, tokenizer


@app.cell
def _(device, mo, model, np, sample, tokenizer, top_terms, torch):
    _seqs = sample["sequence"].to_list()
    # sort by length so each batch has little padding; undo the sort at the end
    _order = np.argsort([len(s) for s in _seqs])
    _batches = []

    with torch.no_grad():
        for _i in mo.status.progress_bar(range(0, len(_seqs), 32), title="Embedding"):
            _batch = [_seqs[j] for j in _order[_i:_i + 32]]
            _inputs = tokenizer(
                _batch, return_tensors="pt", padding=True, return_special_tokens_mask=True
            ).to(device)
            _special = _inputs.pop("special_tokens_mask")  # the model doesn't accept this input
            _hidden = model(**_inputs).last_hidden_state
            # mean over real residues only: skip padding and the <cls>/<eos> tokens
            _mask = (_special == 0).unsqueeze(-1)
            _batches.append(((_hidden * _mask).sum(1) / _mask.sum(1)).cpu().numpy())

    X = np.empty((len(_seqs), model.config.hidden_size), dtype=np.float32)
    X[_order] = np.concatenate(_batches)

    # one row per protein, one 0/1 column per GO term
    Y = np.array([[t in terms for t in top_terms] for terms in sample["go_terms"].to_list()], dtype=int)
    X.shape, Y.shape
    return X, Y


@app.cell
def _(GroupShuffleSplit, X, np, sample, train_test_split):
    _rows = np.arange(len(X))
    _groups = sample["superfamily"].to_numpy()

    # (train_idx, test_idx) for each split
    splits = {
        # random: members of one superfamily can land on both sides
        "random": train_test_split(_rows, test_size=0.2, random_state=0),
        # family: each superfamily lands on one side only
        "family": next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, groups=_groups)),
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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model evaluation

    ### 1. Score with nearest neighbors

    We'll build the kNN model one step at a time using the random split training data, then wrap it in a function.

    `X` is the feature matrix. Each row is one protein. Each column is one number in its ESM2 embedding. `Y` holds the labels: one row per protein, one 0/1 column per GO term.
    """)
    return


@app.cell
def _(X, Y, splits):
    # start with the random split
    train_idx, test_idx = splits["random"]
    X.shape, Y.shape, len(train_idx), len(test_idx)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1a. Find the nearest neighbors

    Cosine similarity is a way to measure how close two embedding vectors are. It looks at the angle between them: 1 means they point the same way, and smaller values mean they point further apart.

    scikit-learn's `NearestNeighbors` finds the closest vectors for us. With `metric="cosine"`, it uses **cosine distance**, which is 1 − cosine similarity. A small distance means close.

    1. `fit` it on the training embeddings, `X[train_idx]`.
    2. Call `kneighbors` on the test embeddings, `X[test_idx]`. It returns two arrays, both of shape `(n_test, k)`: the distances, and the positions of the nearest neighbors.

    Careful: the positions count within `train_idx`, not rows of `X`. To get rows of `X` or `sample`, use `train_idx[near_neighbors]`.

    <details>
    <summary>Hint</summary>

    ```python
    neighbor_finder = NearestNeighbors(n_neighbors=10, metric="cosine").fit(X[train_idx])
    distances, near_neighbors = neighbor_finder.kneighbors(X[test_idx])
    near_neighbors.shape
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: fit NearestNeighbors on the training embeddings, then find the 10 nearest neighbors of each test protein
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1b. Look at the neighbors

    Show test protein 0 and its 3 nearest neighbors, with their names, superfamilies and cosine similarity (1 − distance). Do they look related?

    <details>
    <summary>Hint</summary>

    ```python
    # test protein 0, then its 3 nearest neighbors
    mo.vstack([
        sample[test_idx[:1]].select("name", "superfamily"),
        sample[train_idx[near_neighbors[0, :3]]]
        .select("name", "superfamily")
        .with_columns(pl.Series("similarity", 1 - distances[0, :3]).round(3)),
    ])
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: show test protein 0 and its 3 nearest neighbors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1c. Copy the neighbors' labels

    Get the GO labels of every neighbor. Call the result `neighbor_labels`, shape `(n_test, k, n_terms)`.

    `Y[train_idx]` keeps only the training labels, in the same order `NearestNeighbors` saw them. Indexing it with `near_neighbors` swaps each neighbor position for that protein's row of labels.

    <details>
    <summary>Hint</summary>

    ```python
    neighbor_labels = Y[train_idx][near_neighbors]
    neighbor_labels.shape
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: make neighbor_labels, then check its shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1d. Average into scores

    For each test protein and GO term, the score is the share of its 10 neighbors that have the term. If 7 of 10 neighbors have it, the score is 0.7. Call the result `knn_scores_random`, shape `(n_test, n_terms)`.

    Check: `f_max(knn_scores_random, Y[test_idx])`. You should get about 0.70 on a laptop.

    <details>
    <summary>Hint</summary>

    ```python
    knn_scores_random = neighbor_labels.mean(axis=1)
    knn_scores_random.shape, round(f_max(knn_scores_random, Y[test_idx]), 3)
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: make knn_scores_random, then check its F-max
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1e. Wrap it in a function

    Put steps 1a, 1c and 1d into `knn_scores(train_idx, test_idx, k=10)`. Return the scores, like `lr_scores` does. You'll use it on both splits next.

    Check: `knn_scores(train_idx, test_idx)` should match `knn_scores_random` from 1d.

    <details>
    <summary>Hint</summary>

    ```python
    def knn_scores(train_idx, test_idx, k=10):
        neighbor_finder = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X[train_idx])
        _, near_neighbors = neighbor_finder.kneighbors(X[test_idx])
        return Y[train_idx][near_neighbors].mean(axis=1)


    # should match knn_scores_random from 1d
    np.allclose(knn_scores(train_idx, test_idx), knn_scores_random)
    ```

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

    For each split, compute F-max for kNN and for logistic regression. Show the result as a Polars table with columns `split`, `kNN` and `logistic regression`.

    <details>
    <summary>Hint</summary>

    ```python
    recs = []
    for _split, (_train, _test) in splits.items():
        recs.append({
            "split": _split,
            "kNN": f_max(knn_scores(_train, _test), Y[_test]),
            "logistic regression": f_max(lr_scores(_train, _test), Y[_test]),
        })

    pl.DataFrame(recs)
    ```

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

    Which method scores higher on each split? Does either one lose much more on the family split? Why might that be?

    <details>
    <summary>Answer</summary>

    On a laptop run, logistic regression beats kNN on both splits (about 0.76 vs. 0.70 random, 0.69 vs. 0.65 family). Both drop by a similar amount, about 0.05 to 0.06.

    Why they both drop: on the random split, most test proteins have a close cousin in training. On the family split, those cousins are gone. kNN now copies from a distant protein that may have a different job. Logistic regression learns one direction in embedding space for each GO term, which helps a little. But both methods use the same embeddings. If a new superfamily doesn't sit near proteins with the same job, neither method can find it.

    Look back at the GO-term table in the live session. Terms that span many superfamilies, like DNA binding, transfer well. GTPase activity, with 90% of its proteins in one superfamily, does not.

    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bonus: how many neighbors?

    Try `k` = 1, 5, 10, 25 and 50. Plot F-max against `k`, one line per split. Which `k` is best? Why is `k` = 1 the worst?

    <details>
    <summary>Hint</summary>

    ```python
    _df = pl.DataFrame([
        {"split": name, "k": k, "f_max": f_max(knn_scores(train, test, k=k), Y[test])}
        for name, (train, test) in splits.items()
        for k in [1, 5, 10, 25, 50]
    ])

    alt.Chart(_df).mark_line(point=True).encode(
        x=alt.X("k:Q", scale=alt.Scale(type="log")),
        y=alt.Y("f_max:Q", title="F-max"),
        color="split:N",
    ).properties(width=500, height=300)
    ```

    </details>

    <details>
    <summary>Answer</summary>

    On a laptop run, `k` = 5 is best on the random split and `k` = 10 on the family split. With `k` = 1, every score is 0 or 1: you copy one protein's terms exactly. F-max needs graded scores to pick a good cutoff, and one neighbor can easily be wrong. Averaging over more neighbors smooths that out. Too many neighbors (50) pulls in distant proteins and blurs the signal again.

    </details>
    """)
    return


@app.cell
def _():
    # TODO: F-max vs k, one line per split
    return


if __name__ == "__main__":
    app.run()
