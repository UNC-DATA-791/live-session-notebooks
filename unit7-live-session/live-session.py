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
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from transformers import AutoTokenizer, EsmModel

    return (
        AutoTokenizer,
        EsmModel,
        GroupShuffleSplit,
        LogisticRegression,
        OneVsRestClassifier,
        alt,
        io,
        np,
        pl,
        roc_auc_score,
        torch,
        train_test_split,
        urllib,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### What have we learned so far?
    - Unit 6: variant effect prediction
        - Clinical: is a mutation like R206H harmful? (ClinPred)
        - Protein optimization: benchmark models on [ProteinGym](https://proteingym.org/)
    - Zero-shot PLM scores: how "natural" does a sequence look to the model?
    - Regression on protein embeddings to predict fitness

    ### Functional annotation
    - What does a protein *do*? Predict Gene Ontology (GO) terms from sequence alone
    - Zero-shot vs. supervised: where each one fits
    - PLM embeddings as features for a supervised model
    - How to build a validation set that doesn't fool you
    - [CAFA 6](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction), start to finish
    - Exercise: nearest-neighbor label transfer

    ### What's next?
    - Unit 8: active learning. Let a model pick which variants to test in the lab (train → score → select → experiment)

    ### Questions
    - Project 2, Notebook 3?
    - ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The annotation gap

    UniProt holds more than **250 million** protein sequences. Fewer than **1%** have a known function that has been validated in the lab..

    Function is written in the **Gene Ontology (GO)**. GO has three parts:

    | Subontology | Asks | Example |
    | --- | --- | --- |
    | Molecular Function (MF) | What does it do, chemically? | ATP binding |
    | Biological Process (BP) | What larger job is it part of? | DNA repair |
    | Cellular Component (CC) | Where in the cell is it? | nucleus |

    One protein can have many GO terms. So this is **multilabel** classification, with thousands of labels. The labels form a hierarchy: "ATP binding" is a kind of "nucleotide binding".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zero-shot vs. supervised

    | | Zero-shot PLM score | Supervised model on PLM embeddings |
    | --- | --- | --- |
    | Needs lab labels? | No | Yes |
    | Answers | "Does this sequence look natural?" | "Does this protein have function X?" |
    | Good for | Variant effects (Unit 6) | Functional annotation (today) |

    A zero-shot score can predict if a mutation breaks a protein. It can't tell you the protein is a kinase. For that we need a *supervised* model.

    The workflow for today:

    1. Turn each sequence into one embeddings vector with ESM2 (same trick as Unit 6).
    2. Train one classifier per GO term on those vectors.
    3. Score it on proteins it hasn't seen, and be careful what "hasn't seen" means.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CAFA: the benchmark

    **CAFA** (Critical Assessment of Functional Annotation) has run since 2010. [Round 6 is on Kaggle](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction).

    - It is **prospective**. You predict now. You are scored later, on lab annotations that don't exist yet.
    - Covers all three GO subontologies (MF, BP, CC).
    - Metric: protein-centric **F-max**, with each term weighted by its information accretion (rare, specific terms count more).

    A good starting point: [this CAFA 6 notebook](https://www.kaggle.com/code/olaflundstrom/cafa-6-protein-function-prediction-dtu-proteomics). ProtT5 embeddings → one logistic regression per GO term → push scores up to parent terms. We'll build a small version of the same thing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get labeled proteins

    CAFA data sits behind a Kaggle login, so we pull a stand-in from UniProt: reviewed human proteins, 50–300 residues on a laptop (up to 1000 on a GPU), with MF GO terms and protein family.
    """)
    return


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
    proteins
    return (proteins,)


@app.cell
def _(pl, proteins):
    # term id -> readable name, e.g. GO:0005524 -> "ATP binding"
    go_names = (
        proteins.select(
            pl.col("go_mf").str.split("; ").explode().str.extract_groups(r"^(.*) \[(GO:\d{7})\]$")
        )
        .unnest("go_mf")
        .rename({"1": "go_name", "2": "go_term"})
        .unique("go_term")
    )

    term_counts = (
        proteins.explode("go_terms")
        .group_by(pl.col("go_terms").alias("go_term"))
        .len()
        .join(go_names, on="go_term")
        .sort("len", descending=True)
    )
    term_counts.head(20)
    return (term_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We keep the 8 most common terms. Real CAFA models predict thousands.

    `sample` holds the proteins we model. Its rows line up with `X` and `Y`, which we make below:

    - row 1084 of `sample` is a protein
    - row 1084 of `X` is that protein's embedding
    - row 1084 of `Y` is its 0/1 GO labels
    """)
    return


@app.cell
def _(n_terms, pl, proteins, term_counts):
    top_terms = term_counts["go_term"].head(n_terms).to_list()

    # like CAFA, only score proteins that have at least one true label
    sample = proteins.filter(pl.col("go_terms").list.set_intersection(top_terms).list.len() > 0)
    sample.height, sample["superfamily"].n_unique()
    return sample, top_terms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embed with ESM2

    Same as Unit 6: run each sequence through ESM2 and average the residue vectors into one vector per protein.
    """)
    return


@app.cell
def _(AutoTokenizer, EsmModel, device, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).eval().to(device)
    model_name, device
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Batches and padding

    The model reads 32 proteins at a time. Every row in the batch, all the tokenized sequences, needs to be the same length. The tokenizer fills short proteins with blank tokens up to the longest one. That fill is the **padding**.

    ```
    MKTAYIAKQR....    <- real protein, then padding
    MSEQ..........
    MKVLAAGIVGLLLAGC  <- longest one sets the width
    ```

    Two things to watch:

    - **Correctness**: the tokenizer also adds a start token (`<cls>`) and an end token (`<eos>`) to each protein. We build a mask that is 1 for a real residue and 0 for everything else, then average only the real residues. So padding and these extra tokens don't change a protein's vector.
    - **Speed**: the model still does work on every blank. So we sort proteins by length first. Then each batch holds proteins of about the same length, and there is little to fill. At the end we put the rows back in their original order.

    On this data, sorting cuts padding from about a third of the work to about 2% (from about half to 1% on a GPU, with longer proteins).
    """)
    return


@app.cell
def _(device, mo, model, np, sample, tokenizer, torch):
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
            _batches.append(((_hidden * _mask).sum(1) / _mask.sum(1)).cpu().numpy()) # sequence level emb.

    X = np.empty((len(_seqs), model.config.hidden_size), dtype=np.float32)
    X[_order] = np.concatenate(_batches)
    X.shape
    return (X,)


@app.cell
def _(np, sample, top_terms):
    # one row per protein, one 0/1 column per GO term
    Y = np.array([[t in terms for t in top_terms] for terms in sample["go_terms"].to_list()], dtype=int)
    Y.shape, Y.sum(axis=0)
    return (Y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One model per GO term

    `OneVsRestClassifier` trains one logistic regression per column of `Y`. Each one answers "yes or no" for its GO term.
    """)
    return


@app.cell
def _(LogisticRegression, OneVsRestClassifier, X, Y, roc_auc_score):
    def fit_and_score(train_idx, test_idx):
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
        clf.fit(X[train_idx], Y[train_idx])
        probs = clf.predict_proba(X[test_idx])
        y_test = Y[test_idx]
        aucs = [
            roc_auc_score(y_test[:, j], probs[:, j]) if 0 < y_test[:, j].sum() < len(y_test) else None
            for j in range(Y.shape[1])
        ]
        return probs, y_test, aucs

    return (fit_and_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How you split matters

    Proteins belong to functional families. Two members of one family or similar in sequence *and* function.

    - **Random split**: family members land on both sides. The model can "remember" a close cousin from training.
    - **Family split**: whole superfamilies are held out. The test set looks more like a brand-new protein.

    We group by superfamily, not family. Rab and Ras are two "families", but both are small GTPases. If we split them apart, close cousins still leak across.

    A superfamily and a GO term are two different ways to group proteins. A superfamily groups by shared ancestry, and each protein has one. A GO term groups by job, and one term can span many superfamilies. The table below shows how spread out each of our terms is.

    Watch the GTPase terms. 90% of the GTPase proteins are small GTPases, which are all one superfamily. So the family split puts nearly all of them in the test set. The model learns from only a few distant GTPases, and its AUC for "GTPase activity" can fall far below 0.5.

    The family split is the honest one. It is closer to what CAFA does: predict for proteins nobody has labeled yet.
    """)
    return


@app.cell
def _(pl, sample, term_counts, top_terms):
    # how spread out is each GO term across superfamilies?
    _long = (
        sample.explode("go_terms")
        .filter(pl.col("go_terms").is_in(top_terms))
        .rename({"go_terms": "go_term"})
    )

    (
        _long.group_by("go_term", "superfamily")
        .len()
        .group_by("go_term")
        .agg(
            pl.col("len").sum().alias("proteins"),
            pl.len().alias("superfamilies"),
            pl.col("superfamily").sort_by("len", descending=True).first().alias("biggest_superfamily"),
            (pl.col("len").max() / pl.col("len").sum()).round(2).alias("share_in_biggest"),
        )
        .join(term_counts.select("go_term", "go_name"), on="go_term")
        .select("go_name", "proteins", "superfamilies", "biggest_superfamily", "share_in_biggest")
        .sort("superfamilies")
    )
    return


@app.cell
def _(GroupShuffleSplit, X, fit_and_score, np, sample, train_test_split):
    _rows = np.arange(len(X))
    _groups = sample["superfamily"].to_numpy()

    # (train_idx, test_idx) for each split
    splits = {
        # random: members of one superfamily can land on both sides
        "random": train_test_split(_rows, test_size=0.2, random_state=0),
        # family: each superfamily lands on one side only
        "family": next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, groups=_groups)),
    }

    results = {}
    for _name, (_train, _test) in splits.items():
        results[_name] = fit_and_score(_train, _test)
    return (results,)


@app.cell
def _(pl, results, term_counts, top_terms):
    auc_df = (
        pl.DataFrame([
            {"split": name, "go_term": term, "auc": auc}
            for name, (_, _, aucs) in results.items()
            for term, auc in zip(top_terms, aucs)
        ])
        .drop_nulls()
        .join(term_counts.select("go_term", "go_name"), on="go_term")
    )
    auc_df
    return (auc_df,)


@app.cell
def _(alt, auc_df, pl, top_terms):
    axis_kws = dict(titleFontSize=16, labelFontSize=13, titleFontWeight="bold")

    _points = (
        alt.Chart(auc_df)
        .mark_point(size=150, filled=True)
        .encode(
            x=alt.X("auc:Q", title="ROC AUC", scale=alt.Scale(domain=[0, 1])).axis(**axis_kws),
            y=alt.Y("go_name:N", title=None, sort="-x").axis(labelFontSize=13),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(domain=["random", "family"], range=["#2a78d6", "#d03b3b"]),
                legend=alt.Legend(titleFontSize=16, labelFontSize=14, symbolSize=200),
            ),
            tooltip=["go_name", "split", alt.Tooltip("auc:Q", format=".3f")],
        )
    )

    # AUC 0.5 = coin flip
    _coin_flip = alt.Chart(pl.DataFrame({"auc": [0.5]})).mark_rule(strokeDash=[6, 4], color="gray").encode(x="auc:Q")

    (_coin_flip + _points).properties(
        width=500, height=max(360, 18 * len(top_terms)), title="Per-term AUC: random vs. family split"
    )
    return


@app.cell
def _(auc_df, pl):
    auc_df.group_by("split").agg(pl.col("auc").mean().alias("mean_auc"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scoring like CAFA: F-max

    CAFA scores each **protein**, not each term. Pick a cutoff $t$. Call every term with score $\ge t$ a prediction. Then:

    - **precision**: of the terms we predicted, how many were right? Averaged over proteins with at least one prediction.
    - **recall**: of the true terms, how many did we find? Averaged over all proteins.

    Sweep $t$ from 0 to 1 and keep the best F1. That's **F-max**. (CAFA 6 also weights each term by how specific it is. We skip that here.)
    """)
    return


@app.cell
def _(np, results):
    def f_max(probs, y_true):
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

    {name: round(f_max(probs, y_test), 3) for name, (probs, y_test, _) in results.items()}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From here to a CAFA 6 submission

    - **More terms**: thousands of GO terms across MF, BP and CC, not 8.
    - **Propagate up the hierarchy**: if you predict "ATP binding", you also predict "nucleotide binding". A parent's score should be at least as high as its child's.
    - **Bigger embeddings**: ProtT5 or the larger ESM2 models.
    - **Submit**: one row per (protein, GO term, score) for the CAFA test proteins.
    """)
    return


if __name__ == "__main__":
    app.run()
