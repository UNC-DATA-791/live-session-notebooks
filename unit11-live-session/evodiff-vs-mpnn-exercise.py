# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.4",
#     "polars==1.42.1",
#     "pyarrow==25.0.0",
#     "scipy==1.18.0",
#     "sentencepiece==0.2.3",
#     "torch==2.11.0",
#     "transformers==5.14.1",
#     "umap-learn==0.5.12",
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _():
    import altair as alt
    import numpy as np
    import polars as pl
    from scipy.linalg import sqrtm
    from umap import UMAP
    return UMAP, alt, np, pl, sqrtm

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise: do ProteinMPNN sequences look like natural proteins?

    ProteinMPNN writes a sequence for a **structure**. EvoDiff-Seq writes a sequence from **sequence data alone**. Both make sequences that look like proteins. Which ones look more like **natural** proteins?

    We will ask this the way the EvoDiff authors did: embed the sequences with the protein language model ProtT5, then compare the groups in embedding space.

    ## Before you start: read the paper

    Read the EvoDiff paper, [Protein generation with evolutionary diffusion: sequence is all you need](https://www.biorxiv.org/content/10.1101/2023.09.11.556673.full.pdf) (Alamdari et al., 2023). Find **Figure 3A** and read its caption. Then answer these in your own words:

    - What is plotted in the UMAP? What does each point stand for?
    - What does the FPD number tell you?
    - Which groups did the authors compare in that panel? How are the sequences in each group made?

    We will make our own version of Figure 3A. We skip Figure 3B (secondary structure).
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data

    Three groups of 200 sequences each:

    - **Natural.** Swiss-Prot proteins, 100 to 250 residues.
    - **ProteinMPNN.** One design for each of 200 *other* Swiss-Prot proteins, using their AlphaFold backbones. The natural group is held out, so none of these backbones come from the natural group.
    - **EvoDiff-Seq.** Sequences from the 38M-parameter EvoDiff-Seq model (order-agnostic diffusion, no structure, no conditioning). We matched the lengths to the natural group.
    """)
    return

@app.cell
def _(mo, pl):
    _dir = mo.notebook_dir()
    natural = pl.read_csv(_dir / "natural_sequences.csv")
    mpnn = pl.read_csv(_dir / "mpnn_sequences.csv")
    evodiff = pl.read_csv(_dir / "evodiff_sequences.csv")

    groups = ["Natural", "ProteinMPNN", "EvoDiff-Seq"]
    sequences = pl.concat(
        [
            natural.select("sequence", "length").with_columns(group=pl.lit("Natural")),
            mpnn.select("sequence", "length").with_columns(group=pl.lit("ProteinMPNN")),
            evodiff.select("sequence", "length").with_columns(group=pl.lit("EvoDiff-Seq")),
        ]
    )
    sequences.group_by("group").agg(
        n=pl.len(),
        mean_length=pl.col("length").mean().round(1),
        first_sequence=pl.col("sequence").first().str.slice(0, 40) + "...",
    ).sort("group")
    return groups, sequences

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embeddings

    ProtT5 turns each sequence into a list of 1024 numbers, one per residue, and we average them over the residues. Each sequence becomes a point in a 1024-dimensional space.

    The cell below loads saved embeddings (`prot_t5_embeddings.npy`). If the file is missing, it runs ProtT5 on the GPU. That takes a few minutes.
    """)
    return

@app.cell
def _(mo, np, sequences):
    import re

    def embed_prot_t5(seqs, batch_size=16):
        """Mean-pooled ProtT5-XL encoder embeddings, one 1024-number row per sequence."""
        import sentencepiece as spm
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import T5EncoderModel

        name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sp = spm.SentencePieceProcessor(model_file=hf_hub_download(name, "spiece.model"))
        model = T5EncoderModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        out = []
        for i in range(0, len(seqs), batch_size):
            ids = [sp.encode(" ".join(re.sub(r"[UZOB]", "X", s))) + [1] for s in seqs[i : i + batch_size]]
            width = max(map(len, ids))
            x = torch.tensor([r + [0] * (width - len(r)) for r in ids]).to(device)
            mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in ids]).to(device)
            with torch.no_grad():
                h = model(input_ids=x, attention_mask=mask).last_hidden_state.float()
            for j, r in enumerate(ids):
                out.append(h[j, : len(r) - 1].mean(dim=0).cpu().numpy())  # drop the end token
        return np.stack(out)

    _cache = mo.notebook_dir() / "prot_t5_embeddings.npy"
    if _cache.exists():
        embeddings = np.load(_cache)
    else:
        embeddings = embed_prot_t5(sequences["sequence"].to_list())
        np.save(_cache, embeddings)
    embeddings.shape
    return (embeddings,)

@app.cell
def _(embeddings, np, sequences):
    # One matrix per group, in the same order as the rows of `sequences`
    by_group = {
        g: embeddings[(sequences["group"] == g).to_numpy()]
        for g in ["Natural", "ProteinMPNN", "EvoDiff-Seq"]
    }
    {g: m.shape for g, m in by_group.items()}
    return (by_group,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Fréchet distance

    The **Fréchet distance** compares two clouds of points. Each cloud is described by its **mean** (where it sits) and its **covariance** (how wide it is, and in which directions):

    $$FD = \lVert \mu_A - \mu_B \rVert^2 + \mathrm{Tr}\left(C_A + C_B - 2\,(C_A C_B)^{1/2}\right)$$

    On ProtT5 embeddings it is called the **FPD**. Lower means the two groups look more alike.

    Write `frechet_distance(a, b)`. Each input is a matrix with one row per sequence.

    /// details | Hint
    ```python
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(cov_a @ cov_b).real   # the matrix square root can have a tiny imaginary part
    fd = ((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean)
    ```
    Return `float(fd)`.
    ///
    """)
    return

@app.cell
def _():
    # TODO: fill in the body. Use the formula above.
    def frechet_distance(a, b):
        return ...
    return (frechet_distance,)

@app.cell
def _(by_group, frechet_distance, mo):
    _same = frechet_distance(by_group["Natural"], by_group["Natural"])
    mo.stop(_same is ..., mo.md("Fill in the TODO above."))
    mo.md(
        f"Sanity check: the distance from the natural group to itself should be about 0. You got **{_same:.3f}**."
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. FPD of each group against natural

    Compute the FPD between the **natural** group and each of the other two groups. Put the results in a Polars DataFrame called `fpd` with the columns `group` and `fpd`.

    /// details | Hint
    ```python
    fpd = pl.DataFrame(
        {
            "group": ["ProteinMPNN", "EvoDiff-Seq"],
            "fpd": [frechet_distance(by_group["Natural"], by_group[g]) for g in ["ProteinMPNN", "EvoDiff-Seq"]],
        }
    )
    ```
    ///
    """)
    return

@app.cell
def _():
    # TODO: build the DataFrame `fpd` with the columns "group" and "fpd".
    fpd = ...
    return (fpd,)

@app.cell
def _(fpd, mo):
    mo.stop(fpd is ..., mo.md("Fill in the TODO above."))
    fpd
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. UMAP

    A UMAP squeezes the 1024 dimensions into 2, and keeps points that are close together in 1024 dimensions close together on the page.

    Fit a `UMAP` on **all** the embeddings (all three groups together). Use `n_neighbors=15`, `min_dist=0.1`, `random_state=0`. Save the 2D coordinates in a DataFrame called `umap_df` with columns `x`, `y` and `group`.

    /// details | Hint
    ```python
    coords = UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(embeddings)
    umap_df = pl.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "group": sequences["group"]})
    ```
    ///
    """)
    return

@app.cell
def _():
    # TODO: fit the UMAP on `embeddings` and build `umap_df`.
    umap_df = ...
    return (umap_df,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Plot it, like Figure 3A

    Make a scatter plot of `umap_df` with Altair. Color the points by group. Use gray for natural, orange for ProteinMPNN and blue for EvoDiff-Seq, as in the paper. Put each group's FPD in its legend label.

    /// details | Hint
    ```python
    labels = {"Natural": "Natural", **{r["group"]: f'{r["group"]} (FPD {r["fpd"]:.0f})' for r in fpd.iter_rows(named=True)}}
    plot_df = umap_df.with_columns(label=pl.col("group").replace(labels))
    alt.Chart(plot_df).mark_circle(size=40, opacity=0.6).encode(
        x="x:Q", y="y:Q",
        color=alt.Color("label:N", scale=alt.Scale(domain=list(labels.values()), range=["#8c8c8c", "#e8832a", "#2f6fdb"])),
    )
    ```
    ///
    """)
    return

@app.cell
def _():
    # TODO: make the scatter plot and save it in `umap_chart`.
    umap_chart = ...
    return (umap_chart,)

@app.cell
def _(fpd, mo, umap_chart, umap_df):
    mo.stop(
        fpd is ... or umap_df is ... or umap_chart is ...,
        mo.md("Fill in the TODOs above."),
    )
    umap_chart
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What do you see?

    - Which group has the lower FPD against natural, ProteinMPNN or EvoDiff-Seq? Does the UMAP agree?
    - You might expect that seeing a structure helps ProteinMPNN look more natural. Is that what you found? What could explain your result? Think about what each model was trained to do, how ProteinMPNN was sampled (low temperature, one design per backbone) and how much of the native sequence it recovered.
    - The natural group is held out from the ProteinMPNN backbones. Why does that matter for a fair comparison?
    - We used 200 sequences per group, and the paper used 1000. We have 1024 dimensions and only 200 points per group. How could that affect the FPD?
    - FPD tells you the groups look alike. Does it tell you the ProteinMPNN or EvoDiff sequences fold, or work? What would you run next? (Think back to ESMFold, pLDDT and self-consistency in the live session.)

    ### Extension: how big is the noise?

    Split the natural group at random into two halves of 100 sequences and compute the FPD between the halves. Both halves are natural, so the true distance is 0. Compare that number with your FPDs. What does it tell you about small differences?

    /// details | Hint
    ```python
    rng = np.random.default_rng(0)
    order = rng.permutation(len(by_group["Natural"]))
    half_a = by_group["Natural"][order[:100]]
    half_b = by_group["Natural"][order[100:]]
    frechet_distance(half_a, half_b)
    ```
    ///
    """)
    return

if __name__ == "__main__":
    app.run()
