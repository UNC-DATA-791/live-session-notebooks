# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "polars==1.44.2",
#     "remotezip==0.12.6",
#     "scikit-learn==1.8.0",
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
    from remotezip import RemoteZip
    import polars as pl
    import altair as alt
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import AutoTokenizer, EsmForMaskedLM

    return (
        AutoTokenizer,
        EsmForMaskedLM,
        RemoteZip,
        alt,
        pl,
        roc_auc_score,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Can ESM-1b spot a disease mutation?

    ESM-1b has never seen a patient. It has only read a lot of protein sequences. Today we ask it to score human missense variants anyway, then check its answers against clinical labels (**Benign** or **Pathogenic**) from ProteinGym's clinical benchmark.

    The trick: if ESM-1b thinks the mutant amino acid is *less likely* than the wild-type one at that spot, the mutation is probably bad.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pick a protein

    The default is **TP53** (`NP_000537.3`), the "guardian of the genome". It's mutated in about half of all cancers.

    Want to swap it? Try one of these RefSeq IDs:

    | Protein | ID | Why it's fun |
    | --- | --- | --- |
    | Hemoglobin beta | `NP_000509.1` | Sickle cell and the thalassemias |
    | Hemoglobin alpha | `NP_000508.1` | Its partner in crime |
    | TP53 | `NP_000537.3` | Cancer's favorite target |

    Keep it under 1,022 residues. That's all ESM-1b can read at once.
    """)
    return


@app.cell
def _(mo):
    text_area = mo.ui.text_area(
        label="Which ProteinGym clinical protein did you choose?",
        placeholder="e.g. NP_000537.3",
        value="NP_000537.3",
    )
    text_area
    return (text_area,)


@app.cell
def _(RemoteZip, pl, text_area):
    PROTEINGYM_CLINICAL_URL = (
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
        "clinical_ProteinGym_substitutions.zip"
    )

    with RemoteZip(PROTEINGYM_CLINICAL_URL) as _zf:
        _match = next(n for n in _zf.namelist() if text_area.value in n)
        with _zf.open(_match) as _f:
            df = pl.read_csv(_f).select("mutant", "DMS_bin_score", "protein_sequence")

    wt_sequence = df["protein_sequence"][0]
    df = df.drop("protein_sequence")
    df
    return df, wt_sequence


@app.cell
def _(alt, df):
    alt.Chart(df, width=200).mark_bar().encode(
        x=alt.X("DMS_bin_score", title=None),
        y=alt.Y("count()"),
        color=alt.Color("DMS_bin_score", legend=None),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load ESM-1b

    This is a 650M parameter model, so the first download takes a minute.
    """)
    return


@app.cell
def _(AutoTokenizer, EsmForMaskedLM, torch):
    model_name = "facebook/esm1b_t33_650M_UR50S"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    model_name, device
    return device, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ask ESM-1b about every position at once

    Last unit we masked one spot and asked ESM what goes there. Here we skip the mask. We hand ESM-1b the whole wild-type sequence once. At every position it gives back a probability for all 20 amino acids.

    `log_probs` has one row per residue and one column per token. Row `0` is residue `1`.
    """)
    return


@app.cell
def _(device, model, tokenizer, torch, wt_sequence):
    _inputs = tokenizer(wt_sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        _logits = model(**_inputs).logits

    # drop the <cls> and <eos> tokens on either end so row i = residue i + 1
    log_probs = torch.log_softmax(_logits[0, 1:-1], dim=-1).cpu()
    log_probs.shape
    return (log_probs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Score a mutation

    The score for a mutation like `R175H` is:

    $$
    \text{score} = \log P(\text{H at 175}) - \log P(\text{R at 175})
    $$

    Negative means ESM-1b likes the wild type better, so the mutation is probably bad. Positive means ESM-1b likes the mutant better.

    Finish `esm_score` so both tests pass.

    <details>
    <summary>Hint</summary>

    ```python
    def esm_score(mut):
        wt, pos, mt = mut[0], int(mut[1:-1]), mut[-1]
        wt_id = tokenizer.convert_tokens_to_ids(wt)
        mt_id = tokenizer.convert_tokens_to_ids(mt)
        return (log_probs[pos - 1, mt_id] - log_probs[pos - 1, wt_id]).item()
    ```

    </details>
    """)
    return


@app.cell
def _(wt_sequence):
    def esm_score(mut):
        pass


    def test_esm_score_returns_a_float():
        assert isinstance(esm_score(f"{wt_sequence[0]}1A"), float)


    def test_esm_score_is_zero_for_no_change():
        assert esm_score(f"{wt_sequence[0]}1{wt_sequence[0]}") == 0


    test_esm_score_returns_a_float()
    test_esm_score_is_zero_for_no_change()
    return (esm_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does ESM-1b agree with the clinic?

    Add an `esm_score` column to `df` (call the result `df_scored`). Then plot the `esm_score` distribution for **Benign** and **Pathogenic** variants on top of each other.

    <details>
    <summary>Hint</summary>

    ```python
    df_scored = df.with_columns(
        pl.col('mutant').map_elements(esm_score, return_dtype=pl.Float64).alias('esm_score')
    )

    alt.Chart(df_scored, width=600).mark_bar(opacity=0.6).encode(
        x=alt.X('esm_score').bin(maxbins=30),
        y=alt.Y('count()').stack(None),
        color=alt.Color('DMS_bin_score'),
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: add an esm_score column to df (call it df_scored), then plot Benign vs Pathogenic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Put a number on it. Compute the AUROC with `roc_auc_score`. Treat **Benign** as the positive class, since a higher score should mean "more OK".

    0.5 is a coin flip. 1.0 is perfect. Then look up ESM-1b on the [ProteinGym clinical leaderboard](https://proteingym.org/benchmarks) (switch the first dropdown to **Clinical Substitution**). How close did you get?

    <details>
    <summary>Hint</summary>

    ```python
    roc_auc_score(df_scored['DMS_bin_score'] == 'Benign', df_scored['esm_score'])
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: compute the AUROC of esm_score (Benign = positive class)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: mutate everything

    The clinic only has labels for variants that showed up in patients. ESM-1b can score all of them. Use `esm_score` to score all 19 substitutions at every position, then draw a heatmap with position on the x-axis and mutant amino acid on the y-axis.

    If you stuck with TP53, hover over the famous cancer hotspots R175, R248, and R273. Does ESM-1b flag all of them? Why might it miss one?

    <details>
    <summary>Hint</summary>

    ```python
    heatmap_df = pl.DataFrame([
        {'position': i + 1, 'mutant_aa': aa, 'esm_score': esm_score(f'{wt}{i + 1}{aa}')}
        for i, wt in enumerate(wt_sequence)
        for aa in 'ACDEFGHIKLMNPQRSTVWY'
    ])

    alt.Chart(heatmap_df, width=1000, height=300).mark_rect().encode(
        x=alt.X('position:O').axis(labels=False, ticks=False),
        y=alt.Y('mutant_aa:N'),
        color=alt.Color('esm_score:Q').scale(scheme='redblue', domainMid=0),
        tooltip=['position', 'mutant_aa', 'esm_score'],
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: score every substitution at every position and draw a heatmap
    return


if __name__ == "__main__":
    app.run()
