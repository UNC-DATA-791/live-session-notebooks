# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "polars==1.44.2",
#     "remotezip==0.12.6",
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

    return RemoteZip, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before picking a dataset, browse [ProteinGym's benchmark table](https://proteingym.org/benchmarks) to find a real dataset ID:

    1. Go to the link above.
    2. Make sure the first dropdown says **DMS Substitution**.
    3. Click **Individual View** (top right).
    4. Copy any value from the **DMS ID** column and paste it into the box below.
    """)
    return


@app.cell
def _(mo):
    text_area = mo.ui.text_area(
        label="Which ProteinGym dataset did you choose?",
        placeholder="e.g. Q837P4_ENTFA_Meier_2023",
        value="Q837P4_ENTFA_Meier_2023",
    )
    text_area
    return (text_area,)


@app.cell
def _(RemoteZip, pl, text_area):
    PROTEINGYM_SUBSTITUTIONS_URL = (
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
        "DMS_ProteinGym_substitutions.zip"
    )

    with RemoteZip(PROTEINGYM_SUBSTITUTIONS_URL) as _zf:
        _match = next(n for n in _zf.namelist() if text_area.value in n)
        with _zf.open(_match) as _f:
            df = pl.read_csv(_f)

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histogram of DMS scores
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot a histogram of `DMS_score` from `df`. Bin it into about 40 bins.

    <details>
    <summary>Hint</summary>

    ```python
    alt.Chart(df, width=600).mark_bar().encode(
        x=alt.X('DMS_score').bin(maxbins=40),
        y=alt.Y('count()')
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chart of mutants per position
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Write a `parse_mutation` function that takes a mutation ID like `A192M` (wildtype amino acid, position, mutant amino acid) and returns a dict with `wildtype`, `position`, and `mutant_aa`. A multi-mutant like `A192M:A192L` (two mutations separated by `:`) should return `None`.

    The tests below already check both cases. Instead of writing `parse_mutation` yourself, write a prompt describing what it needs to do and have your notebook agent implement it so both tests pass.
    """)
    return


@app.cell
def _():
    def parse_mutation(mut='A192M'):
        pass


    def test_parse_mutation_returns_correct_data():
        result = parse_mutation('A192M')
        assert result == {'wildtype': 'A', 'position': 192, 'mutant_aa': 'M'}


    def test_parse_mutation_returns_na_for_multimutants():
        assert parse_mutation('A192M:A192L') is None


    test_parse_mutation_returns_correct_data()
    test_parse_mutation_returns_na_for_multimutants()
    return (parse_mutation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use `parse_mutation` to add `wildtype`, `position`, and `mutant_aa` columns to `df` (call the result `df_parsed`), then plot a bar chart of mutant counts per position.

    <details>
    <summary>Hint</summary>

    ```python
    df_parsed = df.with_columns(
        pl.col('mutant')
        .map_elements(
            parse_mutation,
            return_dtype=pl.Struct({'wildtype': pl.Utf8, 'position': pl.Int64, 'mutant_aa': pl.Utf8}),
        )
        .alias('parsed')
    ).unnest('parsed')

    alt.Chart(df_parsed, width=1000).mark_bar().encode(
        x=alt.X('position'),
        y=alt.Y('count()'),
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a mutant sequence from your Protein Gym dataset
    """)
    return


@app.cell
def _(df, mo):
    mutant_id_input = mo.ui.text(
        label='Mutant ID',
        placeholder='A192M',
        value=df['mutant'][0],
    )
    mutant_id_input
    return (mutant_id_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "⚠️ **Heads up:** the cell below will error until you finish the "
            "`parse_mutation` and `df_parsed` exercises above — it depends on both."
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The follow will return an error until the above sections are complete.
    """)
    return


@app.cell
def _(df_parsed, mo, mutant_id_input, parse_mutation, pl):
    _ref = df_parsed.filter(pl.col('wildtype').is_not_null()).row(0, named=True)
    wt_sequence = (
        _ref['mutated_sequence'][:_ref['position'] - 1]
        + _ref['wildtype']
        + _ref['mutated_sequence'][_ref['position']:]
    )

    _parsed = parse_mutation(mutant_id_input.value)
    mutant_sequence = (
        wt_sequence[:_parsed['position'] - 1]
        + _parsed['mutant_aa']
        + wt_sequence[_parsed['position']:]
    )

    mo.vstack([
        mo.md(f'**Wild type:**\n\n`{wt_sequence}`'),
        mo.md(f'**Mutant ({mutant_id_input.value}):**\n\n`{mutant_sequence}`'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find the embedding vector for your mutant
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This part uses a pretrained protein language model (ESM) to turn your mutant sequence into a vector, so you can compare it to the wild type numerically instead of just eyeballing the letters.

    Build this code with molab's free built-in agent instead of from scratch. Use the following prompt:

    > Look at the amino acid and sequence embedding cells in `unit6-live-session/live-session.py` on GitHub ([link](https://github.com/UNC-DATA-791/live-session-notebooks/blob/main/unit6-live-session/live-session.py)) and adapt them to embed the mutant sequence I just built, then compare it to the wild-type embedding.
    """)
    return


@app.cell
def _():
    # Embedding code here


    return


if __name__ == "__main__":
    app.run()
