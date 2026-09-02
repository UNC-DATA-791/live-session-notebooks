# /// script
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.2",
#     "pandas==3.0.5",
#     "polars==1.44.1",
#     "pyarrow==25.0.1",
#     "statsmodels==0.15.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practice with negative binomial regression
    """)
    return


@app.cell
def _():
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import altair as alt
    import polars as pl

    return pl, sm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll use this 🦀 Crab satellites data set. The data comes from a study of 173 female horseshoe crabs recording each female's carapace width (plus weight, color, and spine condition) and the count of male "satellites" clustering around her nest.
    """)
    return


@app.cell
def _(pl, sm):
    crabsat = pl.from_pandas(sm.datasets.get_rdataset('CrabSatellites', 'vcdExtra').data)

    crabsat
    return (crabsat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot `satellites` (y-axis) versus `width` (x-axis).
    """)
    return


@app.cell
def _():
    # Chart code here



    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell calculates variance in satellites count versus mean width.
    """)
    return


@app.cell
def _(crabsat, pl):
    binned = (
        crabsat
        .with_columns(pl.col('width').qcut(8).alias('bin'))
        .group_by('bin')
        .agg(pl.col('satellites').mean().alias('mean_width'), pl.col('satellites').var().alias('sat_variance')))

    binned
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot `mean_width` (x-axis) versus `sat_variance` (y-axis).
    """)
    return


@app.cell
def _():
    # Chart goes here



    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fit a negative binomial model with the formula `"satellites ~ width"`. Call your model `nb`.
    """)
    return


@app.cell
def _():



    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Visualize your model. You can get predictions like this `nb.predict(crabsat)`.
    """)
    return


@app.cell
def _():




    return


if __name__ == "__main__":
    app.run()
