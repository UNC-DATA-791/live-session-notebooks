# /// script
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.2",
#     "pandas==3.0.5",
#     "polars==1.44.1",
#     "statsmodels==0.14.6",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Poisson vs. negative binomial: the `ships` data

    `ships` comes from the R **MASS** package (McCullagh & Nelder's wave-damage data). Each row is a *group* of ships, not a single ship:

    | column | meaning |
    | --- | --- |
    | `type` | ship type, `A`-`E` |
     `year` | year of construction (60, 65, 70, 75) |
    | `period` | period of operation (60 or 75) |
    | `service` | aggregate months of service for the group — the **exposure** |
    | `incidents` | number of damage incidents — the **response** |
    """)
    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    ships = (
        pl.read_csv(
            "https://raw.githubusercontent.com/UNC-DATA-791/"
            "live-session-notebooks/main/unit2-live-session/data/ships.csv"
        )
        .filter(pl.col('service')>0)
        .with_columns((pl.col('incidents') / pl.col('service')).alias('rate'))
    )

    ships
    return alt, mo, np, pd, pl, ships, smf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exposure and the offset

    A group of ships in service for 44,882 months will have more incidents regardless of safety. We are interested in the rate of incidents by month.

    To account for intervals of different length, we need to add an offset term:

    $$\log(\mu_i) = \log(\text{service}_i) + \beta_0 + \beta_1 x_{i1} + \dots$$

    The `log(service)` term coefficient is fixed at 1. The remaining coefficients therefore measure change in *rate*, not raw indicent count.

    **TODO:**

    1. Make `ships_pd`, a pandas `DataFrame` from `ships`. (`statsmodels` formulas want pandas (not polars). `pd.DataFrame(ships.to_dicts())` works.)
    2. Make `offset`, a numpy array of `log(service)`.
    """)
    return


@app.cell
def _():
    # ships_pd and offset code goes here


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make an OLS, Poisson, and negative binomial model of incidents using the formulas provided.

    **TODO:** fill in the `...` blanks with `ships_pd` and `offset`.
    """)
    return


@app.cell
def _(smf):
    formula = "incidents ~ type + year + period"
    formula_ols = "rate ~ type + year + period"

    # Fill in the missing arguments
    ols = smf.ols(formula_ols, data=...).fit()
    pois = smf.poisson(formula, data=..., offset=...).fit(disp=0)
    nb = smf.negativebinomial(formula, data=..., offset=...).fit(disp=0)

    return nb, pois


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How confident is each model?

    For each group, take actual minus predicted. Then take a random draw from a distribution parameterized by our fitted model, and subtract that same predicted mean.

    If the random draws show similar deviation to the actual minus predicted values, then the model is measuring dispersion correctly.

    Below is a table of these "deviation" values.
    """)
    return


@app.cell
def _(nb, np, offset, pl, pois, ships_pd):
    ships_mu_p = pois.predict(ships_pd).to_numpy()
    ships_mu_n = nb.predict(ships_pd, offset=offset).to_numpy()

    ships_alpha = float(nb.params["alpha"])
    ships_rng = np.random.default_rng(0)
    ships_size = 1.0 / ships_alpha  # NB2 -> numpy's (n, p) parameterisation

    ships_err_df = (
        pl.DataFrame(
            {
                "actual": ships_pd["incidents"].to_numpy() - ships_mu_n,
                "poisson": ships_rng.poisson(ships_mu_p) - ships_mu_p,
                "neg_binomial": ships_rng.negative_binomial(
                    ships_size, ships_size / (ships_size + ships_mu_n)
                )
                - ships_mu_n,
            }
        )
        .unpivot(variable_name="source", value_name="residual")
        .sort("residual")
        .with_columns(
            cdf=pl.int_range(1, pl.len() + 1).over("source") / pl.len().over("source")
        )
    )

    ships_err_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Altair to make a chart of CDF (x-axis) versus residual (y_axis) grouped by source.
    """)
    return


@app.cell
def _():
    # Chart code goes here


    return


if __name__ == "__main__":
    app.run()
