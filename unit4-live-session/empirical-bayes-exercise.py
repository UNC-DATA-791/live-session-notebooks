# /// script
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.3",
#     "pandas==3.0.5",
#     "polars==1.44.2",
#     "scipy==1.18.1",
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
    # Unit 4 exercise
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import altair as alt
    from scipy.optimize import minimize
    from scipy.special import gammaln
    from scipy.stats import gamma as gamma_dist

    return alt, gamma_dist, gammaln, minimize, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    County-level kidney cancer death counts and population for the U.S., covering two five-year periods (1980-84 and 1985-89). Deaths and average population are combined across both periods. Many counties are sparsely populated, so their raw death rates are noisy.
    """)
    return


@app.cell
def _(pl):
    URL = "https://raw.githubusercontent.com/robinryder/BDA-kidney/master/KidneyCancerClean.csv"
    PER_100K = 1e5

    kidney = (
        pl.read_csv(URL, skip_rows=4, null_values="NA")
        .select(
            location=pl.col("Location"),
            fips=pl.col("fips").cast(pl.Int64),
            deaths=pl.col("dc") + pl.col("dc.2"),           # 1980-84 + 1985-89
            pop=(pl.col("pop") + pl.col("pop.2")) / 2,      # average population
        )
        .filter(pl.col("pop") > 0)
        .with_columns(rate_raw=pl.col("deaths") / pl.col("pop"))
    )

    kidney
    return PER_100K, kidney


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot `rate_raw` (y-axis) versus `pop` (x-axis). Make sure you plot `pop` on a log axis.


    <details>
    <summary>Hint</summary>

    ```python
    alt.Chart(kidney, width=800, height=300).mark_point(size=8, strokeOpacity=0.65, fillOpacity=0.35).encode(
        x=alt.X('pop').scale(type='log'),
        y=alt.Y('rate_raw'),
        # tooltip=[PUT YOUR TOOLTIPS HERE]
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
    This code calculates our "prior distribution" for the rate of kidney cancer deaths.
    """)

    return


@app.cell
def _(PER_100K, gammaln, kidney, minimize, np):
    y, n = kidney["deaths"].to_numpy(), kidney["pop"].to_numpy()

    def neg_marginal_loglik(params):
        a, b = np.exp(params)                               # keep alpha, beta > 0
        return -np.sum(gammaln(a + y) - gammaln(a) - gammaln(y + 1)
                       + a * np.log(b) + y * np.log(n)
                       - (a + y) * np.log(b + n))
 
    m, v = np.mean(y / n), np.var(y / n)                   # method-of-moments start
    res = minimize(neg_marginal_loglik, np.log([m**2 / v, m / v]), method="Nelder-Mead")
    alpha, beta = np.exp(res.x)
    prior_mean = alpha / beta
    print(f"Prior: Gamma(alpha={alpha:.2f}, beta={beta:,.0f})")
    print(f"Prior mean = {prior_mean*PER_100K:.2f} per 100k per decade; "
          f"prior sd = {np.sqrt(alpha)/beta*PER_100K:.2f}")
    return alpha, beta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This code visualizes our prior distribution for the death rate.
    """)

    return


@app.cell
def _(PER_100K, alpha, alt, beta, gamma_dist, np, pl):
    _x = np.linspace(0, gamma_dist.ppf(0.995, a=alpha, scale=1 / beta) * PER_100K, 300)
    _prior_density = pl.DataFrame({
        "rate_100k": _x,
        "density": gamma_dist.pdf(_x / PER_100K, a=alpha, scale=1 / beta),
    })

    prior_chart = (
        alt.Chart(_prior_density, width=800, height=250)
        .mark_area(opacity=0.5)
        .encode(
            x=alt.X("rate_100k:Q", title="Kidney cancer deaths per 100k, 1980-89"),
            y=alt.Y("density:Q", title="Prior density"),
        )
    )

    prior_chart

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below, create a table called `kidney_shrunk` that adds:
    1. A column called `SF` which calculates the shrinkage factor for each location.
    ```python
     (pl.col("pop") / (beta + pl.col("pop"))).alias("SF")
    ```
    1. A column called `rate_shrunk` which applies this `shrink` function:
    ```python
    def shrink(mle, sf, pop):
        return sf * mle + (1 - sf) * prior_mean - sf / pop
    ```
    1. A column called `rate_diff` which is the absolute value of `rate_raw` minus `rate_shrunk`
    ```python
    np.abs(pl.col("rate_shrunk") - pl.col("rate_raw")).alias('rate_diff')
    ```

    Look at the `top_k` values for `rate_shrunk`. Can you spot a pattern?
    ```python
    kidney_shrunk.top_k(10, by="rate_diff")
    ```

    <details>
    <summary>Hint</summary>
    ```python
    def shrink(mle, sf, pop):
        return sf * mle + (1 - sf) * prior_mean - sf / pop


    kidney_shrunk = (
        kidney
        .with_columns(
            (pl.col("pop") / (beta + pl.col("pop"))).alias("SF"),  # weight on the data
        )
        .with_columns(
            shrink(pl.col("rate_raw"), pl.col("SF"), pl.col("pop")).alias("rate_shrunk"),
        )
        .with_columns(
            np.abs(pl.col("rate_shrunk") - pl.col("rate_raw")).alias('rate_diff')
        )
    )

    kidney_shrunk.top_k(10, by="rate_diff")
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
    Here is a snippet to create a long-form view of the `kidney_shrunk` table:

    ```python
    _source = (
        kidney_shrunk.select("location", "pop", "deaths", "rate_raw", "rate_shrunk")
        .unpivot(index=["location", "pop", "deaths"],
                 on=["rate_raw", "rate_shrunk"],
                 variable_name="estimate", value_name="rate")
        .with_columns(
            rate_100k=pl.col("rate") * PER_100K,
        )
    )
    ```

    ...and a function for making a plot of the Bayesian-moderated death rate versus population:
    ```python
    def funnel_panel(label):
        pts = (
            alt.Chart(_source.filter(pl.col("estimate") == label))
            .mark_circle(size=8, opacity=0.4)
            .encode(
                x=alt.X("pop:Q", scale=alt.Scale(type="log"),
                        title="County population (log scale)"),
                y=alt.Y("rate_100k:Q", scale=alt.Scale(domain=[0, 60], clamp=True),
                        title="Kidney cancer deaths per 100k, 1980-89"),
                tooltip=["location", alt.Tooltip("pop", format=","), "deaths",
                         alt.Tooltip("rate_100k", format=".1f", title="per 100k")],
            )
        )
        prior_line = (
            alt.Chart(pl.DataFrame({"y": [prior_mean * PER_100K]}))
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        return (pts + prior_line).properties(width=380, height=320, title=label)
    ```

    Use `alt.hconcat` and the `funnel_panel` function to plot moderated and unmoderated death rate side by side.
    """)

    return


@app.cell
def _():
    # Code goes here


    return


if __name__ == "__main__":
    app.run()
