# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "polars==1.43.2",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polars and Altair practice

    Below we load COVID-19 time series data from Johns Hopkins.
    """)
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    import marimo as mo

    url = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
           "csse_covid_19_data/csse_covid_19_time_series/"
           "time_series_covid19_confirmed_global.csv")

    df = pl.read_csv(url)

    df
    return alt, df, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data is in a "wide" format not suitable for downstream EDA. Use `unpivot` to convert the table into a "long" format where each row represents a date/country combination and we have one column with the case count for that date/country.
    """)
    return


@app.cell
def _(df, pl):
    # TODO: unpivot df into long_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can do some EDA!

    1. Get a table of max cases by country.
    2. Plot a time series of cases by date for a specific country. (If you're ambitious connect this chart to a [dropdown selector](https://docs.marimo.io/api/inputs/dropdown/) for country.)
    """)
    return


@app.cell
def _(long_df, pl):
    # TODO: table of max cases by country
    return


@app.cell
def _(alt, long_df, pl):
    # TODO: time series chart of cases by date for a specific country
    return


if __name__ == "__main__":
    app.run()
