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


@app.cell
def _():
    import statsmodels.formula.api as smf
    import altair as alt
    import polars as pl
    import numpy as np
    import pandas as pd


    return alt, np, pd, pl, smf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Bikeshare data
    """)
    return


@app.cell
def load_bike_data(pl):
    bike_url = (
        "https://raw.githubusercontent.com/UNC-DATA-791/live-session-notebooks"
        "/main/unit2-live-session/data/bikeshare.csv"
    )

    bike = pl.read_csv(bike_url, try_parse_dates=True)

    bike
    return (bike,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Models
    """)
    return


@app.cell
def _(bike, smf):
    bike_formula = (
        "cnt ~ weekday"
    )

    bike_pd = bike.to_pandas()

    bike_poisson = smf.poisson(bike_formula, data=bike_pd).fit(disp=0)
    bike_nb = smf.negativebinomial(bike_formula, data=bike_pd).fit(disp=0)
    return bike_nb, bike_pd, bike_poisson


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize predictions
    """)
    return


@app.cell(hide_code=True)
def model_choice(mo):
    model_choice = mo.ui.radio(
        options=["Poisson", "Negative Binomial"],
        value="Poisson",
        label="Model",
    )
    show_interval = mo.ui.checkbox(value=False, label="Show prediction interval")
    mo.hstack([model_choice, show_interval])
    return model_choice, show_interval


@app.cell(hide_code=True)
def _(alt, bike_nb, bike_pd, bike_poisson, model_choice, np, show_interval):
    _models = {"Poisson": bike_poisson, "Negative Binomial": bike_nb}
    _selected_model = _models[model_choice.value]

    _colors = {"Poisson": "red", "Negative Binomial": "blue"}
    _line_color = _colors[model_choice.value]

    _dist = _selected_model.get_distribution(bike_pd)

    _jitter_width = 0.15

    bike_predictions = bike_pd.assign(
        predicted_cnt=_selected_model.predict(bike_pd),
        pi_lower=_dist.ppf(0.025),
        pi_upper=_dist.ppf(0.975),
        jittered_weekday=bike_pd["weekday"]
        + np.random.uniform(-_jitter_width, _jitter_width, size=len(bike_pd)),
    )

    _base = alt.Chart(bike_predictions, width=600, height=250).encode(
            x=alt.X("weekday", title="Weekday"),
            y=alt.Y("cnt", title="Rentals").scale(domain=[0, 10_500]),
    )

    _actual_points = (
        _base
        .mark_circle(opacity=0.3, color="gray")
        .encode(
            x=alt.X("jittered_weekday", title="Weekday"),
        )
    )

    _predicted_line = (
        _base
        .mark_line(color=_line_color)
        .encode(
            y=alt.Y("predicted_cnt"),
        )
    )

    if show_interval.value:
        _prediction_ribbon = (
            _base
            .mark_area(opacity=0.2, color=_line_color)
            .encode(
                y=alt.Y("pi_lower"),
                y2=alt.Y2("pi_upper"),
            )
        )
        prediction_chart = _actual_points + _prediction_ribbon + _predicted_line
    else:
        prediction_chart = _actual_points + _predicted_line

    prediction_chart

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weekday effect ($\beta_1$) across models
    """)
    return


@app.cell(hide_code=True)
def _(alt, bike_nb, bike_poisson, pd):
    beta1_summary = pd.DataFrame([
        {
            "model": "Poisson",
            "estimate": bike_poisson.params["weekday"],
            "ci_lower": bike_poisson.conf_int().loc["weekday", 0],
            "ci_upper": bike_poisson.conf_int().loc["weekday", 1],
        },
        {
            "model": "Negative Binomial",
            "estimate": bike_nb.params["weekday"],
            "ci_lower": bike_nb.conf_int().loc["weekday", 0],
            "ci_upper": bike_nb.conf_int().loc["weekday", 1],
        },
    ])

    _axis_title = "Weekday coefficient (β₁)"

    _zero_line = alt.Chart().mark_rule(strokeDash=[4, 4], color="gray").encode(
        x=alt.X(datum=0)
    )

    _base = alt.Chart(beta1_summary, width=500, height=150).encode(
        y=alt.Y("model:N", title=None),
        color=alt.Color("model:N", legend=None),
    )
    _points = (
        _base
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X("estimate:Q", title=_axis_title),
        )
    )

    _errorbars = (
        _base
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("ci_lower:Q", title=_axis_title),
            x2=alt.X2("ci_upper:Q"),
        )
    )

    _lower_caps = (
        _base
        .mark_tick(thickness=2, size=10)
        .encode(
            x=alt.X("ci_lower:Q", title=_axis_title),
        )
    )

    _upper_caps = (
        _base
        .mark_tick(thickness=2, size=10)
        .encode(
            x=alt.X("ci_upper:Q", title=_axis_title),
        )
    )

    beta1_chart = _zero_line + _errorbars + _lower_caps + _upper_caps + _points
    beta1_chart

    return


if __name__ == "__main__":
    app.run()
