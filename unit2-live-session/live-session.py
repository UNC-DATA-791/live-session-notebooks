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


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import html
    import altair as alt
    import numpy as np
    import polars as pl
    import pandas as pd
    import statsmodels.formula.api as smf


    return alt, np, pl, smf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RNA-Seq overview
    """)
    return


@app.cell(hide_code=True)
def _():
    # RNA-seq video timestamps (StatQuest: "A gentle introduction to RNA-seq")
    rnaseq_video_id = "tlf6wYJrwKY"
    rnaseq_video_url = f"https://www.youtube.com/watch?v={rnaseq_video_id}"

    # (seconds, section, topic)
    rnaseq_marks = [
        (23, "Intro", "Welcome + the question: normal vs mutated neural cells"),
        (57, "Intro", "Gene expression basics: chromosomes, genes, mRNA transcripts"),
        (74, "Intro", "What high-throughput sequencing tells you (active genes + how much)"),
        (108, "Intro", "Comparing two cell types: no / big / subtle differences"),
        (124, "Overview", "The 3 main steps of RNA-seq"),
        (139, "1. Library prep", "Preparing a library (Illumina protocol as the example)"),
        (154, "1. Library prep", "Steps 1-2: isolate RNA, break it into ~200-300 bp fragments"),
        (171, "1. Library prep", "Step 3: convert RNA fragments to double-stranded DNA"),
        (186, "1. Library prep", "Step 4: add sequencing adapters (and why: multiplexing)"),
        (232, "1. Library prep", "Step 5: PCR amplify - only adapter-tagged fragments enrich"),
        (247, "1. Library prep", "Step 6: QC - library concentration and fragment length"),
        (250, "2. Sequencing", "Sequencing the library: the flow cell (~400M fragments)"),
        (280, "2. Sequencing", "Fluorescent probes, imaging, and the base-calling cycle"),
        (348, "2. Sequencing", "Quality scores: faded spots and low-confidence base calls"),
        (381, "2. Sequencing", "Low diversity: why the first few cycles matter"),
        (411, "2. Sequencing", "Raw data: FASTQ - 4 lines per read, 1.6 billion lines per run"),
        (463, "3. Data analysis", "Roadmap: filter, align, count"),
        (479, "3. Data analysis", "Filtering garbage reads: low quality + adapter dimers"),
        (510, "3. Data analysis", "Aligning reads: build a genome index of fragments"),
        (542, "3. Data analysis", "Matching read fragments to genome fragments -> position"),
        (557, "3. Data analysis", "Why fragment? Tolerate mismatches and genetic variation"),
        (587, "3. Data analysis", "Counting reads per gene using gene coordinates"),
        (605, "3. Data analysis", "The count matrix: ~20,000 genes x samples"),
        (639, "3. Data analysis", "Bulk RNA-seq vs single-cell RNA-seq"),
        (689, "3. Data analysis", "Why normalize: samples get different read totals"),
        (720, "3. Data analysis", "Normalization example: 635 vs 1270 reads"),
        (771, "3. Data analysis", "Simplest normalization: divide by total mapped reads"),
        (789, "3. Data analysis", "Recap of the whole pipeline"),
        (807, "4. Plot the data", "Step 1 of any analysis: always plot the data"),
        (824, "4. Plot the data", "Why 20,000 genes needs 20,000 axes"),
        (859, "4. Plot the data", "PCA reduces the number of axes you need"),
        (874, "4. Plot the data", "Reading a real PCA plot: WT vs KO clusters"),
        (936, "4. Plot the data", "Single-cell PCA plot: colouring cells by behaviour"),
        (968, "4. Plot the data", "Two reasons to plot: expect differences, exclude samples"),
        (984, "5. Diff. expression", "Step 2: differentially expressed genes (edgeR / DESeq2)"),
        (1000, "5. Diff. expression", "Reading the plot: CPM on x, log fold change on y"),
        (1046, "5. Diff. expression", "So what now? Validate a hypothesis or test pathway enrichment"),
    ]


    def rnaseq_stamp(seconds: int) -> str:
        """Format seconds as mm:ss."""
        return f"{seconds // 60:d}:{seconds % 60:02d}"


    rnaseq_rows = [
        {
            "time": rnaseq_stamp(s),
            "section": section,
            "topic": topic,
            "seconds": s,
            "url": f"{rnaseq_video_url}&t={s}s",
        }
        for s, section, topic in rnaseq_marks
    ]

    return (rnaseq_rows,)


@app.cell
def _():
    rnaseq_picks = [124, 154, 171, 411, 463, 587, 605]

    return (rnaseq_picks,)


@app.cell(hide_code=True)
def _(mo, rnaseq_picks, rnaseq_rows):
    # My RNA-seq timestamps. Each timestamp links to YouTube at that point.
    #
    # Built with mo.md, not mo.Html: marimo sanitises raw HTML, which strips the
    # anchors. Markdown links come out as real <a target="_blank"> tags.
    _rows = "\n".join(
        f"| [`{r['time']}`]({r['url']}) | {r['section']} | {r['topic']} |"
        for r in rnaseq_rows
        if r["seconds"] in rnaseq_picks
    )

    mo.md(f"| Time | Section | Topic |\n| --- | --- | --- |\n{_rows}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling count data

    RNA-Seq produces "count" data. It's discrete and it is `heteroscedastic`. That is, the variance over the range of the data is dynamic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's load the `Bikeshare` dataset.
    """)
    return


@app.cell
def _(pl):
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
    ### Variance versus mean `cnt`

    We want to model the number of rentals per day using some of the features. But let's take a look at if the rental count variance is constant over the rental count range.
    """)
    return


@app.cell
def _(alt, bike):
    alt.Chart(bike, width=600, height=100).mark_bar().encode(
        x=alt.X('cnt').bin(maxbins=50),
        y=alt.Y('count()')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's group our data based on days that we think might be similar.
    """)
    return


@app.cell
def _(bike, pl):
    bike_groups = (
        bike
        .group_by("yr", "mnth", "workingday")
        .agg(
            pl.len().alias("n"),
            pl.col("cnt").mean().alias("mean"),
            pl.col("cnt").var().alias("variance"),
        )
        .filter(pl.col("n") >= 8)
        .with_columns(
            (pl.col("variance") / pl.col("mean")).round(1).alias("var / mean"),
            pl.lit("Day groups (replicates)").alias("source"),
        )
        .sort("mean")
    )

    bike_groups
    return (bike_groups,)


@app.cell
def _(alt, bike_groups, np, pl):
    _grid = np.linspace(bike_groups["mean"].min(), bike_groups["mean"].max(), 100)
    unit_line = pl.DataFrame({"mean": _grid, "variance": _grid})

    _x = alt.X("mean:Q", scale=alt.Scale(type="log"), title="Mean rides in the group")
    _y = alt.Y("variance:Q", scale=alt.Scale(type="log"), title="Variance of rides (log scale)")

    points = alt.Chart(bike_groups, width = 600).mark_point(size=80).encode(_x, _y)

    line = alt.Chart(unit_line).mark_line(stroke='coral', strokeWidth=4).encode(_x, _y)

    points + line
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, variance does seem to scale with the group mean. Also, variance is greater than the mean.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Let's make some models!
    """)
    return


@app.cell
def _(bike, pd, smf):
    bike_formula = (
        "cnt ~ temp + hum + windspeed + yr + workingday + C(season) + C(weathersit)"
    )

    bike_pd = pd.DataFrame(bike.to_dicts())

    bike_poisson = smf.poisson(bike_formula, data=bike_pd).fit(disp=0)

    return bike_formula, bike_pd, bike_poisson


@app.cell
def _(bike_formula, bike_pd, smf):
    bike_nb = smf.negativebinomial(bike_formula, data=bike_pd).fit(disp=0)

    return (bike_nb,)


@app.cell
def _(bike_nb, bike_pd, bike_poisson, np):
    bike_mu_p = bike_poisson.predict(bike_pd).to_numpy()
    bike_mu_n = bike_nb.predict(bike_pd).to_numpy()

    predictions = {"negative binomial": bike_mu_n, "poisson": bike_mu_p}

    bike_alpha = float(bike_nb.params["alpha"])
    rng = np.random.default_rng(0)
    size = 1.0 / bike_alpha  # NB2 -> numpy's (n, p) parameterisation

    bike_series = ["actual", "poisson", "neg_binomial"]

    bike_colors = ["#2a78d6", "#eb6834", "#1baf7a"]

    return (
        bike_colors,
        bike_mu_n,
        bike_mu_p,
        bike_series,
        predictions,
        rng,
        size,
    )


@app.cell
def _(mo, predictions):
    # "actual" residuals need a baseline to subtract. Pick whose fitted means to
    # use. The two simulated series always use their own model's means.
    bike_baseline = mo.ui.dropdown(
        options=list(predictions),
        value="negative binomial",
        label="Center actual residuals on",
    )

    bike_baseline

    return (bike_baseline,)


@app.cell(hide_code=True)
def _(
    alt,
    bike,
    bike_baseline,
    bike_colors,
    bike_mu_n,
    bike_mu_p,
    bike_series,
    pl,
    predictions,
    rng,
    size,
):
    _base = bike_baseline.value or "negative binomial"

    bike_err_df = (
        pl.DataFrame(
            {
                "actual": bike["cnt"].to_numpy() - predictions[_base],
                "poisson": rng.poisson(bike_mu_p) - bike_mu_p,
                "neg_binomial": rng.negative_binomial(size, size / (size + bike_mu_n))
                - bike_mu_n,
            }
        )
        .unpivot(variable_name="source", value_name="residual")
        .sort("residual")
        .with_columns(
            cdf=pl.int_range(1, pl.len() + 1).over("source") / pl.len().over("source")
        )
    )

    alt.Chart(bike_err_df).mark_line(strokeWidth=2).encode(
        x=alt.X("residual:Q", title="Deviation from fitted mean (rides)"),
        y=alt.Y("cdf:Q", title="Cumulative probability", axis=alt.Axis(format="%")),
        color=alt.Color(
            "source:N",
            title=None,
            scale=alt.Scale(domain=bike_series, range=bike_colors),
            legend=alt.Legend(orient="top-left"),
        ),
        tooltip=[
            alt.Tooltip("source:N", title="Series"),
            alt.Tooltip("residual:Q", title="Deviation", format=",.0f"),
            alt.Tooltip("cdf:Q", title="CDF", format=".1%"),
        ],
    ).properties(
        width=620,
        height=340,
        title=f"Actual residuals centered on the {_base} fit",
    ).configure_axis(
        grid=True, gridOpacity=0.25, domainOpacity=0.4, tickOpacity=0.4
    ).configure_view(strokeWidth=0)

    return


if __name__ == "__main__":
    app.run()
