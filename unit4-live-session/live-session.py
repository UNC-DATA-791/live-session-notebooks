# /// script
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.3",
#     "polars==1.44.2",
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
    # Unit 4 live session
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Review
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Count data is discrete → Poisson regression
    2. Count data is overdispersed → Negative Binomial regression
    3. Sample size is very small → Empirical Bayes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Empirical Bayes
    """)
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    import statsmodels.formula.api as smf
    import numpy as np

    curve_colors = {
        'Prior (all ratings)': 'coral',
        'Likelihood (MLE)': 'steelblue',
        'Posterior (MAP)': 'seagreen',
    }
    return alt, curve_colors, np, pl


@app.cell
def _(pl):
    episodes = pl.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-11-28/drwho_episodes.csv")
    writers = pl.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-11-28/drwho_writers.csv")

    dr_who = episodes.join(writers, on='story_number')
    dr_who
    return dr_who, writers


@app.cell
def _(writers):
    writers_l = writers.get_column('writer').unique().to_list()
    writers_l
    return (writers_l,)


@app.cell(hide_code=True)
def _(alt, pl, writers):
    _source = (
        writers.group_by('writer')
        .agg(pl.len().alias('count'))
        .with_columns(
            pl.when(pl.col('count') == 1)
            .then(pl.lit('n=1'))
            .otherwise(pl.col('writer'))
            .alias('writer_group')
        )
        .group_by('writer_group')
        .agg(pl.col('count').sum())
        .rename({'writer_group': 'writer'})
    )

    _y_order = (
        _source.filter(pl.col('writer') != 'n=1')
        .sort('count', descending=True)
        .get_column('writer')
        .to_list()
        + ['n=1']
    )

    alt.Chart(_source).mark_bar().encode(
        y=alt.Y('writer', sort=_y_order),
        x=alt.X('count')
    )
    return


@app.cell(hide_code=True)
def _(alt, pl, writers):
    _source = (
        writers
        .group_by('writer')
        .agg(pl.len().alias('n'))
    )

    alt.Chart(_source, width=600, height=100).mark_bar().encode(
        x=alt.X('n').bin(step=2).title('N episodes'),
        y=alt.Y('count()').title('Frequency')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *What rating would we predict for a brand new writer?*
    """)
    return


@app.cell(hide_code=True)
def _(alt, curve_colors, dr_who, np, pl):
    _mean = dr_who['rating'].mean()
    _std = dr_who['rating'].std()

    _x = np.linspace(dr_who['rating'].min() - 3, dr_who['rating'].max() + 4, 200)
    _pdf = (1 / (_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((_x - _mean) / _std) ** 2)
    _normal_curve = pl.DataFrame({'rating': _x, 'density': _pdf}).with_columns(
        pl.lit('Prior (all ratings)').alias('series')
    )
    _domain = [dr_who.get_column('rating').min(), dr_who.get_column('rating').max()]

    _hist = alt.Chart(dr_who, width=600, height=200).mark_bar().encode(
        x=alt.X('rating').bin(maxbins=30),
        y=alt.Y('count()')
    )

    ratings_normal_curve = alt.Chart(_normal_curve).mark_line(strokeWidth=3).encode(
        x=alt.X('rating').scale(domain=_domain),
        y=alt.Y('density'),
        color=alt.Color(
            'series:N',
            scale=alt.Scale(domain=list(curve_colors.keys()), range=list(curve_colors.values())),
            legend=alt.Legend(title='Distribution'),
        ),
    )

    alt.layer(_hist, ratings_normal_curve).resolve_scale(y='independent')
    return (ratings_normal_curve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *What rating would we expect for an experienced writer? That is, what is the true mean of episode ratings that produced the data we have observed?*
    """)
    return


@app.cell(hide_code=True)
def _(mo, writers_l):
    mle_writer_dropdown = mo.ui.dropdown(
        options=sorted(writers_l),
        value='Russell T Davies',
        label='Pick a writer',
    )
    mle_writer_dropdown
    return (mle_writer_dropdown,)


@app.cell(hide_code=True)
def _(alt, curve_colors, dr_who, mle_writer_dropdown, np, pl):
    _writer_episodes = dr_who.filter(pl.col('writer') == mle_writer_dropdown.value)

    selected_writer_n = _writer_episodes.height
    selected_writer_mle = _writer_episodes.get_column('rating').mean()

    # Same known sigma used everywhere else in the notebook, so this curve is
    # directly comparable to the prior curve: same spread, just recentered at
    # this writer's own MLE.
    _sigma = dr_who['rating'].std()
    _domain = [70, 98]
    _x = np.linspace(_domain[0], _domain[1], 400)
    _pdf = (1 / (_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((_x - selected_writer_mle) / _sigma) ** 2)

    _density_curve = alt.Chart(pl.DataFrame({'rating': _x, 'density': _pdf})).mark_line(
        color=curve_colors['Likelihood (MLE)'], strokeWidth=3
    ).encode(
        x=alt.X('rating', scale=alt.Scale(domain=_domain)).title(None),
        y=alt.Y('density').title('Density'),
    )

    _mle_rule = alt.Chart(pl.DataFrame({'rating': [selected_writer_mle]})).mark_rule(
        color=curve_colors['Likelihood (MLE)'], strokeDash=[4, 4], size=2
    ).encode(
        x=alt.X('rating', scale=alt.Scale(domain=_domain))
    )

    _mle_label = alt.Chart(pl.DataFrame({'rating': [selected_writer_mle]})).mark_text(
        align='left', dx=5, fontSize=12, color=curve_colors['Likelihood (MLE)'],
        text=f'MLE = {selected_writer_mle:.2f}  (n = {selected_writer_n})'
    ).encode(
        x=alt.X('rating', scale=alt.Scale(domain=_domain)),
        y=alt.value(10),
    )

    # Jitter the rug's x-position a bit so overlapping ratings (a common score
    # like 84 shared by several episodes) don't stack on top of each other.
    # Seeded so the jitter doesn't reshuffle on every rerun.
    _rng = np.random.default_rng(0)
    _writer_episodes_jittered = _writer_episodes.with_columns(
        (pl.col('rating') + _rng.uniform(-0.15, 0.15, _writer_episodes.height)).alias('rating_jittered')
    )

    # Rug: one tick per observed episode rating, directly under the curve.
    _rug = alt.Chart(_writer_episodes_jittered).mark_tick(
        color=curve_colors['Likelihood (MLE)'], thickness=2, size=20
    ).encode(
        x=alt.X('rating_jittered', scale=alt.Scale(domain=_domain)).title('Episode rating'),
        tooltip=[alt.Tooltip('episode_title'), alt.Tooltip('rating')],
    )

    alt.vconcat(
        alt.layer(_density_curve, _mle_rule, _mle_label).properties(width=600, height=150),
        _rug.properties(width=600, height=40),
    ).resolve_scale(x='shared')
    return selected_writer_mle, selected_writer_n


@app.cell(hide_code=True)
def _(mo, selected_writer_mle, selected_writer_n):
    n_eps_slider = mo.ui.slider(
        start=1, stop=46, value=selected_writer_n, step=0.5, label='N episodes'
    )
    obs_ratings_mean_slider = mo.ui.slider(
        start=75, stop=91, value=round(selected_writer_mle, 1), step=0.1, label='Observed ratings mean'
    )

    mo.vstack([n_eps_slider, obs_ratings_mean_slider])
    return n_eps_slider, obs_ratings_mean_slider


@app.cell(hide_code=True)
def _(
    alt,
    curve_colors,
    dr_who,
    n_eps_slider,
    np,
    obs_ratings_mean_slider,
    pl,
    ratings_normal_curve,
):
    _sigma = dr_who['rating'].std()
    _n = n_eps_slider.value
    _obs_mean = obs_ratings_mean_slider.value
    _se = _sigma / np.sqrt(_n)

    # Narrower fixed domain so the MLE (likelihood peak) and MAP (posterior peak)
    # stay visually distinguishable instead of getting lost in a wide axis.
    _domain = [70, 98]
    _x = np.linspace(_domain[0], _domain[1], 400)

    _likelihood = (1 / (_se * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((_x - _obs_mean) / _se) ** 2)
    _likelihood_data = pl.DataFrame({'mu': _x, 'density': _likelihood}).with_columns(
        pl.lit('Likelihood (MLE)').alias('series')
    )
    _likelihood_curve = alt.Chart(_likelihood_data, width=600, height=200).mark_line(strokeWidth=3).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain)).title('μ (possible true mean rating)'),
        y=alt.Y('density').title('Density'),
        color=alt.Color(
            'series:N',
            scale=alt.Scale(domain=list(curve_colors.keys()), range=list(curve_colors.values())),
            legend=alt.Legend(title='Distribution'),
        ),
    )

    # Posterior: normal-normal conjugate update. The full-data normal curve
    # (ratings_normal_curve) stands in as the prior, worth one pseudo-observation
    # at the same known sigma as the likelihood.
    _prior_mean = dr_who['rating'].mean()
    _post_n = _n + 1
    _post_mean = (_prior_mean + _n * _obs_mean) / _post_n
    _post_se = _sigma / np.sqrt(_post_n)

    _posterior = (1 / (_post_se * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((_x - _post_mean) / _post_se) ** 2)
    _posterior_data = pl.DataFrame({'mu': _x, 'density': _posterior}).with_columns(
        pl.lit('Posterior (MAP)').alias('series')
    )
    _posterior_curve = alt.Chart(_posterior_data).mark_line(strokeWidth=3).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain)),
        y=alt.Y('density'),
        color=alt.Color(
            'series:N',
            scale=alt.Scale(domain=list(curve_colors.keys()), range=list(curve_colors.values())),
            legend=alt.Legend(title='Distribution'),
        ),
    )

    # Annotate the three key points: the empirical mean (prior), the MLE
    # (likelihood peak), and the MAP (posterior peak).
    _mean_rule = alt.Chart(pl.DataFrame({'mu': [_prior_mean]})).mark_rule(
        color=curve_colors['Prior (all ratings)'], strokeDash=[4, 4]
    ).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain))
    )
    _mean_label = alt.Chart(pl.DataFrame({'mu': [_prior_mean]})).mark_text(
        align='left', dx=5, fontSize=12, color=curve_colors['Prior (all ratings)'],
        text=f'Mean = {_prior_mean:.2f}'
    ).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain)),
        y=alt.value(10)
    )

    _mle_rule = alt.Chart(pl.DataFrame({'mu': [_obs_mean]})).mark_rule(
        color=curve_colors['Likelihood (MLE)'], strokeDash=[4, 4]
    ).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain))
    )
    _mle_label = alt.Chart(pl.DataFrame({'mu': [_obs_mean]})).mark_text(
        align='left', dx=5, fontSize=12, color=curve_colors['Likelihood (MLE)'],
        text=f'MLE = {_obs_mean:.2f}'
    ).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain)),
        y=alt.value(25)
    )

    _map_rule = alt.Chart(pl.DataFrame({'mu': [_post_mean]})).mark_rule(
        color=curve_colors['Posterior (MAP)'], strokeDash=[4, 4]
    ).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain))
    )
    _map_label = alt.Chart(pl.DataFrame({'mu': [_post_mean]})).mark_text(
        align='left', dx=5, fontSize=12, fontWeight='bold', color=curve_colors['Posterior (MAP)'],
        text=f'MAP = {_post_mean:.2f}'
    ).encode(
        x=alt.X('mu', scale=alt.Scale(domain=_domain)),
        y=alt.value(40)
    )

    alt.layer(
        _likelihood_curve, ratings_normal_curve, _posterior_curve,
        _mean_rule, _mean_label, _mle_rule, _mle_label, _map_rule, _map_label,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    🤔 **Why is the MLE estimate just the sample mean?**
    """)
    return


@app.cell(hide_code=True)
def shrinkage(mo):
    mo.md(r"""
    ### Shrinkage factor to get MAP

    The shrinkage factor $SF$ says how much weight the MAP estimate gives to the prior mean versus the observed data (the MLE). Here the prior counts as one pseudo-observation at the same known variance as the likelihood, so:

    $$
    SF = \frac{1}{n + 1}
    $$

    where $n$ is the number of observed episodes.

    The MAP is then a weighted average of the prior mean and the MLE, using the shrinkage factor as the weight on the prior:

    $$
    \text{MAP} = SF \cdot \mu_{\text{prior}} + (1 - SF) \cdot \text{MLE}
    $$

    As $n \to \infty$, $SF \to 0$ and the MAP converges to the MLE. As $n \to 0$, $SF \to 1$ and the MAP shrinks toward the prior mean.

    **Exercise:** write a function `_shrink` that takes the MLE estimate, n and prior mean as input returning the MAP estimate. Use `_shrink` to compute each writer's `MAP_rating` from their `MLE_rating` and episode count `n`, then plot `MLE_rating` vs `MAP_rating` with a 1:1 line.

    <details>
    <summary>Solution</summary>

    ```python
    def _shrink(mle, n, prior_mean):
        sf = 1 / (n + 1)
        return (1 - sf) * mle + sf * prior_mean


    _prior_mean = dr_who['rating'].mean()

    _source = (
        dr_who
        .group_by('writer')
        .agg(
            pl.col('rating').mean().alias('MLE_rating'),
            pl.len().alias('n'),
        )
        .with_columns(
            _shrink(pl.col('MLE_rating'), pl.col('n'), _prior_mean).alias('MAP_rating')
        )
    )

    _axis_min = min(_source.get_column('MLE_rating').min(), _source.get_column('MAP_rating').min())
    _axis_max = max(_source.get_column('MLE_rating').max(), _source.get_column('MAP_rating').max())

    _identity_line = alt.Chart(
        pl.DataFrame({'MLE_rating': [_axis_min, _axis_max], 'MAP_rating': [_axis_min, _axis_max]})
    ).mark_line(color='gray', strokeDash=[4, 4]).encode(
        x='MLE_rating',
        y='MAP_rating',
    )

    _scatter = alt.Chart(_source).mark_point(fill='steelblue', fillOpacity=0.5).encode(
        x=alt.X('MLE_rating').scale(zero=False),
        y=alt.Y('MAP_rating').scale(zero=False),
        size=alt.Size('n').scale(range=[35, 500]),
        tooltip=[alt.Tooltip('writer'), alt.Tooltip('n'), alt.Tooltip('MLE_rating'), alt.Tooltip('MAP_rating')]
    )

    alt.layer(_identity_line, _scatter)
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
    ## Example from DESeq2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://cdn.ncbi.nlm.nih.gov/pmc/blobs/98ea/4302049/5b838bf6a2c3/13059_2014_550_Fig2_HTML.jpg" target="_blank" rel="noopener noreferrer" style="text-decoration: none; cursor: zoom-in;">
    <img src="https://cdn.ncbi.nlm.nih.gov/pmc/blobs/98ea/4302049/5b838bf6a2c3/13059_2014_550_Fig2_HTML.jpg" style="max-width: 100%;" alt="Figure 2 from Love, Huber & Anders (2014), showing MLE vs MAP shrinkage of fold-change estimates" />
    </a>

    [Open full size in a new tab ↗](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/98ea/4302049/5b838bf6a2c3/13059_2014_550_Fig2_HTML.jpg)

    *Figure 2 from Love MI, Huber W, Anders S. "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2." Genome Biology. 2014;15:550. [https://doi.org/10.1186/s13059-014-0550-8](https://doi.org/10.1186/s13059-014-0550-8)*
    """)
    return


if __name__ == "__main__":
    app.run()
