# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "polars==1.44.2",
#     "pyarrow==25.0.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### What have we learned so far?
    - There are regression techniques which account for the specific nature of count data
        - discrete
        - variance is not constant over the domain of the predictor(s)
    - We can use bayesian shrinkage to moderate noisy data
        - Useful for low-N data like in biological experiments
    - DNA sequencing based readouts of biological phenomena get synthesized to count matrices

    ### Assessing protein activity
    - See below 👇

    ### What's next?
    - Second module project: building a model of protein activity to optimize the protein sequence for some application.
    - [Protein Gym](https://proteingym.org/)

    ### Questions
    - Polars?
    - Statsmodels?
    - Assignments?
    - How to AI?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assessing protein activity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://cdn.ncbi.nlm.nih.gov/pmc/blobs/ff8f/11127319/40b6f69fd742/13059_2024_3279_Fig1_HTML.jpg" target="_blank" rel="noopener noreferrer" style="text-decoration: none; cursor: zoom-in;">
    <img src="https://cdn.ncbi.nlm.nih.gov/pmc/blobs/ff8f/11127319/40b6f69fd742/13059_2024_3279_Fig1_HTML.jpg" style="max-width: 100%;" alt="Figure 1 from Rao et al. (2024): deep mutational scanning overview. A: each amino acid is mutated. B: variant pool grown under selection and sequenced at time points to make a count table. C: Rosace takes the count table and outputs posterior functional scores." />
    </a>

    [Open full size in a new tab ↗](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/ff8f/11127319/40b6f69fd742/13059_2024_3279_Fig1_HTML.jpg)

    *Figure 1 from Rao J, Xin R, Macdonald C, et al. "Rosace: a robust deep mutational scanning analysis framework employing position and mean-variance shrinkage." Genome Biology. 2024;25:138. [https://doi.org/10.1186/s13059-024-03279-7](https://doi.org/10.1186/s13059-024-03279-7)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://cdn.ncbi.nlm.nih.gov/pmc/blobs/ff8f/11127319/3df3b67e6a73/13059_2024_3279_Fig2_HTML.jpg" target="_blank" rel="noopener noreferrer" style="text-decoration: none; cursor: zoom-in;">
    <img src="https://cdn.ncbi.nlm.nih.gov/pmc/blobs/ff8f/11127319/3df3b67e6a73/13059_2024_3279_Fig2_HTML.jpg" style="max-width: 100%;" alt="Figure 2 from Rao et al. (2024): Rosace shares information at the same position. A: smoothed position-specific scores across OCT1 positions. B: conceptual view of the generative model, where each position has an overall effect from which variant effects come. C: plate model of Rosace." />
    </a>

    [Open full size in a new tab ↗](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/ff8f/11127319/3df3b67e6a73/13059_2024_3279_Fig2_HTML.jpg)

    *Figure 2 from Rao J, Xin R, Macdonald C, et al. "Rosace: a robust deep mutational scanning analysis framework employing position and mean-variance shrinkage." Genome Biology. 2024;25:138. [https://doi.org/10.1186/s13059-024-03279-7](https://doi.org/10.1186/s13059-024-03279-7)*
    """)
    return


if __name__ == "__main__":
    app.run()
