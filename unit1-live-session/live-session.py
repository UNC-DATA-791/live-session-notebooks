# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/live-session.slides.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DATA 791 live session Unit 1

    Observing biology with DNA sequencing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Often we cannot see biology with the naked eye so we use DNA sequence as a means to observe it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Collect biological samples from an experiment.
    2. Extract DNA/RNA
    3. Sequence typically short stretches of DNA/RNA sequence called reads
    4. Reads can be categorized into biological units (e.g. genes, microoganisms)
    5. And read counts within units can be traced across experimental conditions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Bottomly et al. 2011 **Plos One**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017820)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import gzip
    import urllib.request

    fastq_url = "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR099/SRR099226/SRR099226.fastq.gz"
    _n_reads = 3

    with urllib.request.urlopen(fastq_url) as _resp:
        with gzip.GzipFile(fileobj=_resp) as _gz:
            _lines = []
            for _i in range(_n_reads * 4):
                _line = _gz.readline()
                if not _line:
                    break
                _lines.append(_line.decode().rstrip())

    fastq_preview = "\n".join(_lines)

    mo.md(f"""
    ### A real FASTQ file from the Bottomly et al. 2011 dataset

    Run **SRR099226** (mouse striatum RNA-Seq), streamed directly from ENA — only the
    first {_n_reads} reads are pulled, not the whole file.

    `{fastq_url}`

    ```
    {fastq_preview}
    ```
    """)

    return


if __name__ == "__main__":
    app.run()
