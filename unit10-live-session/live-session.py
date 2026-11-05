# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.4",
#     "polars==1.42.1",
#     "py3dmol==2.5.5",
#     "pyarrow==25.0.0",
#     "torch==2.11.0",
#     "tmtools==0.3.0",
#     "transformers==5.14.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import altair as alt
    import polars as pl

    return alt, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### Today: predicting structure from sequence
    - Why structure prediction spiked up at CASP14 (AlphaFold2)
    - Fold a protein with ESMFold
    - Read confidence: pLDDT along the chain
    - Compare with a lab structure: TM-score
    - A real case: the malaria vaccine candidate Pfs48/45

    ### What's next?
    - Unit 11: binder design
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Protein structures
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import py3Dmol

    _view = py3Dmol.view(query="pdb:1UBQ", width=500, height=400)
    _view.setStyle({"cartoon": {"color": "spectrum"}})
    _view.zoomTo()
    mo.iframe(_view._make_html(), height=420)
    return (py3Dmol,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AlphaFold 2
    """)
    return


@app.cell
def _(pl):
    casp = pl.read_csv(
        "https://raw.githubusercontent.com/UNC-DATA-791/live-session-notebooks/main/unit10-live-session/casp_median_gdt.csv"
    )
    casp
    return (casp,)


@app.cell(hide_code=True)
def _(alt, casp, pl):
    _base = alt.Chart(casp).encode(
        x=alt.X("year:O", title="CASP year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "median_gdt_ts:Q",
            title="Median GDT_TS, best model per hard target",
            scale=alt.Scale(domain=[0, 100]),
        ),
        tooltip=[
            alt.Tooltip("casp:N", title="Round"),
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("median_gdt_ts:Q", title="Median GDT_TS"),
            alt.Tooltip("n_domains:Q", title="Hard targets"),
        ],
    )
    _bars = _base.mark_bar(cornerRadiusEnd=4, size=60).encode(
        color=alt.condition(
            alt.datum.casp == "CASP14",
            alt.value("#2f6fdb"),
            alt.value("#9aa3ad"),
        )
    )
    _labels = _base.mark_text(dy=-8, fontWeight="bold").encode(
        text=alt.Text("median_gdt_ts:Q", format=".0f")
    )
    _threshold = (
        alt.Chart(pl.DataFrame({"y": [90]}))
        .mark_rule(strokeDash=[6, 4], color="#d9552b", strokeWidth=2)
        .encode(y="y:Q")
    )
    _threshold_label = (
        alt.Chart(
            pl.DataFrame(
                {
                    "y": [90],
                    "text": [
                        "~90: about where lab methods agree with each other"
                    ],
                }
            )
        )
        .mark_text(align="left", dx=4, dy=-8, color="#d9552b", fontSize=12)
        .encode(y="y:Q", text="text:N", x=alt.value(0))
    )
    (_bars + _labels + _threshold + _threshold_label).properties(
        width=680,
        height=220,
        title="Hard-target accuracy jumped at CASP14 (AlphaFold2)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        "https://raw.githubusercontent.com/UNC-DATA-791/live-session-notebooks/main/unit10-live-session/af1_vs_af2_gradient_flow.png?v=2",
        alt="AlphaFold 1 trained only the network up to the distogram. AlphaFold 2 put the loss on the 3D atoms, so learning reached every part.",
        width=900,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ESMFold
    """)
    return


@app.cell
def _():
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    esmfold_model = (
        EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        .eval()
        .to(device)
    )
    device
    return device, esmfold_model, esmfold_tokenizer, torch


@app.cell
def _(device, esmfold_model, esmfold_tokenizer, torch):
    def fold(sequence):
        """Fold one sequence with ESMFold. Returns (pdb_text, per-residue pLDDT on a 0-100 scale)."""
        inputs = esmfold_tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        ).to(device)
        with torch.no_grad():
            output = esmfold_model(**inputs)
        pdb_text = esmfold_model.output_to_pdb(output)[0]
        plddt = output.plddt[0, :, 1].cpu().numpy() * 100  # CA atom, 0-100
        return pdb_text, plddt

    return (fold,)


@app.cell(hide_code=True)
def _(fold, mo, py3Dmol):
    ubiquitin = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ubq_pdb, ubq_plddt = fold(ubiquitin)

    _view = py3Dmol.view(width=500, height=400)
    _view.addModel(ubq_pdb, "pdb")
    _view.setStyle(
        {
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "roygb",
                    "min": 0.5,
                    "max": 1.0,
                }
            }
        }
    )
    _view.zoomTo()
    mo.vstack(
        [
            mo.md(
                f"**ESMFold prediction for ubiquitin.** Mean pLDDT: **{ubq_plddt.mean():.1f}**. Red = unsure, blue = confident."
            ),
            mo.iframe(_view._make_html(), height=420),
        ]
    )
    return ubiquitin, ubq_pdb, ubq_plddt


@app.cell(hide_code=True)
def _(alt, pl, ubiquitin, ubq_plddt):
    ubq_df = pl.DataFrame(
        {
            "position": list(range(1, len(ubiquitin) + 1)),
            "residue": list(ubiquitin),
            "plddt": ubq_plddt.tolist(),
        }
    )

    _line = (
        alt.Chart(ubq_df)
        .mark_line(strokeWidth=2, color="#2f6fdb")
        .encode(
            x=alt.X("position:Q", title="Residue position"),
            y=alt.Y(
                "plddt:Q", title="pLDDT", scale=alt.Scale(domain=[0, 100])
            ),
            tooltip=[
                alt.Tooltip("position:Q", title="Position"),
                alt.Tooltip("residue:N", title="Residue"),
                alt.Tooltip("plddt:Q", title="pLDDT", format=".1f"),
            ],
        )
    )
    _cutoffs = (
        alt.Chart(pl.DataFrame({"y": [50, 70, 90]}))
        .mark_rule(strokeDash=[4, 4], color="#9aa3ad")
        .encode(y="y:Q")
    )
    (_cutoffs + _line).properties(
        width=560, height=280, title="ESMFold confidence along ubiquitin"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is pLDDT?

    pLDDT is the model's own guess of how right each residue is, on a scale from 0 to 100. It stands for *predicted local distance difference test*.

    | pLDDT | How to read it |
    | --- | --- |
    | above 90 | Very high. The local shape is probably right. |
    | 70 to 90 | Confident. The backbone is probably right. |
    | 50 to 70 | Low. Treat with care. |
    | below 50 | Very low. Often a floppy region with no single shape. |

    **How the model learns it.** pLDDT is learned during structure training on known structures: the model makes a structure and measures how close each residue is to the truth. A small confidence head learns to guess that accuracy.

    **What it is not.**
    - It is the model grading itself. It is not proof.
    - Low does not always mean wrong. A floppy region has no single shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing actual versus predicted
    """)
    return


@app.cell
def _(ubiquitin, ubq_pdb):
    import urllib.request

    import numpy as np
    from tmtools import tm_align

    def ca_coords(pdb_text, chain="A"):
        """CA coordinates of one chain, as an (n, 3) array."""
        return np.array(
            [
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                for line in pdb_text.splitlines()
                if line.startswith("ATOM")
                and line[12:16].strip() == "CA"
                and line[21] == chain
            ]
        )

    def move_pdb(pdb_text, rotation, translation):
        """Apply a rotation and shift to every atom of a PDB string."""
        lines = []
        for line in pdb_text.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                xyz = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
                x, y, z = rotation @ xyz + translation
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            lines.append(line)
        return "\n".join(lines)

    ubq_ref_pdb = (
        urllib.request.urlopen(
            "https://files.rcsb.org/download/1UBQ.pdb", timeout=30
        )
        .read()
        .decode()
    )
    ubq_tm = tm_align(
        ca_coords(ubq_pdb), ca_coords(ubq_ref_pdb), ubiquitin, ubiquitin
    )
    ubq_tm.tm_norm_chain2, ubq_tm.rmsd
    return move_pdb, ubq_ref_pdb, ubq_tm


@app.cell(hide_code=True)
def _(mo, move_pdb, py3Dmol, ubq_pdb, ubq_ref_pdb, ubq_tm):
    _view = py3Dmol.view(width=500, height=400)
    _view.addModel(ubq_ref_pdb, "pdb")
    _view.setStyle({"model": 0}, {"cartoon": {"color": "#9aa3ad"}})
    _view.addModel(move_pdb(ubq_pdb, ubq_tm.u, ubq_tm.t), "pdb")
    _view.setStyle({"model": 1}, {"cartoon": {"color": "#2f6fdb"}})
    _view.zoomTo()
    mo.vstack(
        [
            mo.md(
                f"**ESMFold vs. the real structure (1UBQ).** "
                f"TM-score: **{ubq_tm.tm_norm_chain2:.2f}**. RMSD: **{ubq_tm.rmsd:.2f} Å**. "
                "Gray = real, blue = predicted."
            ),
            mo.iframe(_view._make_html(), height=420),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A real case: the malaria vaccine candidate Pfs48/45

    Matthew Higgins's lab at Oxford struggled for years to solve the structure of Pfs48/45, a malaria surface protein. Lab methods gave only blurry pictures of part of it. An AlphaFold prediction helped them fill the gaps.

    - Story: [Stopping malaria in its tracks](https://deepmind.google/blog/stopping-malaria-in-its-tracks/) (Google DeepMind)
    - Paper: [Structure of the malaria vaccine candidate Pfs48/45 and its recognition by transmission blocking antibodies](https://www.nature.com/articles/s41467-022-33379-6) (Ko et al., *Nature Communications*, 2022; [preprint](https://www.biorxiv.org/content/10.1101/2022.05.24.493318))
    """)
    return


if __name__ == "__main__":
    app.run()
