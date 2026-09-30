# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "polars==1.44.2",
#     "pyarrow==25.0.1",
#     "torch==2.14.0",
#     "transformers==5.17.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import torch
    import polars as pl
    import altair as alt
    from transformers import AutoTokenizer, EsmForMaskedLM

    return AutoTokenizer, EsmForMaskedLM, alt, pl, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### What have we learned so far?
    - Unit 5: protein language models (PLMs) like ESM2
        - Mask fill: the model guesses a hidden amino acid from its context
        - Amino acid embeddings line up with chemistry (hydropathy, size, charge)
        - Average the residue vectors → one embedding for the whole sequence

    ### Variant effect prediction
    - One amino acid swap can break a protein, or make it better
    - Two ways to ask the question:
        - **Clinical**: is a patient's variant (e.g. ACVR1 R206H) pathogenic or benign?
        - **Protein engineering**: does A30V help or hurt the protein's function?
    - ClinPred walkthrough
    - Zero-shot scoring with a PLM, and how [ProteinGym](https://proteingym.org/) compares models
    - This week: pick a supervised clinical model from ProteinGym

    ### What's next?
    - Project 2 – Notebook 2: regression from protein embeddings to fitness
    - Unit 7: functional annotation. What does a protein *do*? ([CAFA 6](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction))

    ### Open questions
    - A model says "this variant looks unnatural." Is unnatural the same as harmful?
    - When should we trust a zero-shot score, and when do we need labels?
    - Clinical labels come from ClinVar. What biases come along with them?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two questions, one mutation

    | | Clinical | Protein engineering |
    | --- | --- | --- |
    | Example | ACVR1 **R206H** (causes fibrodysplasia ossificans progressiva) | **A30V** in an enzyme we want to improve |
    | Label | Pathogenic / Benign (ClinVar) | A number: fitness from a deep mutational scan (DMS) |
    | Task | Classification | Regression / ranking |
    | Metric | AUC | Spearman correlation |
    | Cost of a mistake | A wrong diagnosis | A wasted round in the lab |

    Mutation shorthand: **R206H** means the wild-type Arginine (R) at position 206 changes to Histidine (H).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ClinPred walkthrough

    ClinPred ([Alirezaie et al., 2018, *AJHG*](https://doi.org/10.1016/j.ajhg.2018.08.005)) is a **supervised** clinical predictor.

    - **Training labels**: missense variants from ClinVar, pathogenic vs. benign
    - **Features**: scores from older tools (SIFT, PolyPhen-2, CADD, conservation scores, ...) plus **population allele frequency** (gnomAD)
    - **Model**: a random forest and a gradient boosted tree model, combined into one score
    - **Idea**: a variant common in healthy people is unlikely to cause severe disease

    Questions to discuss:

    - Allele frequency is a strong feature. Why is it also a bit of a shortcut?
    - The features are other tools' predictions. What happens if those tools were trained on the same ClinVar variants?
    - Would ClinPred help us rank A30V vs. A30L in an enzyme?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zero-shot scoring with a PLM

    "Zero-shot" means no labels. We only ask the PLM from Unit 5 one question:

    > With position *i* masked, how much more does the model like the mutant amino acid than the wild-type?

    $$
    \text{score}(\text{wt}_i \to \text{mt}_i) = \log p(\text{mt}_i \mid x_{\setminus i}) - \log p(\text{wt}_i \mid x_{\setminus i})
    $$

    Negative score → the mutant looks less "natural" to the model → probably bad for the protein.

    ### The zero-shot score workflow
    1. Mask a position
    2. Get the log-probabilities for all AAs
    3. Compare the actual mutation score versus the "wild-type" AA score (see formula above)
    4. Repeat for every position → the heatmap below
    """)
    return


@app.cell
def _(AutoTokenizer, EsmForMaskedLM, torch):
    # use the bigger, better model when a GPU is around (e.g. on molab);
    # otherwise stick with the small one so this stays fast on a laptop
    model_name = (
        "facebook/esm2_t33_650M_UR50D" if torch.cuda.is_available()
        else "facebook/esm2_t6_8M_UR50D"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    model_name, device
    return device, model, tokenizer


@app.cell
def _():
    # human ubiquitin -- 76 residues, tags other proteins for destruction,
    # and one of the first proteins with a full deep mutational scan
    wt_sequence = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    len(wt_sequence)
    return amino_acids, wt_sequence


@app.cell
def _(tokenizer, torch, wt_sequence):
    # one copy of the sequence per position, each with that position masked,
    # all scored in a single batch
    wt_ids = tokenizer(wt_sequence, return_tensors="pt")["input_ids"]
    masked_ids = wt_ids.repeat(len(wt_sequence), 1)
    positions = torch.arange(len(wt_sequence))
    masked_ids[positions, positions + 1]
    return masked_ids, positions


@app.cell
def _(masked_ids, positions, tokenizer):
    masked_ids[positions, positions + 1] = tokenizer.mask_token_id  # +1 skips <cls>
    masked_ids[positions, positions + 1]
    return


@app.cell
def _(device, masked_ids, model, positions, torch):
    with torch.no_grad():
        logits = model(input_ids=masked_ids.to(device)).logits

    # log-probs at each masked spot: (sequence length, vocab size)
    masked_log_probs = torch.log_softmax(logits[positions, positions + 1], dim=-1).cpu()
    masked_log_probs
    return (masked_log_probs,)


@app.cell
def _(amino_acids, masked_log_probs, pl, tokenizer, wt_sequence):
    aa_ids = tokenizer.convert_tokens_to_ids(amino_acids)

    rows = []
    for i, wt_aa in enumerate(wt_sequence):
        wt_lp = masked_log_probs[i, tokenizer.convert_tokens_to_ids(wt_aa)].item()
        for mt_aa, mt_id in zip(amino_acids, aa_ids):
            rows.append({
                "position": i + 1,
                "wt": wt_aa,
                "mt": mt_aa,
                "mutant": f"{wt_aa}{i + 1}{mt_aa}",
                "score": masked_log_probs[i, mt_id].item() - wt_lp,
            })

    scores = pl.DataFrame(rows)
    scores
    return (scores,)


@app.cell(hide_code=True)
def _(alt, scores):
    _axis_kws = dict(titleFontSize=16, labelFontSize=13, titleFontWeight="bold")

    alt.Chart(scores).mark_rect().encode(
        x=alt.X("position:O", title="position").axis(**_axis_kws, labelAngle=0, values=list(range(5, 77, 5))),
        y=alt.Y("mt:N", title="mutant amino acid").axis(**_axis_kws),
        color=alt.Color(
            "score:Q",
            title="log p(mt) − log p(wt)",
            scale=alt.Scale(range=["#d03b3b", "#f0efec", "#0d366b"], domainMid=0),
            legend=alt.Legend(
                orient="bottom", direction="horizontal",
                titleFontSize=14, labelFontSize=13, gradientLength=300,
            ),
        ),
        tooltip=["mutant", alt.Tooltip("score:Q", format=".2f")],
    ).properties(width=800, height=380, title="Zero-shot variant effect map for ubiquitin")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to look for:

    - Which columns are red almost everywhere? Those spots can't take any change.
    - Ubiquitin's C-terminal tail (**LRLRGG**, positions 71–76) is how it attaches to other proteins. What does the model think of it?
    - The hydrophobic core (I23, V26, I30, I61). Red or blue?
    - The I44 surface patch is where partner proteins bind. Does the model care?
    """)
    return


@app.cell
def _(mo, scores):
    mutant_picker = mo.ui.dropdown(
        options=scores["mutant"].to_list(),
        value="I44A",
        label="Score a mutant",
        searchable=True,
    )
    mutant_picker
    return (mutant_picker,)


@app.cell
def _(mo, mutant_picker, pl, scores):
    _score = scores.filter(pl.col("mutant") == mutant_picker.value)["score"][0]
    _rank = (scores["score"] < _score).mean()
    mo.md(f"""
    **{mutant_picker.value}** scores **{_score:.2f}**. That is lower than **{1 - _rank:.0%}** of all single mutants of ubiquitin.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ProteinGym: who scores best?

    [ProteinGym](https://proteingym.org/) ([Notin et al., 2023](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1.full)) is a fair playing field for these models.

    - 250+ standardized DMS assays, plus clinical variant sets
    - 70+ models: alignment-based, protein language models, inverse folding
    - Two tracks:
        - **Zero-shot**: no labels, like our heatmap. Scored with Spearman (DMS) and AUC (clinical)
        - **Supervised**: some labels from the assay are used for training

    The score we just computed is the ESM2 zero-shot baseline on the leaderboard.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## This week: pick a model

    After today, pick one **supervised clinical** model from the [ProteinGym](https://proteingym.org/) clinical leaderboard and read its paper. Questions to answer as you read:

    1. What goes **in**? Sequence only, an alignment (MSA), a structure, allele frequencies?
    2. What labels was it trained on, and where did they come from?
    3. Where does it rank on ProteinGym?
    4. Would you trust it for a patient? For picking variants to test in the lab?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project 2 – Notebook 2: from embeddings to fitness

    Zero-shot scores use no labels. When we *do* have DMS measurements, we can train on them:

    1. Embed each variant sequence with a PLM (mean-pool, as in Unit 5)
    2. Fit a regression model: embedding → measured fitness
    3. Check it on variants the model never saw (Spearman correlation)

    Question to keep in mind: does the supervised model beat the zero-shot score? By how much, and with how many labels?
    """)
    return


if __name__ == "__main__":
    app.run()
