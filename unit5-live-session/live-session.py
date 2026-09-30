# /// script
# dependencies = [
#     "altair==6.3.0",
#     "ipython==9.17.1",
#     "marimo",
#     "pandas==3.0.6",
#     "polars==1.44.2",
#     "pyarrow==25.0.1",
#     "torch==2.14.0",
#     "transformers==5.17.0",
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


@app.cell
def _():
    import torch
    import polars as pl
    import altair as alt
    from transformers import AutoTokenizer, EsmForMaskedLM

    return AutoTokenizer, EsmForMaskedLM, alt, pl, torch


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

    # one model, used two ways below: model.esm for the raw embeddings,
    # and model(...) itself for mask-fill predictions
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    model_name, device
    return device, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mask fill task
    """)
    return


@app.cell
def _(mo):
    # melittin -- the main toxin in bee venom, 26 residues, punches
    # holes in cell membranes
    demo_sequence = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    mask_position = mo.ui.slider(
        1, len(demo_sequence), 1,
        label="Position to mask",
        show_value=True,
    )
    mask_position
    return demo_sequence, mask_position


@app.cell
def _(demo_sequence, device, mask_position, model, tokenizer, torch):
    idx = mask_position.value
    true_aa = demo_sequence[idx]
    masked_sequence = demo_sequence[:idx] + tokenizer.mask_token + demo_sequence[idx + 1:]

    mlm_inputs = tokenizer(masked_sequence, return_tensors="pt").to(device)
    mask_index = (mlm_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        mlm_logits = model(**mlm_inputs).logits

    predicted_id = mlm_logits[0, mask_index].argmax(dim=-1)
    predicted_aa = tokenizer.decode(predicted_id.cpu())

    masked_sequence, predicted_aa, true_aa
    return mask_index, mlm_logits, predicted_aa, true_aa


@app.cell
def _(mo, predicted_aa, true_aa):
    mo.md(f"""
    Model guessed **'{predicted_aa}'** for the masked spot. The real amino acid there was **'{true_aa}'**.
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    mask_index,
    mlm_logits,
    pl,
    predicted_aa,
    tokenizer,
    torch,
    true_aa,
):
    mask_probs = torch.softmax(mlm_logits[0, mask_index[0]], dim=-1)
    vocab = tokenizer.convert_ids_to_tokens(list(range(mask_probs.shape[-1])))

    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    aa_rows = [
        (tok, float(p))
        for tok, p in zip(vocab, mask_probs.detach().cpu().numpy())
        if tok in aa_set
    ]
    aa_rows.sort(key=lambda row: row[1], reverse=True)

    highlight = []
    for tok, _ in aa_rows:
        if tok == true_aa and tok == predicted_aa:
            highlight.append("correct guess")
        elif tok == true_aa:
            highlight.append("true amino acid")
        elif tok == predicted_aa:
            highlight.append("model's guess")
        else:
            highlight.append("other")

    prob_df = pl.DataFrame({
        "amino_acid": [row[0] for row in aa_rows],
        "probability": [row[1] for row in aa_rows],
        "highlight": highlight,
    })

    prob_chart = (
        alt.Chart(prob_df)
        .mark_bar(fillOpacity=0.5)
        .encode(
            x=alt.X("amino_acid:N", sort="-y", title="amino acid")
                .axis(titleFontSize=16, labelFontSize=13, titleFontWeight="bold"),
            y=alt.Y("probability:Q", title="probability", axis=alt.Axis(format="%"))
                .axis(titleFontSize=16, labelFontSize=13, titleFontWeight="bold"),
            color=alt.Color(
                "highlight:N",
                title="",
                scale=alt.Scale(
                    domain=["other", "model's guess", "true amino acid", "correct guess"],
                    range=["#c3c2b7", "#2a78d6", "#0ca30c", "#eda100"],
                ),
                legend=alt.Legend(
                    titleFontSize=16,
                    labelFontSize=15,
                    labelFontWeight="bold",
                    symbolSize=300,
                ),
            ),
            tooltip=["amino_acid", "probability", "highlight"],
        )
        .properties(width=700, height=320, title="Predicted probability of each amino acid at the masked spot")
    )
    prob_chart
    return (prob_df,)


@app.cell
def _(prob_df):
    second_choice = prob_df.row(1, named=True)
    second_choice
    return (second_choice,)


@app.cell
def _(mo, predicted_aa, second_choice):
    mo.md(f"""
    If '{predicted_aa}' is the model's top guess, its runner-up here is "
        f"**'{second_choice['amino_acid']}'** at **{second_choice['probability']:.1%}**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does the model network learn?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > It turns out that these learned building blocks are very similar to the
    > basic visual machinery in the human eye, as well as the handcrafted
    > computer vision features that were developed prior to the days of deep
    > learning.
    >
    > — Jeremy Howard & Sylvain Gugger, *Deep Learning for Coders with fastai
    > and PyTorch*, ["What Our Image Recognizer
    Learned"](https://fastai.github.io/fastbook2e/intro.html#what-our-image-recognizer-learned)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://fastai.github.io/fastbook2e/images/layer2.png" />

    *Figure 1.3, "Activations of the second layer of a CNN," courtesy of
    Matthew D. Zeiler and Rob Fergus. Originally published in Zeiler, M. D., &
    Fergus, R. (2013). ['Visualizing and Understanding Convolutional
    Networks'](https://arxiv.org/abs/1311.2901). arXiv:1311.2901. Reproduced in
    Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with fastai and
    PyTorch*. O'Reilly Media.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Amino acid embeddings
    """)
    return


@app.cell
def _(model, tokenizer):
    token_embedding_table = model.esm.embeddings.word_embeddings.weight

    standard_amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

    fixed_embeddings = {
        aa: token_embedding_table[tokenizer.convert_tokens_to_ids(aa)].detach().cpu()
        for aa in standard_amino_acids
    }

    fixed_embeddings
    return fixed_embeddings, standard_amino_acids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Physiochemical properties of AAs
    """)
    return


@app.cell
def _(pl):
    props = pl.DataFrame({
        "aa": list("ARNDCQEGHILKMFPSTWYV"),
        # Kyte & Doolittle 1982 (AAindex KYTJ820101)
        "hydropathy_kd": [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2],
        # Zamyatnin 1972, residue volume in Å³
        "volume_A3":     [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0],
        # Isoelectric point, textbook values
        "pI":            [6.00,10.76,5.41,2.77,5.07,5.65,3.22,5.97,7.59,6.02,5.98,9.74,5.74,5.48,6.30,5.68,5.60,5.89,5.66,5.96],
        # Grantham 1974 polarity (AAindex GRAR740102)
        "polarity_gr":   [8.1,10.5,11.6,13.0,5.5,10.5,12.3,9.0,10.4,5.2,4.9,11.3,5.7,5.2,8.0,9.2,8.6,5.4,6.2,5.9],
    })

    num = pl.exclude("aa")
    props_z = props.with_columns((num - num.mean()) / num.std(ddof=0))

    X_chem = props_z.drop("aa").to_numpy()  # (20, 4), ready for PCA

    props
    return X_chem, props


@app.cell
def _(X_chem, pl, props, torch):
    chem_tensor = torch.tensor(X_chem, dtype=torch.float32)
    chem_centered = chem_tensor - chem_tensor.mean(dim=0)
    U2, S2, V2 = torch.pca_lowrank(chem_centered, q=2)
    chem_pca_coords = (chem_centered @ V2[:, :2]).numpy()

    chem_pca_df = pl.DataFrame({
        "amino_acid": props["aa"].to_list(),
        "PC1": chem_pca_coords[:, 0],
        "PC2": chem_pca_coords[:, 1],
    })
    chem_pca_df
    return (chem_pca_df,)


@app.cell
def _(fixed_embeddings, pl, standard_amino_acids, torch):
    embedding_matrix = torch.stack([fixed_embeddings[aa] for aa in standard_amino_acids])

    centered = embedding_matrix - embedding_matrix.mean(dim=0)
    U, S, V = torch.pca_lowrank(centered, q=2)
    pca_coords = (centered @ V[:, :2]).numpy()

    emb_pca_df = pl.DataFrame({
        "amino_acid": standard_amino_acids,
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
    })
    emb_pca_df
    return (emb_pca_df,)


@app.cell
def _(alt, chem_pca_df, emb_pca_df, props):
    # build fresh, undisplayed chart objects for the side-by-side view
    # (charts already shown on their own pick up marimo display metadata
    # that breaks concatenation), colored by Kyte-Doolittle hydropathy on a
    # diverging scale: red = hydrophilic (negative), blue = hydrophobic
    # (positive), gray = neutral (0)
    _hydro = props.select(["aa", "hydropathy_kd"])
    _pca_colored = emb_pca_df.join(_hydro, left_on="amino_acid", right_on="aa")
    _chem_colored = chem_pca_df.join(_hydro, left_on="amino_acid", right_on="aa")

    # bigger axis titles and tick labels, reused on every axis below
    axis_kws = dict(titleFontSize=16, labelFontSize=13, titleFontWeight="bold")

    _left = (
    alt.Chart(_pca_colored)
        .mark_text(size=20, fontWeight=900)
        .encode(
            x=alt.X("PC1:Q", title="PC1").axis(**axis_kws),
            y=alt.Y("PC2:Q", title="PC2").axis(**axis_kws),
            text="amino_acid:N",
            tooltip=["amino_acid", "PC1", "PC2", "hydropathy_kd"],
            color=alt.Color(
                "hydropathy_kd:Q",
                title="Kyte-Doolittle hydropathy",
                scale=alt.Scale(
                    range=["#d03b3b", "#f0efec", "#0d366b"],
                    domainMid=0,
                ),
                legend=alt.Legend(
                    titleFontSize=16,
                    labelFontSize=14,
                    gradientLength=180,
                    gradientThickness=22,
                ),
            ),
        )
        .properties(width=380, height=320, title="PCA of ESM2 embeddings (context-free)")
    )
    _right = (
    alt.Chart(_chem_colored)
        .mark_text(size=20, fontWeight=900)
        .encode(
            x=alt.X("PC1:Q", title="PC1").axis(**axis_kws),
            y=alt.Y("PC2:Q", title="PC2").axis(**axis_kws),
            text="amino_acid:N",
            tooltip=["amino_acid", "PC1", "PC2", "hydropathy_kd"],
            color=alt.Color(
                "hydropathy_kd:Q",
                title="Kyte-Doolittle hydropathy",
                scale=alt.Scale(
                    range=["#d03b3b", "#f0efec", "#0d366b"],
                    domainMid=0,
                ),
                legend=alt.Legend(
                    titleFontSize=16,
                    labelFontSize=14,
                    gradientLength=180,
                    gradientThickness=22,
                ),
            ),
        )
        .properties(width=380, height=320, title="PCA of physiochemical properties")
    )
    _left | _right
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sequence level embeddings
    """)
    return


@app.cell
def _(demo_sequence):
    len(demo_sequence)
    return


@app.cell
def _(demo_sequence, device, model, tokenizer, torch):
    seq_inputs = tokenizer(demo_sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        seq_outputs = model.esm(**seq_inputs)

    # drop <cls> and <eos>, then average what's left into one vector
    seq_residue_embeddings = seq_outputs.last_hidden_state[0, 1:-1, :]

    seq_residue_embeddings.shape
    return (seq_residue_embeddings,)


@app.cell
def _(seq_residue_embeddings):
    sequence_embedding = seq_residue_embeddings.mean(dim=0).cpu()
    sequence_embedding.shape
    return (sequence_embedding,)


@app.cell
def _(sequence_embedding):
    sequence_embedding[:10]
    return


@app.cell
def _(mo, seq_residue_embeddings, sequence_embedding):
    mo.md(
        f"Averaging all **{seq_residue_embeddings.shape[0]} residue vectors** "
        f"of melittin gives one **{sequence_embedding.shape[0]}-dim vector** "
        "that stands for the whole sequence."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the same trick, just at a huge scale. Meta AI's ESM Metagenomic
    Atlas embedded over a million protein sequences this same way, then squashed
    each one down to a point on a 2D map with an algorithm called **UMAP**
    (a close cousin of t-SNE, and easy to mix up with it).

    <img src="https://www.biorxiv.org/content/biorxiv/early/2022/12/21/2022.07.20.500902/F3.medium.gif" width="1000"/>

    *Figure 3D-E: a "sequence landscape" of 1 million metagenomic protein
    structures, colored by similarity to known structures. From Lin, Z., Akin,
    H., Rao, R., et al. (2023). ['Evolutionary-scale prediction of atomic-level
    protein structure with a language
    model'](https://doi.org/10.1126/science.ade2574). *Science*, 379(6637),
    1123-1130. See also the [ESM Metagenomic
    Atlas](https://esmatlas.com).*
    """)
    return


if __name__ == "__main__":
    app.run()
