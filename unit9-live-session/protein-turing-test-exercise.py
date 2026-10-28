# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "polars==1.42.1",
#     "pyarrow==25.0.0",
#     "torch==2.11.0",
#     "transformers==4.49.0",
# ]
# ///
#
# NOTE: transformers is pinned below 5 on purpose. ProGen2 on the Hugging Face
# Hub ships the authors' own modelling code (trust_remote_code=True), and that
# code does not run on the transformers 5 API.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import io
    import random
    import urllib.parse
    import urllib.request

    import altair as alt
    import polars as pl
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EsmForMaskedLM,
    )

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        EsmForMaskedLM,
        alt,
        io,
        pl,
        random,
        torch,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The protein Turing test

    This notebook gives you 60 protein sequences.

    30 sequences are natural globins from UniProt. ProGen2 wrote the other 30.
    ProGen2 received the first part of a natural globin. ProGen2 then wrote the
    remainder of the sequence.

    The notebook mixes the two groups together. It hides the labels.
    **Find the sequences that ProGen2 wrote.**

    You have the necessary tool. In Project 2 you scored proteins with ESM2.
    This task uses the same method to answer a new question:

    > A model writes a protein. Can a test show that the protein is not natural?

    Do the steps in sequence. Step 7 shows the answers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    This notebook loads two models. The two models do different work.

    **ProGen2** is the decoder from today. ProGen2 writes the generated sequences.

    **ESM2** is the encoder from Project 2. ESM2 scores the sequences. ESM2 does
    not write any of them.

    Keep the two jobs separate in your mind. The model that writes a sequence
    must not also judge that sequence.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, EsmForMaskedLM, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    esm_name = (
        "facebook/esm2_t33_650M_UR50D"
        if torch.cuda.is_available()
        else "facebook/esm2_t6_8M_UR50D"
    )
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_name)
    esm_model = EsmForMaskedLM.from_pretrained(esm_name).eval().to(device)

    progen_name = (
        "hugohrban/progen2-large"
        if torch.cuda.is_available()
        else "hugohrban/progen2-small"
    )
    progen_tokenizer = AutoTokenizer.from_pretrained(
        progen_name, trust_remote_code=True
    )
    progen_model = (
        AutoModelForCausalLM.from_pretrained(
            progen_name, trust_remote_code=True, torch_dtype=torch.float32
        )
        .eval()
        .to(device)
    )

    esm_name, progen_name, device
    return device, progen_model, progen_tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1. Get natural globins

    Globins carry oxygen. Haemoglobin and myoglobin are globins. Related proteins
    occur in animals, in plants and in bacteria. Pfam gives this family the code
    **PF00042**.

    The next cell asks UniProt for reviewed globins. It keeps the proteins that
    have 130 to 160 amino acids.
    """)
    return


@app.cell
def _(io, pl, urllib):
    _url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        + urllib.parse.urlencode(
            {
                "format": "tsv",
                "query": "reviewed:true AND xref:pfam-PF00042 AND length:[130 TO 160]",
                "fields": "accession,protein_name,organism_name,length,sequence",
            }
        )
    )
    with urllib.request.urlopen(_url) as _response:
        raw = pl.read_csv(
            io.BytesIO(_response.read()), separator="\t", quote_char=None
        )

    raw.select("Entry", "Organism", "Length").head()
    return (raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2. Clean the data

    First, examine the data. This is one of the sequences:

    ```
    SLSDKDKTFVKAFWGKLKGKADDVGAEALARMFGAFPATKSYFAHWPDLSAGSGPVKKHGKIIMAGVGDAVDKIDNLVSGLSKLSDLHATKLRIDXXXXXXXXXXXLVTLAANFPADFTPELHVSLDKFFAAVGAALSDKYR
    ```

    The letter `X` shows an unknown amino acid. Other sequences contain the
    letters `B`, `Z`, `U` and `O`. Some sequences occur more than one time.

    **Your task.** Make a table with the name `globins`:

    1. Keep only the sequences that contain the 20 standard amino acids.
    2. Remove the sequences that occur more than one time.
    3. Sort the table by the `Entry` column. This makes the result repeatable.

    The expression `set(s) <= STANDARD` is true if all the letters in `s` are
    standard.

    <details>
    <summary><b>Hint</b></summary>

    - `set(s) <= STANDARD` is `True` if all the letters in `s` are standard.
    - Use `pl.col("Sequence").map_elements(..., return_dtype=pl.Boolean)` inside
      `.filter()`.
    - `.unique(subset="Sequence")` removes the repeated sequences.
    - `.sort("Entry")` makes the result repeatable.

    </details>
    """)
    return


@app.cell
def _(raw):
    STANDARD = set("ACDEFGHIKLMNPQRSTVWY")

    # TODO: make the table `globins` from the table `raw`.
    #   1. Keep the rows that contain only the 20 standard amino acids.
    #   2. Remove the sequences that occur more than one time.
    #   3. Sort the table by the "Entry" column.
    globins = ...

    len(raw), len(globins)
    return (globins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3. Make the generated sequences

    The cell below selects 60 natural globins. It keeps 30 globins without a change.
    It uses the other 30 globins as **donors**.

    For each donor, the cell keeps the first few amino acids. It gives these amino
    acids to ProGen2. ProGen2 then writes the remainder to the same length.

    Why does ProGen2 get a start? ProGen2 does not know the globin family. Without
    a start, ProGen2 writes a general protein. These first amino acids are a simple
    substitute for family conditioning. The ProGen2 paper does this correctly: it
    adds a family token and then it does fine-tuning.

    **The slider controls the length of the start.** This is the most important
    control in the notebook. Use 15 for your first run. Then try other values.

    Generation needs about 2 minutes. Click the button when you are ready.
    """)
    return


@app.cell
def _(mo):
    prompt_length = mo.ui.slider(
        5,
        60,
        step=5,
        value=15,
        label="Real amino acids given to ProGen2",
        show_value=True,
    )
    forge = mo.ui.run_button(label="Generate the sequences")

    mo.hstack([prompt_length, forge], justify="start", gap=2)
    return forge, prompt_length


@app.cell
def _(
    device,
    forge,
    globins,
    mo,
    progen_model,
    progen_tokenizer,
    prompt_length,
    random,
    torch,
):
    mo.stop(
        not forge.value,
        mo.md("Set the slider, then click **Generate the sequences**."),
    )

    random.seed(0)
    _picked = random.sample(range(len(globins)), 60)
    natural = [globins["Sequence"][i] for i in _picked[:30]]
    donors = [globins["Sequence"][i] for i in _picked[30:]]

    torch.manual_seed(0)
    generated = []
    with mo.status.progress_bar(
        total=len(donors), title="Writing proteins"
    ) as _bar:
        for _seq in donors:
            _start = torch.tensor(
                [progen_tokenizer.encode("1" + _seq[: prompt_length.value])],
                device=device,
            )
            _out = progen_model.generate(
                _start,
                attention_mask=torch.ones_like(_start),
                max_new_tokens=len(_seq) - prompt_length.value,
                min_new_tokens=len(_seq) - prompt_length.value,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=progen_tokenizer.pad_token_id,
                eos_token_id=progen_tokenizer.convert_tokens_to_ids("2"),
            )
            generated.append(
                progen_tokenizer.decode(_out[0])
                .replace("<|pad|>", "")
                .strip("12")
            )
            _bar.update()

    generated[0]
    return generated, natural


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4. Mix the sequences and hide the answers

    The table `mystery` is the data that you can examine.

    The table `answer_key` contains the correct labels. Do not read `answer_key`
    before Step 7.
    """)
    return


@app.cell
def _(generated, natural, pl, random):
    _rows = [{"sequence": s, "truth": "natural"} for s in natural] + [
        {"sequence": s, "truth": "generated"} for s in generated
    ]
    random.Random(1).shuffle(_rows)

    answer_key = pl.DataFrame(_rows).with_row_index("id")
    mystery = answer_key.select("id", "sequence")

    mystery.head()
    return answer_key, mystery


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5. Score each sequence with ESM2

    ESM2 reads the full sequence at the same time. For each position, ESM2 gives a
    probability for the amino acid that occurs there.

    Take the log of each probability. Then calculate the mean for the sequence:

    $$
    \text{score} = \frac{1}{L} \sum_{i=1}^{L} \log p(x_i \mid x)
    $$

    The score is a negative number. A value near zero shows that the protein looks
    usual to ESM2.

    A more accurate method hides one position at a time. That method needs $L$
    forward passes. One pass is sufficient for this task.

    **Your task.** Write the function. It receives one sequence. It returns one
    float value.

    <details>
    <summary><b>Hint</b></summary>

    ```python
    ids = esm_tokenizer(sequence, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        log_probs = esm_model(input_ids=ids).logits[0].float().log_softmax(-1)
    targets = ids[0, 1:-1]          # do not use <cls> and <eos>
    positions = torch.arange(1, len(targets) + 1)
    return log_probs[positions, targets].mean().item()
    ```

    </details>
    """)
    return


@app.function
def esm_score(sequence):
    """Calculate the mean log probability for each amino acid with ESM2."""
    # TODO
    #   1. Tokenize the sequence. Move the input ids to `device`.
    #   2. Do one forward pass. Use torch.no_grad().
    #   3. Apply log_softmax to the last axis of the logits.
    #   4. Do not use <cls> at the start and <eos> at the end.
    #   5. Get the log probability of each amino acid that occurs.
    #   6. Return the mean as a float value.
    ...


@app.cell
def _(mystery, pl):
    scored = mystery.with_columns(
        esm=pl.col("sequence").map_elements(esm_score, return_dtype=pl.Float64)
    )

    scored.select("id", "esm").head()
    return (scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6. Examine the data before you decide

    Make the plot before you select a threshold. If the two groups overlap fully,
    no threshold can separate them.
    """)
    return


@app.cell(hide_code=True)
def _(alt, scored):
    _axis = dict(titleFontSize=15, labelFontSize=13, titleFontWeight="bold")

    alt.Chart(scored).mark_bar(color="#0d366b", opacity=0.65).encode(
        x=alt.X(
            "esm:Q",
            bin=alt.Bin(maxbins=40),
            title="ESM2 score",
            axis=alt.Axis(**_axis),
        ),
        y=alt.Y("count()", title="Number of proteins", axis=alt.Axis(**_axis)),
        tooltip=[alt.Tooltip("count()", title="proteins")],
    ).properties(
        width=640,
        height=300,
        title=alt.Title(
            "60 proteins without labels",
            subtitle="Are there two groups? Where do you put the threshold?",
            fontSize=18,
            subtitleFontSize=14,
            anchor="start",
            offset=10,
        ),
    ).configure_view(stroke=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7. Select a threshold

    Select **one** threshold. A sequence with a score below the threshold is a
    generated sequence.

    Select the threshold before you look at the answers. This is the most
    important part of the task.

    **Your task.** Add a column with the name `guess`. Each value is `"natural"`
    or `"generated"`.

    <details>
    <summary><b>Hint</b></summary>

    - Read the threshold from the plot in Step 6. Put it between the two groups.
    - Use `pl.when(...).then(pl.lit("generated")).otherwise(pl.lit("natural"))`.
    - A value near `-0.30` is a good first try.

    </details>
    """)
    return


@app.cell
def _(scored):
    # TODO: select one threshold. Then give each row the label
    #       "natural" or "generated".
    CUTOFF = ...

    guesses = scored.with_columns(guess=...)

    guesses.group_by("guess").len()
    return CUTOFF, guesses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 8. The answers

    The next cell adds the correct labels to your table.
    """)
    return


@app.cell
def _(CUTOFF, answer_key, guesses, mo, pl, prompt_length):
    graded = guesses.join(answer_key.select("id", "truth"), on="id")
    accuracy = (graded["guess"] == graded["truth"]).mean()

    def auc(column):
        _natural = graded.filter(pl.col("truth") == "natural")[
            column
        ].to_list()
        _generated = graded.filter(pl.col("truth") == "generated")[
            column
        ].to_list()
        _wins = sum(
            (r > f) + 0.5 * (r == f) for r in _natural for f in _generated
        )
        return _wins / (len(_natural) * len(_generated))

    mo.md(f"""
    | Measurement | Value |
    | --- | --- |
    | Real amino acids given to ProGen2 | {prompt_length.value} |
    | Your threshold | {CUTOFF} |
    | Your accuracy | **{accuracy:.0%}** |
    | AUC for the ESM2 score | **{auc("esm"):.3f}** |

    Accuracy shows the result for your threshold. AUC shows the result for all
    possible thresholds. An AUC of 1.0 is a perfect test. An AUC of 0.5 is a guess.
    """)
    return (graded,)


@app.cell(hide_code=True)
def _(alt, graded):
    _axis = dict(titleFontSize=15, labelFontSize=13, titleFontWeight="bold")

    alt.Chart(graded).mark_bar(opacity=0.65).encode(
        x=alt.X(
            "esm:Q",
            bin=alt.Bin(maxbins=40),
            title="ESM2 score",
            axis=alt.Axis(**_axis),
        ),
        xOffset=alt.XOffset("truth:N", sort=["natural", "generated"]),
        y=alt.Y(
            "count()",
            title="Number of proteins",
            stack=None,
            axis=alt.Axis(**_axis),
        ),
        color=alt.Color(
            "truth:N",
            title=None,
            scale=alt.Scale(
                domain=["natural", "generated"], range=["#0d366b", "#d03b3b"]
            ),
            legend=alt.Legend(
                labelFontSize=13, orient="bottom", symbolSize=140
            ),
        ),
        tooltip=["truth", alt.Tooltip("count()", title="proteins")],
    ).properties(
        width=640,
        height=300,
        title=alt.Title(
            "The same 60 proteins with labels",
            subtitle="Did your threshold go in the correct place?",
            fontSize=18,
            subtitleFontSize=14,
            anchor="start",
            offset=10,
        ),
    ).configure_view(stroke=None)
    return


if __name__ == "__main__":
    app.run()
