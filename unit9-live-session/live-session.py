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
    import torch
    import polars as pl
    import altair as alt
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EsmForMaskedLM,
    )

    return AutoModelForCausalLM, AutoTokenizer, EsmForMaskedLM, alt, pl, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### What have we learned so far?
    - Project 2: **encoder** protein language models (ESM2)
        - Mask fill: guess a hidden amino acid from the residues around it
        - Embeddings for variant effects, function annotation, active learning, RAG
    - An encoder **reads** a protein and hands back numbers

    ### Today: from reading to writing
    - A **decoder** model writes a protein one amino acid at a time
    - Encoder vs. decoder on the same protein: ESM2 vs. ProGen2
    - The autoregressive loop, written out by hand
    - Sampling knobs: temperature and top-p

    ### What's next?
    - Project 3 - Notebook 1: gather and curate a training set for one protein family
    - Unit 10: inverse folding and structure prediction. Sequence alone is not enough.

    ### Open questions
    - How do we judge a protein that nobody has ever seen?
    - What training data does a model need to write sequences that are new *and* plausible?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading vs. writing

    | | Encoder (ESM2) | Decoder (ProGen2) |
    | --- | --- | --- |
    | Training task | Guess a **masked** amino acid | Guess the **next** amino acid |
    | Context it sees | Both sides of the gap | Only what came before |
    | Natural output | Embeddings, likelihood scores | A brand new sequence |
    | We used it for | Variant effects, function, active learning | Writing proteins (Project 3) |

    Both learn from the same natural protein sequences but differ int the training task. The task is what makes one a reader and the other a writer.
    """)
    return


@app.cell
def _(AutoTokenizer, EsmForMaskedLM, torch):
    # same switch as Unit 6: the big model on a GPU, the tiny one on a laptop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_name = (
        "facebook/esm2_t33_650M_UR50D"
        if torch.cuda.is_available()
        else "facebook/esm2_t6_8M_UR50D"
    )

    esm_tokenizer = AutoTokenizer.from_pretrained(esm_name)
    esm_model = EsmForMaskedLM.from_pretrained(esm_name).eval().to(device)
    esm_name, device
    return device, esm_model, esm_tokenizer


@app.cell
def _():
    # human ubiquitin again, our running example from Unit 6
    wt_sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    len(wt_sequence)
    return amino_acids, wt_sequence


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, device, torch):
    # ProGen2: a decoder trained on ~280M natural protein sequences.
    # The Hugging Face port runs the authors' own model code (trust_remote_code),
    # which needs transformers < 5 -- that is why this notebook pins 4.49.
    progen_name = (
        "hugohrban/progen2-large"
        if torch.cuda.is_available()  # 2.8B params, ~20 GB in fp32
        else "hugohrban/progen2-small"  # 151M params, CPU-friendly
    )

    # float32 on purpose. In bfloat16 the same probability can come out up to 0.1
    # different depending on how the sequence was batched, which would quietly
    # corrupt every chart below.
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

    # ProGen2 marks the start of a protein with "1" and the end with "2"
    progen_name, progen_tokenizer.encode("1MQIF2")
    return progen_model, progen_tokenizer


@app.cell
def _(mo, wt_sequence):
    position = mo.ui.slider(
        1,
        len(wt_sequence),
        value=4,
        label="Hide this position",
        show_value=True,
    )
    position
    return (position,)


@app.cell(hide_code=True)
def _(mo, position, wt_sequence):
    _i = position.value - 1
    mo.md(f"""
    `{wt_sequence[:_i]}`**<span style="color:#d03b3b">?</span>**`{wt_sequence[_i + 1 :]}`

    **ESM2** sees both sides of the `?`. **ProGen2** only sees the part on the left.
    The real amino acid at position {position.value} is **{wt_sequence[_i]}**.
    """)
    return


@app.cell
def _(
    amino_acids,
    device,
    esm_model,
    esm_tokenizer,
    progen_model,
    progen_tokenizer,
    torch,
):
    def esm_mask_fill(sequence, i):
        """Encoder: hide position i, then read the probabilities at the gap."""
        ids = esm_tokenizer(sequence, return_tensors="pt")["input_ids"]
        ids[0, i + 1] = esm_tokenizer.mask_token_id  # +1 skips <cls>
        with torch.no_grad():
            logits = esm_model(input_ids=ids.to(device)).logits[0, i + 1]
        probs = logits.float().softmax(-1).cpu()
        return [
            probs[esm_tokenizer.convert_tokens_to_ids(aa)].item()
            for aa in amino_acids
        ]

    def progen_next(prefix):
        """Decoder: feed in everything to the left, read what it wants next."""
        ids = torch.tensor(
            [progen_tokenizer.encode("1" + prefix)], device=device
        )
        with torch.no_grad():
            logits = progen_model(ids).logits[0, -1]
        probs = logits.float().softmax(-1).cpu()
        return [
            probs[progen_tokenizer.convert_tokens_to_ids(aa)].item()
            for aa in amino_acids
        ]

    return esm_mask_fill, progen_next


@app.cell
def _(amino_acids, esm_mask_fill, pl, position, progen_next, wt_sequence):
    _i = position.value - 1
    compare = pl.concat(
        [
            pl.DataFrame(
                {
                    "model": "ESM2 (encoder): sees both sides",
                    "aa": amino_acids,
                    "prob": esm_mask_fill(wt_sequence, _i),
                }
            ),
            pl.DataFrame(
                {
                    "model": "ProGen2 (decoder): sees the left side only",
                    "aa": amino_acids,
                    "prob": progen_next(wt_sequence[:_i]),
                }
            ),
        ]
    ).with_columns(is_real=pl.col("aa") == wt_sequence[_i])

    compare.sort("prob", descending=True).group_by(
        "model", maintain_order=True
    ).head(3)
    return (compare,)


@app.cell(hide_code=True)
def _(alt, amino_acids, compare, position, wt_sequence):
    _axis = dict(titleFontSize=15, labelFontSize=13, titleFontWeight="bold")

    alt.Chart(compare).mark_bar().encode(
        x=alt.X(
            "aa:N",
            title="Amino acid",
            sort=amino_acids,
            axis=alt.Axis(labelAngle=0, **_axis),
        ),
        y=alt.Y(
            "prob:Q",
            title="Probability",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(**_axis),
        ),
        color=alt.Color(
            "is_real:N",
            scale=alt.Scale(
                domain=[True, False], range=["#d03b3b", "#9aa5b1"]
            ),
            legend=alt.Legend(
                title="The real one?",
                titleFontSize=13,
                labelFontSize=12,
                orient="top",
            ),
        ),
        row=alt.Row(
            "model:N",
            title=None,
            sort=[
                "ESM2 (encoder): sees both sides",
                "ProGen2 (decoder): sees the left side only",
            ],
            header=alt.Header(
                labelAngle=0,  # stop the sideways text
                labelOrient="top",  # put the name above its own panel
                labelAlign="left",
                labelAnchor="start",
                labelFontSize=14,
                labelFontWeight="bold",
                labelPadding=4,
            ),
        ),
        tooltip=["model", "aa", alt.Tooltip("prob:Q", format=".3f")],
    ).properties(
        width=620,
        height=150,
    ).properties(
        title=alt.Title(
            f"Position {position.value}: what belongs here?",
            subtitle=f"The real amino acid is {wt_sequence[position.value - 1]}",
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
    ### Things to look for

    - At **position 4** ESM2 is sure the answer is `F`. ProGen2 has only read `MQI`,
      so it guesses `K` and gives the real answer about a 4% chance.
    - Drag the slider to **position 10**. Now ProGen2 is just as sure as ESM2. What changed?
    - Try **position 1**. ESM2 still has all 75 other residues to work with.
      ProGen2 has nothing but the start token, so it falls back on "proteins begin with M".
    - Nothing about ProGen2 is worse. It is answering a harder question at the start,
      and an easier one by the end.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Watch the decoder catch up

    We can score **every** position at once.

    - **ESM2** needs one masked copy of the protein per position: 76 copies, run as
      a single batch.
    - **ProGen2** needs exactly **one** forward pass over the whole protein.

    That second one deserves a moment. Inside a decoder every position is blocked
    from seeing anything to its right, so you can hand it the finished protein and
    no position is able to cheat:

    | Output row | What that row had read | What it predicts |
    | --- | --- | --- |
    | 0 | `1` | position 1 |
    | 1 | `1M` | position 2 |
    | 2 | `1MQ` | position 3 |
    | 3 | `1MQI` | position 4 |

    Row 2 was handed the whole sequence, but it only ever saw `1MQ`. So row *i* is
    already the answer to "what comes after the first *i* residues", and all 76
    answers fall out of one pass.
    """)
    return


@app.cell
def _(
    device,
    esm_model,
    esm_tokenizer,
    pl,
    progen_model,
    progen_tokenizer,
    torch,
    wt_sequence,
):
    # ESM2: one masked copy per position, all in one batch (same trick as Unit 6)
    _wt_ids = esm_tokenizer(wt_sequence, return_tensors="pt")["input_ids"]
    _masked = _wt_ids.repeat(len(wt_sequence), 1)
    _pos = torch.arange(len(wt_sequence))
    _masked[_pos, _pos + 1] = esm_tokenizer.mask_token_id
    with torch.no_grad():
        _esm_logits = esm_model(input_ids=_masked.to(device)).logits[
            _pos, _pos + 1
        ]
    _esm_probs = _esm_logits.float().softmax(-1).cpu()
    _esm_real = [
        _esm_probs[i, esm_tokenizer.convert_tokens_to_ids(aa)].item()
        for i, aa in enumerate(wt_sequence)
    ]

    # ProGen2: a single forward pass over "1" + the whole protein.
    # Row i predicts residue i + 1, so row 0 is the prediction for position 1.
    _ids = torch.tensor(
        [progen_tokenizer.encode("1" + wt_sequence)], device=device
    )
    with torch.no_grad():
        _progen_probs = (
            progen_model(_ids).logits[0, :-1].float().softmax(-1).cpu()
        )
    _progen_real = [
        _progen_probs[i, progen_tokenizer.convert_tokens_to_ids(aa)].item()
        for i, aa in enumerate(wt_sequence)
    ]

    by_position = pl.concat(
        [
            pl.DataFrame(
                {
                    "model": "ESM2 (encoder)",
                    "position": range(1, len(wt_sequence) + 1),
                    "prob": _esm_real,
                }
            ),
            pl.DataFrame(
                {
                    "model": "ProGen2 (decoder)",
                    "position": range(1, len(wt_sequence) + 1),
                    "prob": _progen_real,
                }
            ),
        ]
    ).with_columns(
        residue=pl.col("position").map_elements(
            lambda p: wt_sequence[p - 1], return_dtype=pl.String
        )
    )

    by_position.head()
    return (by_position,)


@app.cell(hide_code=True)
def _(alt, by_position, wt_sequence):
    _axis = dict(titleFontSize=15, labelFontSize=13, titleFontWeight="bold")

    alt.Chart(by_position).mark_line(
        point=alt.OverlayMarkDef(size=26), strokeWidth=2
    ).encode(
        x=alt.X(
            "position:Q",
            title="Position in ubiquitin",
            scale=alt.Scale(domain=[1, len(wt_sequence)], nice=False),
            axis=alt.Axis(**_axis),
        ),
        y=alt.Y(
            "prob:Q",
            title="Probability of the real amino acid",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(**_axis),
        ),
        color=alt.Color(
            "model:N",
            title=None,
            scale=alt.Scale(
                domain=["ESM2 (encoder)", "ProGen2 (decoder)"],
                range=["#0d366b", "#d03b3b"],
            ),
            legend=alt.Legend(
                labelFontSize=13, orient="bottom", symbolSize=140
            ),
        ),
        tooltip=[
            "model",
            "position",
            "residue",
            alt.Tooltip("prob:Q", format=".3f"),
        ],
    ).properties(
        width=740,
        height=320,
    ).properties(
        title=alt.Title(
            "The decoder starts blind, then stops doubting",
            subtitle=[
                "ProGen2 is lost for five residues. After that it never drops below 0.84 again.",
                "ESM2 sees the whole protein the entire time and is still unsure in plenty of places.",
            ],
            fontSize=18,
            subtitleFontSize=13,
            anchor="start",
            offset=10,
        ),
    ).configure_view(stroke=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### That flat red line should bother you

    Once ProGen2 has read about six residues, it is right with near-total
    confidence at **every remaining position** — including spots where ESM2, which
    can see the whole protein, is genuinely torn.

    ProGen2 is not better at protein chemistry than ESM2. It has simply worked out
    which protein this is. Ubiquitin is almost unchanged across all eukaryotes, so
    the training data contains it over and over. From residue six onward the model
    is not predicting. It is **reciting**.

    Keep an eye on that red line. We are about to catch the model doing the same
    thing with whole sequences.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The autoregressive loop

    ProGen2 only ever answers one question: **what comes next?** To write a whole
    protein we ask it over and over, feeding its own answer back in:

    1. Start with a prompt. `1` on its own means "begin a protein"
    2. Run the model and read the scores (logits) for every token
    3. Softmax the scores into probabilities
    4. **Sample** one token from those probabilities
    5. Stick it on the end
    6. Go back to step 2, until the model writes `2` or we run out of patience

    "Autoregressive" is the name for step 5. Every new letter depends on all the
    letters the model already wrote, including its own mistakes.
    """)
    return


@app.cell
def _(device, pl, progen_model, progen_tokenizer, torch):
    def write_protein(prompt="1", n_steps=12, temperature=1.0, seed=0):
        """The loop above, spelled out. Returns one row per step."""
        generator = torch.Generator(device=device).manual_seed(seed)
        ids = progen_tokenizer.encode(prompt)  # 1
        steps = []
        for step in range(n_steps):
            with torch.no_grad():
                logits = progen_model(
                    torch.tensor([ids], device=device)
                ).logits[0, -1]  # 2
            probs = (logits.float() / temperature).softmax(-1)  # 3
            next_id = torch.multinomial(
                probs, 1, generator=generator
            ).item()  # 4
            ids.append(next_id)  # 5
            token = progen_tokenizer.decode([next_id])
            steps.append(
                {
                    "step": step + 1,
                    "picked": token,
                    "its probability": probs[next_id].item(),
                    "protein so far": progen_tokenizer.decode(ids).lstrip("1"),
                }
            )
            if token == "2":  # 6
                break
        return pl.DataFrame(steps)

    write_protein(n_steps=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `model.generate` is that same loop

    Hugging Face wraps the loop in one call. Below we hand ProGen2 the first 20
    residues of ubiquitin and always take the single most likely next token
    (`do_sample=False`, called **greedy** decoding).
    """)
    return


@app.cell
def _(device, mo, progen_model, progen_tokenizer, torch, wt_sequence):
    _prompt_len = 20
    _ids = torch.tensor(
        [progen_tokenizer.encode("1" + wt_sequence[:_prompt_len])],
        device=device,
    )
    _out = progen_model.generate(
        _ids,
        attention_mask=torch.ones_like(_ids),
        max_new_tokens=len(wt_sequence),
        do_sample=False,
        pad_token_id=progen_tokenizer.pad_token_id,
        eos_token_id=progen_tokenizer.convert_tokens_to_ids("2"),
    )
    greedy = (
        progen_tokenizer.decode(_out[0]).replace("<|pad|>", "").strip("12")
    )

    # how far does the greedy output track real ubiquitin?
    _matched = next(
        (i for i, (a, b) in enumerate(zip(greedy, wt_sequence)) if a != b),
        min(len(greedy), len(wt_sequence)),
    )

    mo.md(f"""
    | | Sequence |
    | --- | --- |
    | Prompt (given) | `{wt_sequence[:_prompt_len]}` |
    | ProGen2, greedy | `{greedy}` |
    | Real ubiquitin | `{wt_sequence}` |

    The greedy output matches real ubiquitin for the first **{_matched}** residues.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hold on. It just copied ubiquitin.

    It did. All 76 residues, exactly. Ubiquitin is one of the most conserved
    proteins in biology, so the training data holds it over and over. Give the model
    a real start and greedy decoding **recites** the rest from memory.

    Then look at what comes *after* residue 76. It starts a second copy:
    `...VLRLRGG` **`MQIFVKTLTGK`**`...`

    That is not a glitch. Real polyubiquitin genes such as human *UBB* and *UBC*
    encode several ubiquitins head-to-tail in one transcript, and the model learned
    that too.

    So the recitation is faithful right down to the gene architecture. Which is
    impressive, and completely useless for design:

    - Low perplexity on a famous protein does **not** mean the model can design one.
    - The real question is never "can it reproduce something we already have?"
      but "can it write something new that still works?"

    To get something new we have to sample instead of recite. Which means we need to
    talk about *how* we sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling knobs

    **Temperature** $T$ divides the scores before the softmax:

    $$
    p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
    $$

    - $T < 1$ **sharpens**. The favourite wins even more often, so output is safe and repetitive
    - $T = 1$ leaves the model's own probabilities alone
    - $T > 1$ **flattens**. Rare amino acids get a look in, so output is novel and often junk

    **Top-p** (nucleus sampling) is a separate knob. Sort the amino acids by
    probability, keep adding them to a shortlist until the shortlist is worth $p$ of
    the total, then throw the rest away and sample from the shortlist. It trims the
    long tail of bad options without touching the shape of the good ones.
    """)
    return


@app.cell
def _(
    amino_acids,
    device,
    pl,
    progen_model,
    progen_tokenizer,
    torch,
    wt_sequence,
):
    # what does ProGen2 want after MQIFV, at three temperatures?
    _prefix = wt_sequence[:5]
    _ids = torch.tensor(
        [progen_tokenizer.encode("1" + _prefix)], device=device
    )
    with torch.no_grad():
        _logits = progen_model(_ids).logits[0, -1].float().cpu()
    _aa_ids = progen_tokenizer.convert_tokens_to_ids(amino_acids)

    by_temperature = pl.concat(
        [
            pl.DataFrame(
                {
                    "setting": f"T = {_t}",
                    "aa": amino_acids,
                    "prob": (_logits / _t).softmax(-1)[_aa_ids].tolist(),
                }
            )
            for _t in [0.5, 1.0, 1.5]
        ]
    )

    by_temperature.sort("prob", descending=True).group_by(
        "setting", maintain_order=True
    ).head(3)
    return (by_temperature,)


@app.cell(hide_code=True)
def _(alt, amino_acids, by_temperature, wt_sequence):
    _axis = dict(titleFontSize=14, labelFontSize=12, titleFontWeight="bold")

    alt.Chart(by_temperature).mark_bar(color="#0d366b").encode(
        x=alt.X(
            "aa:N",
            title="Next amino acid",
            sort=amino_acids,
            axis=alt.Axis(labelAngle=0, **_axis),
        ),
        y=alt.Y(
            "prob:Q",
            title="Probability",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(**_axis),
        ),
        row=alt.Row(
            "setting:N",
            title=None,
            sort=["T = 0.5", "T = 1.0", "T = 1.5"],
            header=alt.Header(
                labelAngle=0,
                labelOrient="top",
                labelAlign="left",
                labelAnchor="start",
                labelFontSize=14,
                labelFontWeight="bold",
                labelPadding=4,
            ),
        ),
        tooltip=["setting", "aa", alt.Tooltip("prob:Q", format=".3f")],
    ).properties(
        width=620,
        height=110,
    ).properties(
        title=alt.Title(
            f"What comes after {wt_sequence[:5]}?",
            subtitle="Cold is confident. Hot spreads the bet around.",
            fontSize=18,
            subtitleFontSize=14,
            anchor="start",
            offset=10,
        ),
    ).configure_view(stroke=None)
    return


@app.cell
def _(mo):
    temperature = mo.ui.slider(
        0.1, 2.0, step=0.1, value=1.0, label="Temperature", show_value=True
    )
    top_p = mo.ui.slider(
        0.1, 1.0, step=0.05, value=1.0, label="Top-p", show_value=True
    )
    prompt = mo.ui.text(value="1", label="Prompt")
    n_samples = mo.ui.slider(1, 8, value=4, label="How many", show_value=True)
    seed = mo.ui.number(0, 9999, value=0, label="Seed")
    go = mo.ui.run_button(label="Write proteins")

    mo.vstack(
        [
            mo.hstack([temperature, top_p], justify="start", gap=2),
            mo.hstack([prompt, n_samples, seed, go], justify="start", gap=2),
        ]
    )
    return go, n_samples, prompt, seed, temperature, top_p


@app.cell
def _(
    device,
    go,
    mo,
    n_samples,
    pl,
    progen_model,
    progen_tokenizer,
    prompt,
    seed,
    temperature,
    top_p,
    torch,
):
    mo.stop(
        not go.value, mo.md("Set the knobs, then click **Write proteins**.")
    )

    # fixed seed so a demo in class is repeatable; change it to redraw
    torch.manual_seed(seed.value)

    _ids = torch.tensor([progen_tokenizer.encode(prompt.value)], device=device)
    _out = progen_model.generate(
        _ids,
        attention_mask=torch.ones_like(_ids),
        max_new_tokens=200,
        do_sample=True,
        temperature=temperature.value,
        top_p=top_p.value,
        num_return_sequences=n_samples.value,
        pad_token_id=progen_tokenizer.pad_token_id,
        eos_token_id=progen_tokenizer.convert_tokens_to_ids("2"),
    )

    samples = (
        pl.DataFrame(
            {
                "sequence": [
                    progen_tokenizer.decode(r).replace("<|pad|>", "")
                    for r in _out
                ]
            }
        )
        .with_columns(
            finished=pl.col("sequence").str.ends_with("2"),
        )
        .with_columns(
            sequence=pl.col("sequence").str.strip_chars("12"),
        )
        .with_columns(
            length=pl.col("sequence").str.len_chars(),
            # share taken by the single commonest letter. High means low-complexity mush.
            commonest_letter=pl.col("sequence").map_elements(
                lambda s: (
                    max(s.count(a) for a in set(s)) / len(s) if s else None
                ),
                return_dtype=pl.Float64,
            ),
        )
    )

    samples
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Things to try

    - **T = 0.2.** Watch `commonest_letter`. Cold sampling collapses into runs of a
      single residue — `QQQQQQ`, `AAAAAA`. Some sequences end up 99% one letter.
      Safe is not the same as good.
    - **T = 1.8.** Look at `commonest_letter` again. It is *low*, so by that measure
      the hot sequences look healthier than the cold ones. Our junk detector only
      catches one kind of junk.
    - **T = 1.8, still.** Now look at `finished` and `length`. Most of them stop early.
      Why? Because `2`, the end token, is just another token — heat raises its odds
      too. The model is not deciding the protein is done. It is tripping over the exit.
    - **T = 1.8 with top-p = 0.8.** Almost nothing finishes now. Top-p cut the rare
      tokens, and `2` was one of them. Two knobs, pulling on the same thing.
    - Put `1MQIFVKT` in the prompt and raise the temperature. How hot does it have to
      get before the model stops reciting ubiquitin?

    The point: **every cheap quality number we invent measures one failure mode and
    misses the others.** Keep that in mind for the next section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Open question: how do we judge what it wrote?

    We now have a machine that writes protein sequences on demand. Nothing so far
    tells us whether any of them are **good**.

    The trap is that our two obvious instincts pull in opposite directions:

    | We want it to be... | Cheap test | What goes wrong if we only use this |
    | --- | --- | --- |
    | **Plausible** — looks like a real protein | High likelihood under the model | The safest high-likelihood sequence is a copy of something already in the training set |
    | **New** — not just a copy | Low identity to anything natural | Random letters score perfectly on novelty and fold into nothing |

    So neither number means much on its own. A useful generative model has to land
    in the awkward middle: **sequence-distinct from anything observed, yet still
    structurally sound.**

    That is the claim the ProGen2 fine-tuning paper has to defend, and it is why
    the authors reach for four checks rather than one:

    1. **HMMER** — does the sequence still score as a member of the target family?
    2. **Sequence identity** — how close is the nearest natural relative?
    3. **ESMFold** — does it fold, and how confident is the structure prediction?
    4. **Attention maps** — is the model looking at the parts we would expect?

    Checks 3 and 4 are where Unit 10 picks up.

    **Discussion:** if you could only keep one of those four checks, which would you
    keep, and what would you be giving up?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## This week

    **Readings**

    - Hugging Face: *Introduction to Fine-tuning* — the `Trainer` API, dynamic padding,
      `TrainingArguments`, and writing a `compute_metrics` function
    - *Protein Family Sequence Generation through ProGen2 Fine-Tuning* (IEEE) — adding
      family tokens to the vocabulary and fine-tuning on seven Pfam families

    **Do**

    - Unit 9 Knowledge Check
    - **Project 3 – Notebook 1**: gather and curate a training set for one protein family

    Notebook 1 is where today's open question turns practical. Before you can
    fine-tune, you have to decide what counts as a member of your family, how many
    sequences you need, how similar they are allowed to be to each other, and what
    you will hold out to test with.
    """)
    return


if __name__ == "__main__":
    app.run()
