# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "numpy==2.4.6",
#     "polars==2.0.0",
#     "pyarrow==25.0.0",
#     "torch==2.11.0",
#     "transformers==5.20.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def outline(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### What have we learned so far?
    - Protein language models learn from sequences alone (Unit 8, Unit 9)
    - Inverse folding starts with a structure and predicts sequence (Unit 11)

    ### Today: DNA language models
    - Why DNA is a different kind of language
    - Evo 2: a DNA model trained on genomes from all domains of life
    - Carbon-X: a 14B model that reads DNA and answers in plain English
      - How it is trained
      - Try it on a real sequence
      - What it gets wrong
    - Sparse autoencoders: a way to look inside a model and see what it learned

    ### What's next?
    - Exercise: train a small sparse autoencoder on ESM2-8M and find out what biology the model learned
    """)
    return


@app.cell(hide_code=True)
def why_dna(mo):
    mo.md(r"""
    ## DNA versus protein sequences

    A protein language model reads **20 letters** (amino acids). A DNA language model reads **4 letters**: A, C, G and T.

    | | Protein | DNA |
    | --- | --- | --- |
    | Letters | 20 | 4 |
    | Typical length | a few hundred | thousands to billions |
    | Where the meaning is | in the letters and the fold | in the letters, the reading frame, and far-away neighbors |

    Four things make DNA harder:

    - **Few letters, so little signal per letter.** One amino acid says a lot. One DNA base says very little.
    - **Very long.** The human genome is about 3 billion bases. A model needs a long **context** (how much text it reads at once).
    - **Mostly not protein.** Only about 1 to 2% of the human genome codes for protein. The rest holds switches, repeats and other things. A protein model never sees these.
    - **Meaning depends on the frame.** The cell reads DNA in groups of 3 bases (a **codon**). Shift the start by one base and you read a different protein.

    DNA also has two strands. The other strand is the **reverse complement**: read it backwards and swap A with T and C with G. A gene can sit on either strand.
    """)
    return


@app.cell(hide_code=True)
def evo2(mo):
    mo.md(r"""
    ## Evo 2

    Evo 2 is a DNA language model trained on genomes from all domains of life.

    For a visual walk-through, read [The Illustrated Evo 2](https://research.nvidia.com/labs/dbr/blog/illustrated-evo2/) from NVIDIA.
    """)
    return


@app.cell(hide_code=True)
def try_carbon_x(mo):
    mo.md(r"""
    ## Try Carbon-X

    Carbon-X reads DNA and English in one stream. Paste a sequence into the chat and ask a question. Runs of 6 or more A, C, G, T are tagged as DNA automatically.

    Try the starter prompts below.

    - The second sequence is the **reverse complement** of the first: the same gene, read from the other strand. Can Carbon-X tell?
    - The next two prompts hide a **single-base variant** in a made-up gene. Does Carbon-X find it, and does it say what it does to the protein?
    - The last two variant prompts use a **real human gene** (HBB) with a **disease-causing variant**. The first uses the prompt format from the Carbon-X model card. Read the reasoning, not only the answer. Does it describe the change correctly? Run it a few times.
    """)
    return


@app.cell
def demo_sequences():
    def revcomp(seq):
        return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))

    demo_orf = "ATGGCTGAACTGAAAGGTATCGTTGCTGACTTCCGTAAGCTGGGTTAA"
    demo_orf_rc = revcomp(demo_orf)

    # Two single-base variants of the made-up gene
    variant_missense = (
        demo_orf[:12] + "G" + demo_orf[13:]
    )  # codon 5: AAA (K) -> GAA (E)
    variant_nonsense = (
        demo_orf[:6] + "T" + demo_orf[7:]
    )  # codon 3: GAA (E) -> TAA (stop)

    variant_question = (
        "Here is a reference DNA sequence: {ref}\n"
        "Here is a variant: {var}\n"
        "What is different, and what does it do to the protein?"
    )

    # The prompt format from the Carbon-X model card
    pathogenic_question = (
        "Reference: {ref}\n"
        "Variant: {var}\n\n"
        "Is this variant likely to be benign or pathogenic? Explain your reasoning."
    )
    return (
        demo_orf,
        demo_orf_rc,
        pathogenic_question,
        variant_missense,
        variant_nonsense,
        variant_question,
    )


@app.cell
def human_variant_sequences():
    # Human HBB (hemoglobin subunit beta) coding sequence, NCBI RefSeq NM_000518.5, 444 bases
    hbb_cds = (
        "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAG"
        "TTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGG"
        "GGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGT"
        "GCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACT"
        "GTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCA"
        "TCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAAT"
        "GCCCTGGCCCACAAGTATCACTAA"
    )
    # The sickle cell variant: c.20A>T, codon 7 GAG (Glu) to GTG (Val)
    hbb_sickle = hbb_cds[:19] + "T" + hbb_cds[20:]
    return hbb_cds, hbb_sickle


@app.cell(hide_code=True)
def carbon_x_fn(model, tokenizer, torch):
    def carbon_x_chat(messages, config):
        """Answer the chat with Carbon-X. Thinking is off, as the model card recommends."""
        _history = [{"role": m.role, "content": m.content} for m in messages]
        _inputs = tokenizer.apply_chat_template(
            _history,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        with torch.no_grad():
            _out = model.generate(
                **_inputs,
                max_new_tokens=min(
                    int(getattr(config, "max_tokens", 512) or 512), 512
                ),
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )
        return tokenizer.decode(
            _out[0][_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    return (carbon_x_chat,)


@app.cell(hide_code=True)
def chat(
    carbon_x_chat,
    demo_orf,
    demo_orf_rc,
    hbb_cds,
    hbb_sickle,
    mo,
    pathogenic_question,
    variant_missense,
    variant_nonsense,
    variant_question,
):
    chat = mo.ui.chat(
        carbon_x_chat,
        prompts=[
            f"What can you tell me about this DNA? {demo_orf}",
            f"Is this the same gene as {demo_orf}? Here it is: {demo_orf_rc}",
            variant_question.format(ref=demo_orf, var=variant_missense),
            variant_question.format(ref=demo_orf, var=variant_nonsense),
            pathogenic_question.format(ref=hbb_cds, var=hbb_sickle),
            f"What gene is this, and is this variant harmful?\nReference: {hbb_cds}\nVariant: {hbb_sickle}",
            f"Translate this DNA into protein: {demo_orf}",
            "What is a codon?",
        ],
        max_height=520,
    )
    chat
    return


@app.cell(hide_code=True)
def variant_answer_key(mo):
    mo.accordion(
        {
            "Answer key for the variant prompts": mo.md(
                """
            | Variant | DNA change | Protein change | Type |
            | --- | --- | --- | --- |
            | Made-up gene, first | codon 5: AAA to GAA | K5E (lysine to glutamate) | Missense |
            | Made-up gene, second | codon 3: GAA to TAA | E3* (glutamate to stop) | Nonsense |
            | HBB (hemoglobin beta) | c.20A>T, codon 7: GAG to GTG | E7V (glutamate to valine) | Missense, **sickle cell disease** |

            Made-up gene protein: `MAELKGIVADFRKLG`

            Carbon-X often answers "pathogenic" for HBB, which is right. Its reasoning is often wrong. It may call the change synonymous or name the wrong bases. A right answer is not a right explanation.

            The HBB variant is called Glu6Val in older papers. They do not count the first Met. The sequence is RefSeq NM_000518.5.
            """
            )
        }
    )
    return


@app.cell(hide_code=True)
def how_trained(mo):
    mo.md(r"""
    ## How Carbon-X is trained

    Carbon-X starts from **Qwen3-14B**, a text model. It is trained in four steps.

    | Step | What happens | Size |
    | --- | --- | --- |
    | 0. Add DNA to the vocabulary | 5,515 new DNA tokens, mostly **6-mers** (4,096 of the 6-letter words). Their embeddings start random. First, train only the new embeddings with the rest of the model frozen. Then unfreeze everything. | about 10B tokens frozen |
    | 1. Mid-training | Learn to read DNA. Same next-token prediction as pre-training, on a new data mix. Mix: **55% raw DNA**, 20% English, 15% biology text, 10% instructions. | 302B tokens, 8,192 tokens per sequence |
    | 2. Supervised fine-tuning (SFT) | Learn to answer questions. Examples are question and answer pairs from 20 datasets. The loss counts only the answer. About 30% of the mix is **variant-effect reasoning**. | 3 passes over the data |
    | 3. Reinforcement learning (RL) | The model writes answers. Answers that are checkable and correct get a reward (GRPO). The rewards also penalize reasoning that is not grounded in the sequence. | two rounds: 98,433 then 58,249 prompts, 16 answers each |

    Why keep the English? The same model reads your question, reads the DNA and writes the answer. DNA and text share **one token stream** and **one set of weights**.

    Compare the last step with what you saw in the chat. The training rewards grounded reasoning. The reasoning you saw was still often wrong. A reward can push a model toward good answers. It does not guarantee them.
    """)
    return


@app.cell
def imports():
    import altair as alt
    import numpy as np
    import polars as pl

    return alt, np, pl


@app.cell(hide_code=True)
def superposition_intro(mo):
    mo.md(r"""
    ## Looking inside: what does an embedding hold?

    Inside Carbon-X, every token becomes a list of **5,120 numbers**. This list is an **embedding** (or hidden state). Is each number one real-world idea, like "this is a gene" or "this is a repeat"?

    Not usually. A model has more ideas to store than numbers to store them in. So it packs several ideas into the same numbers. This is called **superposition**.

    Here is a toy with 3 ideas and only 2 numbers.
    """)
    return


@app.cell(hide_code=True)
def superposition_toy(alt, np, pl):
    _rng = np.random.default_rng(0)
    _ideas = ["Coding", "Repeat", "Promoter"]
    _angles = np.deg2rad([90, 210, 330])
    _rows = []
    for _name, _a in zip(_ideas, _angles):
        _direction = np.array([np.cos(_a), np.sin(_a)])
        for _strength in _rng.uniform(0.3, 1.0, 40):
            _x, _y = _strength * _direction + _rng.normal(0, 0.04, 2)
            _rows.append({"idea": _name, "dim1": _x, "dim2": _y})
    _points = pl.DataFrame(_rows)
    _scale = alt.Scale(domain=[-1.25, 1.25])
    _color = alt.Color(
        "idea:N",
        scale=alt.Scale(
            domain=_ideas, range=["#d62728", "#7f8790", "#2f6fdb"]
        ),
        title="Idea",
    )
    _dots = (
        alt.Chart(_points)
        .mark_circle(size=50, opacity=0.7)
        .encode(
            x=alt.X("dim1:Q", scale=_scale, title="Number 1"),
            y=alt.Y("dim2:Q", scale=_scale, title="Number 2"),
            color=_color,
            tooltip=["idea"],
        )
    )
    _dots.properties(
        width=380,
        height=380,
        title="Each dot is one input. Each color is the idea it carries.",
    )
    return


@app.cell(hide_code=True)
def superposition_after(mo):
    mo.md(r"""
    Look at the two axes.

    - **Number 2 is low for two different ideas** (Repeat and Promoter). If you read only this number, you cannot tell them apart.
    - **No idea sits on an axis.** Each idea points in its own **direction**. It is spread over both numbers.

    The ideas are there. You find them in the directions, not in the individual numbers. In a real model with 5,120 numbers and thousands of ideas, nobody can read the directions by eye.

    That is the job of a **sparse autoencoder**.
    """)
    return


@app.cell(hide_code=True)
def hidden_intro(mo):
    mo.md(r"""
    ## Carbon-X hidden states

    Let us look at real embeddings. We feed the human HBB coding sequence to Carbon-X and save the hidden state after **layer 20 of 40**. We use a middle layer because the model has built up a rich picture of the sequence by then, but has not yet turned it into a next-token guess.

    Each row is one DNA token (6 bases). Each column is one of the 5,120 numbers. We show the first 64.
    """)
    return


@app.cell
def hidden_states_fn(hbb_cds, model, tokenizer, torch):
    probe_layer = 20

    def dna_hidden_states(seq, layer=probe_layer):
        """Hidden state after `layer` for each DNA token of `seq`. Returns (tokens, array of shape (n_tokens, 5120))."""
        _ids = tokenizer(
            "<dna>" + seq + "</dna>",
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")["input_ids"]
        with torch.no_grad():
            _out = model(input_ids=_ids, output_hidden_states=True)
        _h = _out.hidden_states[layer][0].float().cpu().numpy()
        _tokens = tokenizer.convert_ids_to_tokens(_ids[0])
        return _tokens[1:-1], _h[1:-1]  # drop the <dna> and </dna> markers

    hbb_tokens, hbb_hidden = dna_hidden_states(hbb_cds)
    hbb_hidden.shape
    return hbb_hidden, hbb_tokens


@app.cell(hide_code=True)
def hidden_heatmap(alt, hbb_hidden, hbb_tokens, pl):
    _n_dims = 64
    _long = pl.DataFrame(
        {
            "token": [
                f"{_i + 1:02d} {_t}"
                for _i, _t in enumerate(hbb_tokens)
                for _ in range(_n_dims)
            ],
            "number": [_d + 1 for _ in hbb_tokens for _d in range(_n_dims)],
            "value": hbb_hidden[:, :_n_dims].reshape(-1).tolist(),
        }
    )
    alt.Chart(_long).mark_rect().encode(
        x=alt.X(
            "number:O",
            title="Embedding number (first 64 of 5,120)",
            axis=alt.Axis(labelOverlap=True, values=[1, 16, 32, 48, 64]),
        ),
        y=alt.Y(
            "token:O",
            title="DNA token (HBB)",
            sort=None,
            axis=alt.Axis(labelFontSize=7),
        ),
        color=alt.Color(
            "value:Q",
            title="Value",
            scale=alt.Scale(scheme="redblue", domain=[2, -2], clamp=True),
        ),
        tooltip=["token", "number", alt.Tooltip("value:Q", format=".2f")],
    ).properties(
        width=560, height=620, title="Layer 20 hidden states of Carbon-X"
    )
    return


@app.cell(hide_code=True)
def hidden_after(mo):
    mo.md(r"""
    It looks like noise. No column has a name. No column says "exon" or "repeat". The picture holds a lot of information, but it is mixed together, as in the toy.

    Can we pull the ideas apart? That is what a sparse autoencoder does.
    """)
    return


@app.cell(hide_code=True)
def sae_intro(mo):
    mo.md(r"""
    ## Sparse autoencoders (SAEs)

    An SAE is a small network trained on the hidden states of a big model. It has two parts:

    - An **encoder** turns the hidden state into a much longer list of **features**.
    - A **decoder** turns the features back into the hidden state.

    Two rules shape the training. The rebuilt hidden state must match the original. And **almost all features must stay off** for any one input. That is the *sparse* part.
    """)
    return


@app.cell(hide_code=True)
def sae_diagram(mo, np):
    _rng = np.random.default_rng(3)
    _parts = []

    def _column(x, y0, n, w, h, gap):
        _shades = _rng.uniform(0.15, 0.85, n)
        return "".join(
            f"<rect x='{x}' y='{y0 + i * (h + gap):.1f}' width='{w}' height='{h}' rx='2' "
            f"fill='currentColor' fill-opacity='{_shades[i]:.2f}'/>"
            for i in range(n)
        )

    # Input column
    _parts.append(_column(50, 60, 14, 56, 11, 3))
    # Feature grid: 10 columns by 14 rows, 3 lit
    _lit = {(2, 3): "#d62728", (6, 7): "#2f6fdb", (8, 11): "#e8832a"}
    for _r in range(14):
        for _c in range(10):
            _x, _y = 300 + _c * 17, 60 + _r * 17
            if (_c, _r) in _lit:
                _parts.append(
                    f"<rect x='{_x}' y='{_y}' width='14' height='14' rx='2' fill='{_lit[(_c, _r)]}'/>"
                )
            else:
                _parts.append(
                    f"<rect x='{_x}' y='{_y}' width='14' height='14' rx='2' fill='none' "
                    "stroke='currentColor' stroke-opacity='0.25'/>"
                )
    # Output column
    _parts.append(_column(654, 60, 14, 56, 11, 3))

    def _arrow(x1, x2, y, label):
        return (
            f"<line x1='{x1}' y1='{y}' x2='{x2}' y2='{y}' stroke='currentColor' stroke-width='2' "
            "marker-end='url(#head)'/>"
            f"<text x='{(x1 + x2) / 2}' y='{y - 10}' text-anchor='middle' font-size='14' fill='currentColor'>{label}</text>"
        )

    _parts.append(_arrow(125, 285, 160, "Encoder"))
    _parts.append(_arrow(485, 640, 160, "Decoder"))

    def _label(x, y, lines):
        return "".join(
            f"<text x='{x}' y='{y + 17 * i}' text-anchor='middle' font-size='14' "
            f"font-weight='{600 if i == 0 else 400}' fill='currentColor'>{t}</text>"
            for i, t in enumerate(lines)
        )

    _parts.append(_label(78, 18, ["Hidden state", "5,120 numbers"]))
    _parts.append(_label(385, 18, ["Features", "many more, nearly all off"]))
    _parts.append(_label(682, 18, ["Rebuilt", "hidden state"]))
    _parts.append(
        "<text x='385' y='318' text-anchor='middle' font-size='13' fill='currentColor'>"
        "3 of 140 boxes shown on. Each lit box is a candidate idea.</text>"
    )
    mo.Html(
        "<svg viewBox='0 0 740 330' width='100%' style='max-width:740px;color:inherit' role='img' "
        "aria-label='Sparse autoencoder: hidden state, encoder, sparse features, decoder, rebuilt hidden state'>"
        "<defs><marker id='head' markerWidth='10' markerHeight='8' refX='9' refY='4' orient='auto'>"
        "<path d='M0,0 L10,4 L0,8 z' fill='currentColor'/></marker></defs>"
        + "".join(_parts)
        + "</svg>"
    )
    return


@app.cell(hide_code=True)
def sae_link(mo):
    mo.md(r"""
    Each lit feature is a **candidate** for one idea the model has learned. We still have to check what each one responds to.

    Arc Institute and Goodfire trained an SAE on **Evo 2** and built a viewer. Explore the features here: [Evo 2 mechanistic interpretability viewer](https://arcinstitute.org/tools/evo/evo-mech-interp).
    """)
    return


if __name__ == "__main__":
    app.run()
