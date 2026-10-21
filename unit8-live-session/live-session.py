# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "anthropic==1.8.0",
#     "marimo>=0.25.2",
#     "numpy==2.4.6",
#     "polars==1.42.1",
#     "pyarrow==25.0.0",
#     "remotezip==0.12.6",
#     "scikit-learn==1.9.0",
#     "scipy==1.18.0",
#     "sentence-transformers==5.6.0",
#     "torch==2.11.0",
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
    import anthropic
    import altair as alt
    import numpy as np
    import polars as pl
    import torch
    from remotezip import RemoteZip
    from scipy.stats import spearmanr
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    from transformers import AutoTokenizer, EsmModel, pipeline

    return (
        AutoTokenizer,
        EsmModel,
        NearestNeighbors,
        RemoteZip,
        SentenceTransformer,
        alt,
        anthropic,
        np,
        pipeline,
        pl,
        spearmanr,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline for this notebook

    ### What have we learned so far?
    - Unit 5: ESM2 embeddings. Average the residue vectors → one vector per protein
    - Unit 6: variant effects and [ProteinGym](https://proteingym.org/)
    - Unit 7: function by homology. Copy GO terms from the nearest neighbors

    ### Today: active learning
    - The lab can only test tens to hundreds of variants per round. The space of sequences is huge
    - The loop: train a model → score the untested variants → pick the best ones → test them in the lab → repeat
    - [EVOLVEpro](https://www.science.org/doi/10.1126/science.adr6006): ESM2 embeddings + a random forest, a few rounds at a time
    - Case study: [engineering the Fanzor MmeFz2 editor](https://www.nature.com/articles/s41467-026-68644-5)
    - Embeddings + nearest neighbors, again. The same idea is behind **RAG** (retrieval-augmented generation)

    ### What's next?
    - Exercise: build a small RAG search with [EmbeddingGemma 2](https://www.marktechpost.com/2026/10/06/google-deepmind-releases-embeddinggemma-2-a-740m-open-multimodal-embedding-model-built-on-gemma-4/)
    - Unit 9: Project 3, generative protein models

    ### Open questions
    - When does active learning beat random picks? When does it not?
    - EVOLVEpro found that PLM scores track lab activity poorly. Why might that be?
    - In RAG, what happens when retrieval finds the wrong text?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The active learning loop

    1. Train a model on the few variants we have labels for
    2. Score every untested variant
    3. Pick the next batch to test
    4. Test them in the lab, add the new labels, and repeat

    Directed evolution used to test random mutants. Active learning lets the model choose what to test next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.image(
                "https://dsgissin.github.io/DiscriminativeActiveLearning/assets/images/AL-Intro/blobs-AL.png",
                width=640,
            ),
            mo.md(
                "Source: [Gissin, *Introduction to Active Learning*](https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/AL-Intro.html)"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How to read the figure:

    - Two groups of points. Both rows start with one labeled point per group (**red**)
    - Each step adds one more labeled point (**green**), then trains a new line to split the groups
    - **Top, random**: the new points land anywhere. After 4 picks, the line is still wrong
    - **Bottom, active learning**: the model asks for the point it is least sure about, near the line. After 2 or 3 picks, the line is right

    Two ways to pick the next batch:

    - **Explore**: test what the model is least sure about (like the figure). Makes the model better
    - **Exploit**: test what the model scores highest. Finds a winner fast. EVOLVEpro mostly does this: each round it tests about 10 top predictions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get the variants

    The Unit 5 default ProteinGym data set: a deep mutational scan (DMS) with one fitness score per single mutant.
    """)
    return


@app.cell
def _(RemoteZip, pl):
    PROTEINGYM_SUBSTITUTIONS_URL = (
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
        "DMS_ProteinGym_substitutions.zip"
    )
    DMS_ID = "Q837P4_ENTFA_Meier_2023"

    with RemoteZip(PROTEINGYM_SUBSTITUTIONS_URL) as _zf:
        _match = next(n for n in _zf.namelist() if DMS_ID in n)
        with _zf.open(_match) as _f:
            df = pl.read_csv(_f)

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embed every variant with ESM2

    Same recipe as Unit 5: run the sequence through ESM2, then average the residue vectors. One vector per variant.
    """)
    return


@app.cell
def _(AutoTokenizer, EsmModel, torch):
    # use the bigger, better model when a GPU is around (e.g. on molab);
    # otherwise stick with the small one so this stays fast on a laptop
    model_name = (
        "facebook/esm2_t33_650M_UR50D"
        if torch.cuda.is_available()
        else "facebook/esm2_t6_8M_UR50D"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = EsmModel.from_pretrained(model_name).eval().to(device)
    model_name, device
    return device, esm, tokenizer


@app.cell
def _(device, df, esm, tokenizer, torch):
    def embed(sequences, batch_size=16):
        vectors = []
        for _i in range(0, len(sequences), batch_size):
            _batch = tokenizer(
                sequences[_i : _i + batch_size],
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                _hidden = esm(**_batch).last_hidden_state
            # average over real residues only: no padding, no <cls>/<eos>
            _mask = _batch["attention_mask"].clone()
            _mask[:, 0] = 0
            _mask[torch.arange(len(_mask)), _mask.sum(1)] = 0
            _mask = _mask.unsqueeze(-1)
            vectors.append(((_hidden * _mask).sum(1) / _mask.sum(1)).cpu())
        return torch.cat(vectors).numpy()

    embeddings = embed(df["mutated_sequence"].to_list())
    embeddings.shape
    return (embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One problem: every vector looks the same

    Each variant differs from the wild type by **one** amino acid out of 589. So the averaged vectors are almost identical, and every cosine distance is about 0.

    Fix: subtract the average vector. What is left is the part that makes each variant different.
    """)
    return


@app.cell
def _(NearestNeighbors, embeddings, np):
    _raw_dist = (
        NearestNeighbors(n_neighbors=2, metric="cosine")
        .fit(embeddings)
        .kneighbors(embeddings)[0][:, 1]
    )
    centered = embeddings - embeddings.mean(axis=0)
    print(
        f"median distance to nearest neighbor, raw: {np.median(_raw_dist):.1e}"
    )
    return (centered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nearest neighbors

    `NearestNeighbors` does not predict anything. It is an index: give it vectors, then ask "which stored vectors are closest to this one?"

    We use cosine distance. 0 means two vectors point the same way.
    """)
    return


@app.cell
def _(NearestNeighbors, centered):
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(centered)
    nn
    return (nn,)


@app.cell
def _(df, mo):
    variant_picker = mo.ui.dropdown(
        options=df["mutant"].to_list(),
        value=df["mutant"][0],
        label="Pick a variant",
        searchable=True,
    )
    variant_picker
    return (variant_picker,)


@app.cell
def _(centered, df, nn, pl, variant_picker):
    _i = df["mutant"].index_of(variant_picker.value)
    _dist, _idx = nn.kneighbors(centered[[_i]])

    # the first hit is the variant itself (distance 0)
    df[_idx[0]].select("mutant", "DMS_score").with_columns(
        cosine_distance=pl.Series(_dist[0]).round(3)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to look for:

    - Do the neighbors share a **position**? Or the same **new amino acid**?
    - Are the neighbors' fitness scores close to the picked variant's score?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Guess fitness from the neighbors

    Like Unit 7: copy the labels of the nearest neighbors. For each variant, hide its own score and guess the average score of its `k` neighbors.
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(
        1, 30, value=5, label="k (number of neighbors)", show_value=True
    )
    k_slider
    return (k_slider,)


@app.cell
def _(NearestNeighbors, centered, df, k_slider, mo, spearmanr):
    # ask for k + 1 neighbors, then drop the first one: it is the variant itself
    _, _idx = (
        NearestNeighbors(n_neighbors=k_slider.value + 1, metric="cosine")
        .fit(centered)
        .kneighbors(centered)
    )
    knn_df = df.select("mutant", "DMS_score").with_columns(
        knn_guess=df["DMS_score"].to_numpy()[_idx[:, 1:]].mean(axis=1)
    )
    knn_rho = spearmanr(knn_df["knn_guess"], knn_df["DMS_score"]).statistic
    mo.md(f"Spearman correlation between guess and truth: **{knn_rho:.2f}**")
    return (knn_df,)


@app.cell(hide_code=True)
def _(alt, k_slider, knn_df):
    alt.Chart(knn_df).mark_circle(opacity=0.5).encode(
        x=alt.X("DMS_score:Q", title="measured fitness"),
        y=alt.Y("knn_guess:Q", title=f"average of {k_slider.value} neighbors"),
        tooltip=[
            "mutant",
            alt.Tooltip("DMS_score:Q", format=".2f"),
            alt.Tooltip("knn_guess:Q", format=".2f"),
        ],
    ).properties(width=450, height=350, title="kNN guess vs. measured fitness")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## This is not unlike RAG

    An LLM only knows what was in its training data. It does not know your lab notebook, your course, or last week's paper. **Retrieval-augmented generation (RAG)** fixes this: find the right text first, then hand it to the LLM with the question.

    Look at the steps. We just did all of them with proteins.

    | Step | Protein variants (above) | RAG (below) |
    | --- | --- | --- |
    | Things we store | 697 variant sequences | short text notes |
    | Embedding model | ESM2 | `all-MiniLM-L6-v2` |
    | Index | `NearestNeighbors` | `NearestNeighbors` (big systems use a vector database) |
    | Query | a variant | a question |
    | Use the neighbors | average their fitness | paste them into the LLM's prompt |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A tiny RAG system

    ### 1. The documents

    Short notes about this course. A real system would split PDFs, papers or web pages into chunks like these.
    """)
    return


@app.cell
def _():
    notes = [
        "Unit 5 introduced protein language models. ESM2 turns each residue into a vector; averaging them gives one embedding per protein.",
        "Unit 5's exercise used the ProteinGym DMS data set Q837P4_ENTFA_Meier_2023, with 697 single mutants and a fitness score for each.",
        "Unit 6 scored variant effects zero-shot with ESM: log p(mutant) minus log p(wild type) at a masked position.",
        "Unit 6's exercise scored human missense variants in TP53 with ESM-1b and measured AUC against ClinVar labels from ProteinGym.",
        "ProteinGym is a benchmark with more than 250 deep mutational scans and clinical variant sets, used to compare more than 70 models.",
        "Unit 7 covered functional annotation: predicting a protein's Gene Ontology (GO) terms, as in the CAFA 6 competition.",
        "In Unit 7, a nearest-neighbor model copied GO terms from the most similar training proteins, and was scored with F-max.",
        "Unit 7 showed that splitting by protein superfamily gives a harder, more honest test than a random split.",
        "Unit 8 is about active learning: train a model, score untested variants, pick the next batch, test it in the lab, and repeat.",
        "In active learning, exploiting means testing the top-scored variants; exploring means testing variants the model is unsure about.",
        "EVOLVEpro (Science) feeds ESM2 embeddings into a random forest regressor and improves proteins in a few rounds of about 10 variants each.",
        "EVOLVEpro beat zero-shot protein language model scores on 12 DMS benchmarks and gave up to 100-fold gains on six proteins.",
        "A key EVOLVEpro finding: zero-shot protein language model scores correlate poorly with measured activity.",
        "The Fanzor case study engineered the MmeFz2 editor. AlphaFold3 guided a new omega-RNA scaffold, and EVOLVEpro found protein mutations.",
        "The engineered Fanzor MmeFz2 system raised editing efficiency more than 70-fold and fits in a single AAV vector.",
        "In a mouse model of Duchenne muscular dystrophy, the engineered Fanzor editor restored dystrophin.",
        "EmbeddingGemma 2 is an open embedding model from Google DeepMind, released in October 2026, with 768-dimensional vectors.",
        "This week's exercise uses EmbeddingGemma 2 to build a small RAG search.",
        "Unit 9 starts Project 3, on generative protein models.",
        "molab gives free notebooks; you can attach a GPU from the notebook specs button in the header.",
    ]
    len(notes)
    return (notes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Embed the notes and build the index

    Same two lines as before: embed, then `NearestNeighbors`. Only the embedding model changed.
    """)
    return


@app.cell
def _(NearestNeighbors, SentenceTransformer, device, notes):
    text_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    note_embeddings = text_model.encode(notes)
    note_nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(
        note_embeddings
    )
    note_embeddings.shape
    return note_nn, text_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Retrieve

    Embed the question, then find the 3 closest notes.
    """)
    return


@app.cell
def _(mo):
    question = mo.ui.text(
        value="How much did EVOLVEpro improve the Fanzor editor?",
        label="Ask a question",
        full_width=True,
    )
    question
    return (question,)


@app.cell
def _(note_nn, notes, pl, question, text_model):
    _dist, _idx = note_nn.kneighbors(text_model.encode([question.value]))
    retrieved = [notes[_i] for _i in _idx[0]]
    pl.DataFrame({"note": retrieved, "cosine_distance": _dist[0].round(3)})
    return (retrieved,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Generate

    Paste the retrieved notes into the prompt, then ask an LLM.

    - Paste an Anthropic API key below → the notebook uses **Claude**
    - Leave it empty → the notebook downloads a small free model (**SmolLM3-3B**) and runs it here

    The key stays in this box. It is not saved in the notebook file.
    """)
    return


@app.cell
def _(mo):
    api_key = mo.ui.text(kind="password", label="Anthropic API key (optional)")
    api_key
    return (api_key,)


@app.cell
def _(anthropic, api_key, device, mo, pipeline):
    CLAUDE_MODEL = "claude-haiku-5-5"
    LOCAL_MODEL = "HuggingFaceTB/SmolLM3-3B"

    @mo.cache
    def load_local_llm():
        return pipeline(
            "text-generation", model=LOCAL_MODEL, device=device, dtype="auto"
        )

    def generate(prompt):
        """Return (model name, answer). Claude if there is a key, else the local model."""
        if api_key.value:
            _client = anthropic.Anthropic(api_key=api_key.value)
            _msg = _client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return CLAUDE_MODEL, _msg.content[0].text
        # "/no_think" turns off SmolLM3's long thinking step
        _messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt},
        ]
        _out = load_local_llm()(_messages, max_new_tokens=300, do_sample=False)
        return LOCAL_MODEL, _out[0]["generated_text"][-1]["content"]

    return (generate,)


@app.cell
def _(mo):
    use_notes = mo.ui.switch(
        value=True, label="Give the LLM the retrieved notes"
    )
    ask = mo.ui.run_button(label="Ask the LLM")
    mo.hstack([use_notes, ask], justify="start")
    return ask, use_notes


@app.cell
def _(mo, question, retrieved, use_notes):
    if use_notes.value:
        _context = "\n".join(f"- {_n}" for _n in retrieved)
        rag_prompt = (
            "Answer the question using only the notes below. "
            "If the notes do not have the answer, say so.\n\n"
            f"Notes:\n{_context}\n\n"
            f"Question: {question.value}"
        )
    else:
        rag_prompt = question.value

    mo.md(f"The prompt we send:\n\n```\n{rag_prompt}\n```")
    return (rag_prompt,)


@app.cell
def _(ask, generate, mo, rag_prompt):
    mo.stop(not ask.value, mo.md("Click **Ask the LLM** to get an answer."))

    _model, _answer = generate(rag_prompt)
    mo.md(f"**{_model}** says:\n\n{_answer}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try this:

    - Ask with the small local model, then with Claude. Check each answer against the notes. Small models can mix up facts even when the right note is in front of them
    - Turn the notes **off** and ask again. Does the LLM know about Fanzor? Does it make something up?
    - Ask something the notes do not cover. Does the LLM say so?
    - Ask *"What did we do in Unit 7?"* Which notes come back?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## This week: RAG with EmbeddingGemma 2

    In the exercise you build the same pipeline with a stronger embedding model, [EmbeddingGemma 2](https://www.marktechpost.com/2026/10/06/google-deepmind-releases-embeddinggemma-2-a-740m-open-multimodal-embedding-model-built-on-gemma-4/). Keep these questions in mind:

    1. Does a better embedding model find better notes?
    2. How many notes should you retrieve? What goes wrong with too few, or too many?
    3. Where else in this course could "embed → nearest neighbors" help?
    """)
    return


if __name__ == "__main__":
    app.run()
