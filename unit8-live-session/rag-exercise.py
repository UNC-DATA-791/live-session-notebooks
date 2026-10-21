# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ipython==9.17.1",  # transformers needs it when it runs inside marimo
#     "langchain-core==1.6.13",
#     "langchain-huggingface==1.2.3",
#     "langchain-text-splitters==1.1.3",
#     "marimo",
#     "polars==1.44.2",
#     "sentence-transformers[image]==6.2.0",
#     "torch==2.14.1",
#     "transformers==5.20.2",
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
    import polars as pl
    import torch
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import (
        ChatHuggingFace,
        HuggingFaceEmbeddings,
        HuggingFacePipeline,
    )
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    return (
        ChatHuggingFace,
        ChatPromptTemplate,
        HuggingFaceEmbeddings,
        HuggingFacePipeline,
        InMemoryVectorStore,
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
        RunnablePassthrough,
        StrOutputParser,
        pl,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ask a paper questions with RAG

    In the live session, we found the nearest text notes for a question, and gave them to an LLM. Today you do the same thing for one full paper: [*Active Learning Enables Efficient Directed Evolution of a Far-Red Fluorescent Protein with Minimal Experimental Data*](https://www.biorxiv.org/content/10.64898/2026.08.12.744534v1.full) (Brown et al., 2026). It uses active learning, the topic of this unit.

    You use [LangChain](https://python.langchain.com/) for each step:

    1. Get the paper as Markdown
    2. Split it into chunks
    3. Embed the chunks with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
    4. Find the chunks for a question
    5. Make a prompt from the question and the chunks
    6. Send the prompt to an LLM to get an answer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Turn on the GPU

    The embedding model is much faster on a GPU. In molab:

    1. Click the **notebook specs** button in the header (top right).
    2. Select the GPU.
    3. When the notebook restarts, run this cell again. It must show a green box.
    """)
    return


@app.cell
def _(mo, torch):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(
            mo.md("**No GPU found.** Turn on the GPU with the steps above."),
            kind="danger",
        ),
    )
    device = "cuda"
    mo.callout(
        mo.md(f"GPU found: **{torch.cuda.get_device_name(0)}**"),
        kind="success",
    )
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Get the paper as Markdown

    1. Download the PDF from [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.12.744534v1.full) (click **Download PDF**).
    2. Open [pdfdown](https://stephenturner.github.io/pdfdown/). Drop the PDF on the page, then click **Convert**. It runs in your browser, so the PDF does not leave your computer.
    3. Copy the Markdown, and paste it in the box below.
    """)
    return


@app.cell
def _(mo):
    paper_box = mo.ui.text_area(
        label="Paste the paper (Markdown)",
        placeholder="# Active Learning Enables Efficient Directed Evolution ...",
        full_width=True,
        rows=10,
    )
    paper_box
    return (paper_box,)


@app.cell
def _(mo, paper_box):
    mo.stop(not paper_box.value, mo.md("Paste the paper in the box above."))

    paper_md = paper_box.value
    print(f"{len(paper_md):,} characters")
    return (paper_md,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Split the paper into chunks

    One embedding for the full paper is too general. A question is about one part of the paper. Small chunks let the search find that part.

    Look at your Markdown first. pdfdown makes the section names (Introduction, Results, Methods, ...) into `######` headers. It also makes some figure labels into headers, like `# d`. Those are not sections, so split on `######` only.

    Split in two steps:

    1. Split at the section headers with `MarkdownHeaderTextSplitter`. Use `strip_headers=False`, so each chunk keeps its header text.
    2. Split long sections again with `RecursiveCharacterTextSplitter`. Use `chunk_size=1000` and `chunk_overlap=100`.

    Call the result `chunks`. Then show a table with the length of each chunk.

    <details>
    <summary>Hint</summary>

    ```python
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("######", "section")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(paper_md)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(sections)

    pl.DataFrame({
        "chunk": range(len(chunks)),
        "section": [c.metadata.get("section", "\") for c in chunks],
        "length": [len(c.page_content) for c in chunks],
        "start": [c.page_content[:80] for c in chunks],
    })
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: split the paper into sections, then into chunks. Call the result `chunks`
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Embed the chunks

    Make an embedding model with `HuggingFaceEmbeddings` and `google/embeddinggemma-2`.

    EmbeddingGemma 2 wants a short label in front of each text:

    - For chunks, use `prompt_name="Document"` in `encode_kwargs`
    - For questions, use `prompt_name="QuestionAnswering"` in `query_encode_kwargs`

    We only need text, so turn off the image and audio parts of the model with `config_kwargs`. Use `bfloat16` numbers. They make the model fast on the GPU. (Do not use `float16`. This model gives bad vectors with it.)

    Then put all the chunks in an `InMemoryVectorStore`. It does the same job as `NearestNeighbors` in the live session: it keeps the vectors and finds the closest ones.

    <details>
    <summary>Hint</summary>

    ```python
    embeddings = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-2",
        model_kwargs={
            "device": device,
            "model_kwargs": {"dtype": torch.bfloat16},
            "config_kwargs": {"vision_config": None, "audio_config": None},
        },
        encode_kwargs={"prompt_name": "Document"},
        query_encode_kwargs={"prompt_name": "QuestionAnswering"},
    )

    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: make the EmbeddingGemma 2 embeddings, then put the chunks in an InMemoryVectorStore called `vector_store`
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Find the chunks for a question

    Make a `retriever` from the vector store. Set it to return the 4 nearest chunks. Then use it to find the chunks for the question below, and show them in a table.

    <details>
    <summary>Hint</summary>

    ```python
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    found = retriever.invoke(question.value)

    pl.DataFrame({
        "section": [d.metadata.get("section", "\") for d in found],
        "text": [d.page_content for d in found],
    })
    ```

    </details>
    """)
    return


@app.cell
def _(mo):
    question = mo.ui.text(
        value="How did the batch size change how fast active learning found bright variants?",
        label="Question",
        full_width=True,
    )
    question
    return (question,)


@app.cell
def _():
    # TODO: make a `retriever` that returns 4 chunks, then show the chunks for the question
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Make the prompt

    Now join the steps into one LangChain pipeline (a "chain"). Its input is the question. Its output is a prompt that is ready for an LLM.

    1. Write a `ChatPromptTemplate` with two slots: `{context}` and `{question}`. Tell the LLM to use only the context. Also add the system message `/no_think`. It stops the LLM in step 6 from writing out long "thinking" before the answer.
    2. Write `format_docs(docs)`. It joins the text of the chunks into one string.
    3. Make the chain: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt`
    4. Call `chain.invoke(question.value)` and show the prompt as text.

    The `|` sends the output of one step into the next step.

    <details>
    <summary>Hint</summary>

    ```python
    prompt = ChatPromptTemplate.from_messages([
        ("system", "/no_think"),
        (
            "human",
            "Answer the question with only the context below. "
            "If the context does not have the answer, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}",
        ),
    ])


    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)


    chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt

    rag_prompt = chain.invoke(question.value).to_string()
    mo.md(f"```\n{rag_prompt}\n```")
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: make the prompt template, `format_docs` and the `chain`, then show the prompt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Get an answer

    Now add the "G" in RAG: generation. Load a small free LLM, [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), and add it to the end of your chain.

    1. Load the model with `HuggingFacePipeline.from_model_id`, then wrap it in `ChatHuggingFace`. Call it `llm`.
    2. Make a new chain: `chain | llm | StrOutputParser()`. The parser takes the text out of the LLM's reply.
    3. Call it with the question, and show the answer.

    Then read the chunks from step 4 again. Is the answer correct? Small LLMs can get facts wrong, even when the correct text is in the prompt.

    <details>
    <summary>Hint</summary>

    ```python
    llm = ChatHuggingFace(
        llm=HuggingFacePipeline.from_model_id(
            model_id="HuggingFaceTB/SmolLM3-3B",
            task="text-generation",
            device=0,
            model_kwargs={"dtype": torch.bfloat16},
            pipeline_kwargs={"max_new_tokens": 300, "do_sample": False, "return_full_text": False},
        )
    )
    ```

    ```python
    rag_chain = chain | llm | StrOutputParser()
    answer = rag_chain.invoke(question.value)
    mo.md(answer)
    ```

    </details>
    """)
    return


@app.cell
def _():
    # TODO: load SmolLM3-3B as `llm`
    return


@app.cell
def _():
    # TODO: add the LLM to the chain, then show the answer
    return



if __name__ == "__main__":
    app.run()
