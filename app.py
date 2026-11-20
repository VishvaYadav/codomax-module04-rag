"""
📚 Chat With Your Documents: a RAG app (Codomax Internship, Module 04)

Document → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer
Run locally:  streamlit run app.py
"""
import hashlib
from pathlib import Path

import streamlit as st

from llm import Gemini, LLMError, friendly_error, get_api_key
from rag import (SYSTEM_PROMPT, VectorStore, build_prompt, chunk_documents,
                 load_document, retrieval_query)

st.set_page_config(page_title="Chat With Your Documents", page_icon="📚", layout="wide")

SAMPLE = Path(__file__).parent / "sample_docs" / "nimbus_labs_handbook.md"
MAX_FILES, MAX_CHUNKS = 5, 600          # protect the free API quota
SAMPLE_QUESTIONS = [
    "How many days of paid leave do employees get?",
    "What is the meal allowance when travelling abroad?",
    "Within how many hours must a security incident be reported, and to whom?",
    "What is the company's stock price?",   # not in the document -> should say so
]

ss = st.session_state
ss.setdefault("store", None)       # VectorStore
ss.setdefault("index_key", None)   # which files/settings the index was built from
ss.setdefault("doc_stats", {})
ss.setdefault("messages", [])


def show_error(e: Exception):
    err = friendly_error(e)
    st.error(err.user_message)
    if getattr(err, "detail", ""):
        with st.expander("Technical details"):
            st.code(err.detail)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> Gemini:
    return Gemini(api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def model_lists(api_key: str) -> tuple[list[str], list[str]]:
    g = Gemini(api_key)
    try:
        return g.chat_models(), g.embedding_models()
    except Exception:
        from llm import FALLBACK_CHAT_MODELS, FALLBACK_EMBED_MODELS
        return FALLBACK_CHAT_MODELS, FALLBACK_EMBED_MODELS


# ============================================================================ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_api_key() or st.text_input("Gemini API key", type="password",
                                             help="Free key from aistudio.google.com")
    chat_models, embed_models = model_lists(api_key) if api_key else ([], [])
    chat_model = st.selectbox("Chat model", chat_models or ["(add API key)"])
    embed_model = st.selectbox("Embedding model", embed_models or ["(add API key)"],
                               help="Turns text into vectors (lists of numbers) that capture meaning.")

    st.subheader("✂️ Chunking")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, 100,
                           help="Small chunks = precise matches; large chunks = more context per match.")
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25,
                        help="Repeats the end of each chunk at the start of the next, so facts on a boundary aren't lost.")

    st.subheader("🔎 Retrieval")
    top_k = st.slider("Top-k chunks", 1, 10, 4, help="How many chunks are sent to the LLM as context.")
    min_score = st.slider("Minimum similarity", 0.0, 1.0, 0.3, 0.05,
                          help="Ignore chunks that are not similar enough to the question.")
    use_mmr = st.toggle("Diverse results (MMR)", value=True,
                        help="Maximal Marginal Relevance: avoids sending near-duplicate chunks.")
    compare = st.toggle("Also answer WITHOUT documents", value=False,
                        help="Shows what the LLM says with no retrieval, to spot hallucinations.")

    st.divider()
    if st.button("🗑️ Clear chat", width="stretch"):
        ss.messages = []
        st.rerun()

# ============================================================================ header
st.title("📚 Chat With Your Documents")
st.caption("Retrieval-Augmented Generation (RAG) with Google Gemini · Codomax GenAI Internship · Module 04")
st.markdown("**Pipeline:** 📄 Document → ✂️ Chunking → 🔢 Embeddings → 🗄️ Vector DB → 🔎 Retrieval → 🤖 LLM → 💬 Answer")

llm = None
if api_key:
    try:
        llm = get_llm(api_key)
    except Exception as e:
        show_error(e)
else:
    st.info("👈 Add your free Gemini API key in the sidebar to start.")

# ============================================================================ documents
c1, c2 = st.columns([3, 1])
with c1:
    files = st.file_uploader("Upload PDF, TXT or Markdown files", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
with c2:
    st.write("")
    use_sample = st.toggle("Use sample document", value=not files,
                           help="A fictional company handbook. An LLM can't know it, so it shows RAG clearly.")

sources: dict[str, bytes] = {}
if use_sample and SAMPLE.exists():
    sources[SAMPLE.name] = SAMPLE.read_bytes()
for f in (files or [])[:MAX_FILES]:
    sources[f.name] = f.getvalue()
if files and len(files) > MAX_FILES:
    st.warning(f"Only the first {MAX_FILES} files are used.")

key_src = "|".join(f"{n}:{hashlib.md5(b).hexdigest()}" for n, b in sorted(sources.items()))
index_key = f"{key_src}|{chunk_size}|{overlap}|{embed_model}"
needs_index = bool(sources) and ss.index_key != index_key

if sources and llm:
    if st.button("⚡ Build index" if needs_index else "✅ Index ready (rebuild)",
                 type="primary" if needs_index else "secondary"):
        try:
            with st.status("Building the knowledge base...", expanded=True) as status:
                st.write("📄 Reading documents...")
                docs = {name: load_document(name, data) for name, data in sources.items()}
                pages = sum(len(p) for p in docs.values())
                st.write(f"✂️ Chunking {pages} page(s)...")
                chunks = chunk_documents(docs, chunk_size, overlap)
                if not chunks:
                    raise LLMError("No text found. Scanned PDFs (images) have no extractable text.")
                if len(chunks) > MAX_CHUNKS:
                    st.write(f"⚠️ {len(chunks)} chunks; keeping the first {MAX_CHUNKS} to save free quota.")
                    chunks = chunks[:MAX_CHUNKS]
                st.write(f"🔢 Creating embeddings for {len(chunks)} chunks with `{embed_model}`...")
                vectors = llm.embed([c.text for c in chunks], embed_model, "RETRIEVAL_DOCUMENT")
                store = VectorStore()
                store.add(chunks, vectors)
                ss.store, ss.index_key = store, index_key
                ss.doc_stats = {"Documents": len(docs), "Pages": pages, "Chunks": len(chunks),
                                "Vector size": store.dim}
                status.update(label="✅ Knowledge base ready!", state="complete", expanded=False)
            st.rerun()
        except Exception as e:
            show_error(e)
elif not sources:
    st.info("Upload a document or turn on **Use sample document**.")

if ss.store is not None and ss.index_key == index_key:
    cols = st.columns(len(ss.doc_stats))
    for col, (k, v) in zip(cols, ss.doc_stats.items()):
        col.metric(k, v)
    with st.expander("👀 Peek at the chunks"):
        for c in ss.store.chunks[:50]:
            st.markdown(f"**Chunk {c.id}** · {c.label} · {len(c.text)} chars")
            st.text(c.text[:600])
elif ss.store is not None and sources:
    st.warning("Documents or chunk settings changed. Click **Build index** to update.")

# ============================================================================ chat
ready = llm is not None and ss.store is not None and ss.index_key == index_key


def render_sources(retrieved):
    with st.expander(f"📚 Sources ({len(retrieved)} chunks retrieved)"):
        if not retrieved:
            st.caption("No chunk was similar enough. Try lowering **Minimum similarity**.")
        for i, r in enumerate(retrieved, 1):
            st.markdown(f"**[{i}] {r['label']}** · similarity `{r['score']:.3f}`")
            st.text(r["text"][:700])


for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("no_rag"):
            with st.expander("🧪 Answer WITHOUT documents (hallucination check)"):
                st.markdown(m["no_rag"])
        if m.get("sources") is not None:
            render_sources(m["sources"])
        if m.get("meta"):
            st.caption(m["meta"])

if ready and not ss.messages:
    st.markdown("##### Try asking:" if use_sample else "##### Ask anything about your documents")
    if use_sample:
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, q in zip(cols, SAMPLE_QUESTIONS):
            if col.button(q, width="stretch"):
                ss.pending = q
                st.rerun()

question = st.chat_input("Ask a question about your documents...", disabled=not ready) \
    or ss.pop("pending", None)

if question and ready:
    history = [m for m in ss.messages if m["role"] in ("user", "assistant")]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("🔎 Retrieving relevant chunks..."):
                q_vec = llm.embed([retrieval_query(question, history)], embed_model, "RETRIEVAL_QUERY")[0]
                hits = ss.store.search(q_vec, k=top_k, min_score=min_score, mmr=use_mmr)
            with st.spinner("🤖 Generating answer..."):
                answer, usage = llm.generate(build_prompt(question, hits, history), chat_model,
                                             system=SYSTEM_PROMPT, temperature=0.2)
                no_rag = None
                if compare:
                    no_rag, _ = llm.generate(question, chat_model, temperature=0.2)
            sources_data = [{"label": c.label, "score": s, "text": c.text} for c, s in hits]
            meta = (f"🔢 {usage.input_tokens} in · {usage.output_tokens} out · ⏱️ {usage.latency_s}s · "
                    f"top score {max((s for _, s in hits), default=0):.3f}")
            ss.messages.append({"role": "assistant", "content": answer or "_(empty answer)_",
                                "sources": sources_data, "no_rag": no_rag, "meta": meta})
            st.rerun()
        except Exception as e:
            ss.messages.pop()
            show_error(e)
