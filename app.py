import re
import streamlit as st
from groq import RateLimitError
import config
from query.answer import answer_stream, cited_indices
from query.retrieve import retrieve, _dense, _sparse, _client
from query.rewrite import rewrite_query

REFUSAL = "I couldn't find this in the indexed code"

# Matches inline code (left alone) or a citation like [1], 【1】 or 【1†L5-L7】
CITE = re.compile(r"(`[^`\n]*`)|[\[【](\d+)(?:†[^\]】]*)?[\]】]")


@st.cache_resource(show_spinner="Loading models (first run only)...")
def warm_up():
    _dense()
    _sparse()
    _client()
    return True


def linkify(text: str, chunks: list[dict]) -> str:
    """Turn [n] citations into links to the commit-pinned GitHub lines.
    Fenced code blocks are skipped; odd segments after splitting on ``` are code."""
    def sub(m):
        if m.group(1):
            return m.group(1)
        n = int(m.group(2))
        if 1 <= n <= len(chunks):
            return f"[\\[{n}\\]]({chunks[n - 1]['url']})"
        return m.group(0)

    segments = text.split("```")
    for i in range(0, len(segments), 2):
        segments[i] = CITE.sub(sub, segments[i])
    return "```".join(segments)


def render_sources(answer: str, chunks: list[dict]):
    if REFUSAL.lower() in answer.lower():
        return                                  # nothing to cite for a refusal
    used = cited_indices(answer, len(chunks))
    if not used:
        st.caption("The model cited no excerpts.")
        return
    st.markdown("**Sources**")
    for i in used:
        c = chunks[i - 1]
        st.markdown(f"**[{i}]** [`{c['path']}:{c['start_line']}-{c['end_line']}`]({c['url']})"
                    f" · {c['symbol']}")


def render_chunks(chunks: list[dict]):
    with st.expander(f"Retrieved excerpts ({len(chunks)})"):
        st.caption("Scores are rank-fusion values, not similarities, so only the order is meaningful.")
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**[{i}]** [`{c['path']}:{c['start_line']}-{c['end_line']}`]({c['url']})"
                        f" · {c['kind']}: `{c['symbol']}` · score {c['score']:.3f}")
            st.code(c["text"][:1500], language="python" if c["language"] == "python" else None)


def render_entry(e: dict):
    with st.chat_message("user"):
        st.markdown(e["question"])
    with st.chat_message("assistant"):
        if e["rewrites"]:
            st.caption("Also searched for: " + " · ".join(e["rewrites"]))
        st.markdown(linkify(e["answer"], e["chunks"]))
        render_sources(e["answer"], e["chunks"])
        render_chunks(e["chunks"])


def answer_question(question: str, repo: str, use_rewrite: bool):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the code..."):
                chunks = retrieve(question, k=config.TOP_K, repo=repo, hybrid=True,
                                  rewrite=use_rewrite, rerank=False)
                rewrites = rewrite_query(question) if use_rewrite else []
        except Exception:
            st.error("I couldn't reach the search index. Please try again in a moment.")
            return
        if rewrites:
            st.caption("Also searched for: " + " · ".join(rewrites))

        box, text = st.empty(), ""
        try:
            for token in answer_stream(question, chunks):
                text += token
                box.markdown(linkify(text, chunks) + " ▌")
        except RateLimitError as e:
            if "tokens per day" in str(e):
                st.error("The demo's daily AI quota is used up. Please try again tomorrow.")
            else:
                st.error("The AI service is busy. Please wait a few seconds and try again.")
            if not text:
                return
        except Exception:
            st.error("Something went wrong while generating the answer.")
            if not text:
                return

        box.markdown(linkify(text, chunks))
        render_sources(text, chunks)
        render_chunks(chunks)

    st.session_state.history.append(
        {"question": question, "answer": text, "chunks": chunks, "rewrites": rewrites})


# ---------------------------------------------------------------- page
st.set_page_config(page_title="repo-whisperer", page_icon="🔎", layout="wide")
st.session_state.setdefault("history", [])
warm_up()

with st.sidebar:
    st.header("Settings")
    repo = st.selectbox("Repository", config.REPOS)
    use_rewrite = st.toggle(
        "Query rewriting", value=True,
        help="An LLM adds search queries using code vocabulary. Helps when your wording "
             "differs from the code's, but adds 1-2 seconds.")
    left = config.APP_MAX_QUESTIONS - len(st.session_state.history)
    st.caption(f"Questions left in this session: {max(left, 0)}")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption("Hybrid search (dense + BM25) over AST-chunked code on Qdrant. "
               f"Answers by {config.GROQ_MODEL} on Groq, citing exact lines at a pinned commit.")

st.title("🔎 repo-whisperer")
st.caption("Ask questions about a codebase and get answers that cite the exact lines.")

question = st.chat_input("Ask about the code, e.g. how are blueprints registered?")
question = question or st.session_state.pop("pending", None)

for entry in st.session_state.history:
    render_entry(entry)

if question:
    question = question.strip()[: config.APP_MAX_QUESTION_CHARS]
    if len(st.session_state.history) >= config.APP_MAX_QUESTIONS:
        st.warning("You've reached the question limit for this session. "
                   "Use 'Clear conversation' in the sidebar to start over.")
    else:
        answer_question(question, repo, use_rewrite)
elif not st.session_state.history:
    st.markdown("**Try one:**")
    cols = st.columns(2)
    for i, q in enumerate(config.SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"sample{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()