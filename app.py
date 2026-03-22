import html
import pandas as pd
import streamlit as st

from src.css.ui_custom_css import apply_custom_css
from src.ingestion import IngestionEngine
from src.orchestrator import TutorOrchestrator

# SETTINGS
is_local = True

st.set_page_config(page_title="Anson's RAG-bot", layout="wide")

apply_custom_css()


@st.cache_resource
def init_engines():
    ingestion = IngestionEngine()
    orchestrator = TutorOrchestrator(is_local=is_local)
    vector_db = orchestrator.vector_db
    return ingestion, vector_db, orchestrator


ingestion_engine, vector_db, tutor = init_engines()

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Top Header ---
st.title("Anson's RAG-bot")
st.markdown("---")

# --- Sidebar: Upload & Processing ---
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload attachments to vectorise (PDF)",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🚀 Process & Index Files", use_container_width=True):
        if uploaded_files:
            with st.status("Indexing...", expanded=False):
                chunks, metadatas = ingestion_engine.process_all_files(uploaded_files)
                vector_db.add_documents(chunks, metadata_list=metadatas)
            st.success("Indexed successfully!")
            st.rerun()

    st.markdown("---")
    st.header("Tools")

    with st.expander("🗄️ Manage Database"):
        current_sources = vector_db.get_all_sources()
        if not current_sources:
            st.info("Your brain is currently empty.")
        else:
            st.write(f"**Files in Brain: {len(current_sources)}**")
            to_delete = st.multiselect("Select files to remove:", current_sources)

            c1, c2 = st.columns(2)
            if c1.button("🗑️ Delete Selected", type="primary"):
                for s in to_delete:
                    vector_db.delete_by_source(s)
                st.rerun()
            if c2.button("Delete All"):
                vector_db.delete_all()
                st.rerun()

    with st.expander("🔬 Inspect Sources"):
        st.caption("Paste text from RAG-BOT to trace its source")

        target_text = st.text_area(
            "Text to analyze",
            height=120,
            placeholder="Paste text here...",
            key="analysis_input",
            label_visibility="collapsed"
        )

        if target_text.strip():
            with st.spinner("Searching..."):
                results = vector_db.collection.query(
                    query_texts=[target_text],
                    n_results=1,
                    include=['documents', 'metadatas', 'distances']
                )

                if results['ids'] and results['ids'][0]:
                    distance = results['distances'][0][0]
                    confidence = max(0, 100 - (distance * 100))

                    # Confidence badge
                    if confidence >= 80:
                        badge_color = "#d1fae5"
                        text_color = "#065f46"
                    elif confidence >= 50:
                        badge_color = "#fef3c7"
                        text_color = "#92400e"
                    else:
                        badge_color = "#fee2e2"
                        text_color = "#991b1b"

                    st.markdown(f"""
                        <div class="confidence-badge" style="background: {badge_color}; color: {text_color};">
                            {confidence:.1f}% Match
                        </div>
                    """, unsafe_allow_html=True)

                    doc_id = results['ids'][0][0]
                    source_name = results['metadatas'][0][0].get('source', 'Unknown')
                    content = results['documents'][0][0]

                    escaped_content = html.escape(content)

                    st.write(f"**Vector ID:\n{doc_id}**")
                    st.write(f"**Source File:\n{source_name}**\n")
                    st.caption(f"**Content:**\n{escaped_content}")
                else:
                    st.warning("No match found")

    st.markdown("---")
    st.header("Debugs")

    with st.expander("Show Raw Vector Store"):
        raw_data = vector_db.collection.get(include=['documents', 'metadatas'])
        if raw_data['ids']:
            debug_table = [{"ID": raw_data['ids'][i], "Source": raw_data['metadatas'][i].get('source', 'Unknown'),
                            "Content Preview": raw_data['documents'][i]} for i in range(len(raw_data['ids']))]
            st.dataframe(pd.DataFrame(debug_table), use_container_width=True, hide_index=True)
        else:
            st.info("Database is empty.")

        st.caption(f"Total chunks in brain: {len(raw_data['ids'])}")

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        label = "YOU" if message["role"] == "user" else "RAG-BOT"
        st.markdown(
            f'<div class="sender-label"><div class="label-marker"></div>{label}</div>',
            unsafe_allow_html=True
        )
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your notes..."):
    with st.chat_message("user"):
        st.markdown(
            '<div class="sender-label"><div class="label-marker"></div>YOU</div>',
            unsafe_allow_html=True
        )
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.markdown(
            '<div class="sender-label"><div class="label-marker"></div>RAG-BOT</div>',
            unsafe_allow_html=True
        )

        stream, sources = tutor.ask(prompt)
        full_response = st.write_stream(stream)

        if sources:
            with st.expander("🔍 Citations"):
                for src in sources:
                    st.info(src)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()