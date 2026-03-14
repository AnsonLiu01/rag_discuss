import html
import pandas as pd
import streamlit as st

from src.ingestion import IngestionEngine
from src.vector_store import VectorStore
from src.llm_discuss import LLMDiscuss
from src.orchestrator import TutorOrchestrator

st.set_page_config(page_title="Anson's RAG-bot", layout="wide")


def apply_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #f6f8fa; }

        [data-testid="stChatMessage"] {
            background-color: white !important;
            border-radius: 12px !important;
            padding: 16px !important;
            margin-bottom: 8px !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
            border: 1px solid #edf2f7 !important;
        }

        [data-testid="stChatMessageAvatar"] { display: none !important; }

        .sender-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
            color: #718096;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
        }

        .label-marker {
            width: 4px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }

        [data-testid="stChatMessage"][data-test-role="assistant"] .sender-label .label-marker {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        [data-testid="stChatMessage"][data-test-role="user"] .sender-label .label-marker {
            background: linear-gradient(135deg, #1a202c 0%, #4a5568 100%);
        }

        .block-container { padding-top: 2rem !important; }

        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0;
        }

        h1 { font-weight: 900 !important; letter-spacing: -1px; color: #1a202c; }

        /* Source Inspector Styling */
        .confidence-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
            margin: 12px 0;
        }

        .confidence-high {
            background: #d1fae5;
            color: #065f46;
        }

        .confidence-medium {
            background: #fef3c7;
            color: #92400e;
        }

        .confidence-low {
            background: #fee2e2;
            color: #991b1b;
        }

        .source-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            margin-top: 12px;
        }

        .source-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .source-name {
            font-size: 11px;
            font-weight: 600;
            color: #334155;
            font-family: 'Monaco', 'Courier New', monospace;
            background: #ffffff;
            padding: 6px 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            word-break: break-all;
        }

        .content-preview {
            background: #ffffff;
            border-left: 3px solid #667eea;
            padding: 10px;
            border-radius: 4px;
            font-size: 11px;
            line-height: 1.5;
            color: #475569;
            max-height: 150px;
            overflow-y: auto;
        }

        .empty-state {
            text-align: center;
            padding: 20px;
            color: #94a3b8;
        }

        .empty-icon {
            font-size: 32px;
            margin-bottom: 8px;
            opacity: 0.5;
        }

        .empty-text {
            font-size: 11px;
            line-height: 1.4;
        }
        </style>
    """, unsafe_allow_html=True)


apply_custom_css()


@st.cache_resource
def init_engines():
    ingestion = IngestionEngine()
    orchestrator = TutorOrchestrator()
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