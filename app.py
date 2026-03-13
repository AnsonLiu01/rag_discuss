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
        /* Main background - slightly warmer */
        .stApp {
            background-color: #f6f8fa;
        }

        /* ----------------------------------------------------------- */
        /* Modernized Chat "Islands" - Slimmer and Iconless */
        /* ----------------------------------------------------------- */

        [data-testid="stChatMessage"] {
            background-color: white !important;
            border-radius: 12px !important;
            padding: 16px !important; /* Reduced padding from 24px for slimmer feel */
            margin-bottom: 8px !important; /* SIGNIFICANTLY reduced margin between islands */
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
            border: 1px solid #edf2f7 !important;
        }

        /* Remove the default human/bot avatar images */
        [data-testid="stChatMessageAvatar"] {
            display: none !important;
        }

        /* Modern Text-Based Identity Labels */
        .sender-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
            color: #718096;
            margin-bottom: 4px; /* Space between label and content */
            display: flex;
            align-items: center;
        }

        /* Gradient marker next to the label */
        .label-marker {
            width: 4px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }

        /* Assistant Specific (Marker Color: Blue) */
        [data-testid="stChatMessage"][data-test-role="assistant"] .sender-label .label-marker {
            background: linear_gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        /* User Specific (Marker Color: Dark Grey/Black) */
        [data-testid="stChatMessage"][data-test-role="user"] .sender-label .label-marker {
            background: linear_gradient(135deg, #1a202c 0%, #4a5568 100%);
        }

        /* ----------------------------------------------------------- */
        /* Other Modern Touches */
        /* ----------------------------------------------------------- */

        /* Block container spacing */
        .block-container {
            padding-top: 2rem !important;
        }

        /* Sidebar modernization */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0;
        }

        h1 {
            font-weight: 900 !important;
            letter-spacing: -1px;
            color: #1a202c;
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

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Anson's RAG-bot")
st.markdown("---")

# --- 4. Sidebar: Upload & Processing ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload attachments to vectorise (PDF)",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🚀 Process & Index Files", use_container_width=True):
        if uploaded_files:
            with st.status("Indexing...", expanded=False):
                # 1. Get chunks and the new metadata tags
                chunks, metadatas = ingestion_engine.process_all_files(uploaded_files)

                # 2. Add BOTH to the vector store
                vector_db.add_documents(chunks, metadata_list=metadatas)

            st.success("Indexed successfully!")
            st.rerun()  # This forces the list below to update immediately

    st.markdown("---")

    with st.expander("🗄️ Manage Database"):
        # This line runs every time the expander is toggled or the app reruns
        current_sources = vector_db.get_all_sources()

        if not current_sources:
            st.info("Your brain is currently empty. Upload some PDFs to start.")
        else:
            st.write(f"**{len(current_sources)} Files in Brain**")

            # Use a selectbox or multiselect to choose what to kill
            to_delete = st.multiselect("Select files to remove:", current_sources)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Delete Selected", type="primary"):
                    for source in to_delete:
                        vector_db.delete_by_source(source)
                    st.rerun()
            with col2:
                if st.button("Delete All"):
                    vector_db.delete_all()
                    st.rerun()

    st.markdown("---")
    with st.expander("Show Raw Vector Store"):
        raw_data = vector_db.collection.get(include=['documents', 'metadatas'])

        if raw_data['ids']:
            debug_table = []
            for i in range(len(raw_data['ids'])):
                debug_table.append({
                    "ID": raw_data['ids'][i],
                    "Source": raw_data['metadatas'][i].get('source', 'Unknown'),
                    "Content Preview": raw_data['documents'][i]
                })

            st.dataframe(
                pd.DataFrame(debug_table, columns=["ID", "Source", "Content Preview"]),
                column_config={
                    "Full Content": st.column_config.TextColumn(
                        "Full Content",
                        width="large",  # Gives it more horizontal room
                        help="Full text extracted from the PDF chunk"
                    ),
                    "Source": st.column_config.TextColumn("Source", width="small")
                },
                use_container_width=True,
                hide_index=True
            )

            st.caption(f"Total chunks in brain: {len(raw_data['ids'])}")

            if st.checkbox("View as Text List"):
                for i, doc in enumerate(raw_data['documents']):
                    st.text_area(f"Chunk {i + 1} (from {raw_data['metadatas'][i].get('source')})",
                                 value=doc, height=200)
        else:
            st.info("Database is empty.")

for message in st.session_state.messages:
    # Use standard role identifier but apply custom labeling via HTML injection
    role = message["role"]
    with st.chat_message(role):
        if role == "user":
            label = "YOU"
        else:
            label = "RAG-BOT"

        # Inject modern label with subtle gradient marker
        st.markdown(f'<div class="sender-label"><div class="label-marker"></div>{label}</div>', unsafe_allow_html=True)
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your notes..."):
    # Show user message with label
    with st.chat_message("user"):
        st.markdown(f'<div class="sender-label"><div class="label-marker"></div>YOU</div>', unsafe_allow_html=True)
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Response
    with st.chat_message("assistant"):
        stream, sources = tutor.ask(prompt)

        # Apply label BEFORE the stream
        st.markdown(f'<div class="sender-label"><div class="label-marker"></div>RAG-BOT</div>', unsafe_allow_html=True)

        full_response = st.write_stream(stream)

        # Slimmed down source expander
        if sources:
            with st.expander("🔍 Citations"):
                for i, source in enumerate(sources):
                    st.caption(f"**Snippet {i + 1}**")
                    st.info(f"{source[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": full_response})