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
            with st.status("Building local brain...", expanded=True) as status:
                st.write("Reading PDFs...")
                chunks = ingestion_engine.process_all_files(uploaded_files)

                st.write("Creating vector embeddings...")
                vector_db.add_documents(chunks)

                st.session_state.chunks = chunks
                status.update(label="Index Ready!", state="complete", expanded=False)
            st.success(f"Indexed {len(chunks)} lecture segments.")
        else:
            st.warning("Please upload files first.")


# *** UPDATED: Modifying the Chat Loop for Custom Labels ***
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