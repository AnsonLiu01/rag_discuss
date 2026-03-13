import streamlit as st

from src.ingestion import IngestionEngine
from src.vector_store import VectorStore
from src.llm_discuss import LLMDiscuss
from src.orchestrator import TutorOrchestrator


# --- 1. Resource Initialization (Cached) ---
@st.cache_resource
def init_engines():
    ingestion = IngestionEngine()
    vector_db = VectorStore()
    orchestrator = TutorOrchestrator()
    return ingestion, vector_db, orchestrator


ingestion_engine, vector_db, tutor = init_engines()

# --- 2. State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Anson's RAG-bot")

# --- 3. Sidebar: Upload & Processing ---
with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Files"):
        if uploaded_files:
            with st.status("Analyzing your lectures...", expanded=True) as status:
                st.write("Extracting text...")
                chunks = ingestion_engine.process_all_files(uploaded_files)

                st.write("Generating embeddings & indexing...")
                vector_db.add_documents(chunks)

                # We store chunks just in case, but the VectorDB is the real brain now
                st.session_state.chunks = chunks
                status.update(label="Indexing Complete!", state="complete", expanded=False)
            st.success(f"Added {len(chunks)} segments to the knowledge base.")
        else:
            st.warning("Please upload files first.")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your notes..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Response
    with st.chat_message("assistant"):
        # We call the orchestrator's streaming method
        stream, sources = tutor.ask(prompt)

        # Use Streamlit's built-in typewriter effect
        full_response = st.write_stream(stream)

        # Show citations
        if sources:
            with st.expander("📚 Sources used from your notes"):
                for i, source in enumerate(sources):
                    st.info(f"Source {i + 1}: {source[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": full_response})