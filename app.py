import streamlit as st
from src.ingestion import IngestionEngine


# 1. Initialize the Class (Cached so it doesn't reload every time)
@st.cache_resource
def get_ingestion_engine():
    return IngestionEngine()


engine = get_ingestion_engine()

st.title("🎓 AI Lecture Tutor")

# 2. Initialize the variable via the UI widget
with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    process_button = st.button("Process Files")

# 3. Use the variable (Only after it has been defined above)
if process_button:
    if uploaded_files:
        with st.spinner("Processing..."):
            chunks = engine.process_all_files(uploaded_files)
            # store for chat use
            st.session_state.chunks = chunks

            st.success(f"Done! Created {len(chunks)} text segments.")
    else:
        st.warning("Please upload files first.")