import streamlit as st
import os
import requests

API_URL = "http://127.0.0.1:8000"  # FastAPI backend

st.set_page_config(page_title="RAG Pipeline Demo", layout="wide")
st.title("RAG Data Pipeline Demo")
st.write("This app demonstrates your RAG pipeline powered by a FastAPI backend.")

# Directory selection
st.header("Step 1: Data Directory Selection")
data_dir = st.text_input("Enter your data directory path:", value="data")
if not os.path.isdir(data_dir):
    st.warning(f"Directory '{data_dir}' does not exist on Streamlit host. Backend will validate path.")
else:
    st.success(f"Using data directory: {data_dir}")

# Run backend pipeline
if st.button("Run Pipeline on Directory"):
    with st.spinner("Running pipeline..."):
        resp = requests.post(f"{API_URL}/run_pipeline", json={"data_dir": data_dir})
        if resp.status_code == 200:
            stats = resp.json()
            st.success("Pipeline completed successfully!")
            st.write("PDF Documents collected: ", stats["pdf_count"])
            st.write("JSON Documents collected: ", stats["json_count"])
            st.write("Total Documents: ", stats["total_count"])
            st.write("Cleaning Stats:")
            st.json(stats["cleaning_stats"])
            st.write("Number of chunks produced: ", stats["chunk_count"])
        else:
            st.error(f"Backend error: {resp.text}")

# Fetch sample documents/chunks
if st.button("Show Sample Cleaned Documents"):
    resp = requests.get(f"{API_URL}/sample_docs", params={"data_dir": data_dir, "n": 5})
    docs = resp.json() if resp.status_code == 200 else []
    st.header("Sample Cleaned Documents")
    for i, doc in enumerate(docs):
        st.markdown(f"**Doc {i+1} Metadata:**")
        st.json(doc["metadata"])
        st.markdown(f"**Doc {i+1} Content:**\n{doc['page_content'][:300]} ...")
        st.markdown("---")

if st.button("Show Sample Chunked Documents"):
    resp = requests.get(f"{API_URL}/sample_chunks", params={"data_dir": data_dir, "n": 5})
    chunks = resp.json() if resp.status_code == 200 else []
    st.header("Sample Chunked Documents")
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1} Metadata:**")
        st.json(chunk["metadata"])
        st.markdown(f"**Chunk {i+1} Content:**\n{chunk['page_content'][:200]} ...")
        st.markdown("---")

# Semantic retrieval
st.header("Step 2: Semantic Query Retrieval (Vector Search)")
search_query = st.text_input("Type your retrieval query for semantic search:", "")
if st.button("Search Query") and search_query:
    resp = requests.post(f"{API_URL}/semantic_search", json={
        "data_dir": data_dir,
        "query": search_query,
        "k": 5
    })
    results = resp.json() if resp.status_code == 200 else []
    st.write(f"Found `{len(results)}` semantic results:")
    if len(results) == 0:
        st.warning("No semantic matches found in chunks.")
    for i, chunk in enumerate(results):
        st.markdown(f"**Result {i+1} Metadata:**")
        st.json(chunk["metadata"])
        st.markdown(f"**Result {i+1} Content:**\n{chunk['page_content']}")
        st.markdown("---")
else:
    st.info("Type a query and press the button to retrieve semantic matches.")

st.success("Backend integration complete. Explore your RAG pipeline from a unified frontend!")
