import os
import sys
import uuid
import requests
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

# Set Page Config
st.set_page_config(
    page_title="MediBot - AI Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme & Cards)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E293B;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 0.9rem;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-info {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE INITIALIZATION ---------------- #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_BASE_URL", "http://localhost:8000")


# ---------------- HELPER FUNCTIONS ---------------- #
def check_api_health(api_url: str) -> bool:
    try:
        res = requests.get(f"{api_url}/", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


def fetch_catalog(api_url: str) -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{api_url}/catalog/", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.sidebar.error(f"Catalog fetch error: {e}")
    return []


def upload_file_to_api(api_url: str, uploaded_file, category: str) -> Optional[Dict[str, Any]]:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"category": category}
        res = requests.post(f"{api_url}/upload/", files=files, data=data, timeout=300)
        if res.status_code == 200:
            return res.json()
        else:
            st.sidebar.error(f"Upload failed ({res.status_code}): {res.text}")
    except Exception as e:
        st.sidebar.error(f"Upload request error: {e}")
    return None


def ingest_url_to_api(api_url: str, url: str, category: str) -> Optional[Dict[str, Any]]:
    try:
        payload = {"url": url, "category": category}
        res = requests.post(f"{api_url}/ingest-url/", json=payload, timeout=300)
        if res.status_code == 200:
            return res.json()
        else:
            st.sidebar.error(f"URL ingestion failed ({res.status_code}): {res.text}")
    except Exception as e:
        st.sidebar.error(f"URL ingestion error: {e}")
    return None


def delete_doc_from_api(api_url: str, doc_id: str) -> bool:
    try:
        res = requests.delete(f"{api_url}/catalog/{doc_id}", timeout=30)
        return res.status_code == 200
    except Exception as e:
        st.sidebar.error(f"Delete doc error: {e}")
        return False


def query_rag_api(api_url: str, question: str, session_id: str, file_name: Optional[str], category: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        payload = {
            "question": question,
            "session_id": session_id,
            "file_name": file_name if file_name != "All Documents" else None,
            "category": category if category != "All Categories" else None
        }
        res = requests.post(f"{api_url}/ask/", json=payload, timeout=180)
        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"API Error ({res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Connection error to backend: {e}")
    return None



# ---------------- SIDEBAR CONTROLS ---------------- #
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=64)
    st.title("MediBot Control Panel")

    # API Base URL Config
    api_url = st.text_input("FastAPI Base URL", value=st.session_state.api_url)
    st.session_state.api_url = api_url.rstrip("/")

    # Check API Status
    api_online = check_api_health(st.session_state.api_url)
    if api_online:
        st.success("🟢 Backend API: Online")
    else:
        st.warning("🔴 Backend API: Offline (Start `python main.py` in backend)")

    st.divider()

    # 1. Document Ingestion Section
    st.subheader("📥 Add Documents")
    ingest_tab1, ingest_tab2 = st.tabs(["📁 Local File", "🌐 Web URL"])

    with ingest_tab1:
        doc_category = st.selectbox("Category", ["general", "medical_guidelines", "pharmacology", "lab_reports"], key="cat_file")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv", "docx"])
        if uploaded_file and st.button("Upload & Ingest", use_container_width=True):
            with st.spinner("Parsing, chunking & indexing document..."):
                res = upload_file_to_api(st.session_state.api_url, uploaded_file, doc_category)
                if res:
                    st.success(f"Ingested '{uploaded_file.name}' ({res.get('chunks_created', 0)} chunks)")
                    st.rerun()

    with ingest_tab2:
        url_category = st.selectbox("Category", ["general", "medical_guidelines", "pharmacology", "lab_reports"], key="cat_url")
        url_input = st.text_input("Document URL", placeholder="https://example.com/guideline.pdf")
        if url_input and st.button("Download & Ingest URL", use_container_width=True):
            with st.spinner("Downloading & indexing URL..."):
                res = ingest_url_to_api(st.session_state.api_url, url_input, url_category)
                if res:
                    st.success(f"Ingested URL ({res.get('chunks_created', 0)} chunks)")
                    st.rerun()

    st.divider()

    # 2. Document Catalog Management
    st.subheader("📚 Active Catalog")
    catalog = fetch_catalog(st.session_state.api_url)

    if catalog:
        for doc in catalog:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{doc.get('filename')}**")
                st.caption(f"Category: `{doc.get('category')}` | Chunks: `{doc.get('chunks_count')}`")
            with col2:
                if st.button("🗑️", key=f"del_{doc.get('doc_id')}", help="Delete document vectors & metadata"):
                    if delete_doc_from_api(st.session_state.api_url, doc.get('doc_id')):
                        st.toast(f"Deleted {doc.get('filename')}", icon="🗑️")
                        st.rerun()
            st.divider()
    else:
        st.info("No documents cataloged yet.")

    # 3. Search Scope & Filters
    st.subheader("🎯 Search Scope")
    doc_options = ["All Documents"] + [doc.get("filename") for doc in catalog if doc.get("filename")]
    selected_doc_filter = st.selectbox("Filter Document Scope", doc_options)

    cat_options = ["All Categories", "general", "medical_guidelines", "pharmacology", "lab_reports"]
    selected_cat_filter = st.selectbox("Filter Category Scope", cat_options)

    st.divider()
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ---------------- MAIN CHAT INTERFACE ---------------- #
st.markdown('<div class="main-header">🩺 MediBot AI Medical Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Document RAG with Dual-Stage Hybrid Search (Pinecone 384d + BM25) & Reciprocal Rank Fusion</div>',
    unsafe_allow_html=True
)

# Metric Badges
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Active Documents</div>
        <div class="metric-value">{}</div>
    </div>
    """.format(len(catalog)), unsafe_allow_html=True)
with col_m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Retrieval Mode</div>
        <div class="metric-value">Dense + BM25 RRF</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Scope</div>
        <div class="metric-value">{}</div>
    </div>
    """.format(selected_doc_filter if selected_doc_filter != "All Documents" else "Multi-Document"), unsafe_allow_html=True)
with col_m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Session ID</div>
        <div class="metric-value">{}...</div>
    </div>
    """.format(st.session_state.session_id[:8]), unsafe_allow_html=True)

st.divider()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Retrieved Sources & Chunks"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {idx}:** `{src}`")
                if "chunks" in message and message["chunks"]:
                    st.divider()
                    for idx, chunk_text in enumerate(message["chunks"], 1):
                        st.markdown(f"**Chunk {idx}:**\n> {chunk_text}")

# User Input
if prompt := st.chat_input("Ask any medical, dosage, or clinical question..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Decomposing sub-queries, retrieving vectors & generating answer..."):
            response = query_rag_api(
                api_url=st.session_state.api_url,
                question=prompt,
                session_id=st.session_state.session_id,
                file_name=selected_doc_filter,
                category=selected_cat_filter
            )

            if response:
                answer = response.get("answer", "No answer received.")
                sources = response.get("sources", [])
                chunks = response.get("chunks_text", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View Retrieved Sources & Chunks"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**Source {idx}:** `{src}`")
                        if chunks:
                            st.divider()
                            for idx, chunk_text in enumerate(chunks, 1):
                                st.markdown(f"**Chunk {idx}:**\n> {chunk_text}")

                # Append assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks": chunks
                })
