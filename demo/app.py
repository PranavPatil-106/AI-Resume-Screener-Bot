import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import streamlit as st
from pypdf import PdfReader

try:
    from vectorstore.hf_store import HuggingFaceVectorStore
    from rag_pipeline.rag import ResumeRAG
    from groq import Groq
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# --- Helpers --- PDF TextExtraction
def extract_text_from_pdf(file_obj):
    reader = PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def extract_name_from_resume(content):
    """Extract candidate name from resume content"""
    import re
    
    # Common patterns for names in resumes
    patterns = [
        r'^([A-Z][a-z]+ [A-Z][a-z]+)',  # First Last at start of line
        r'Name:\s*([A-Z][a-z]+ [A-Z][a-z]+)',  # Name: First Last
        r'Full Name:\s*([A-Z][a-z]+ [A-Z][a-z]+)',  # Full Name: First Last
        r'Candidate:\s*([A-Z][a-z]+ [A-Z][a-z]+)',  # Candidate: First Last
    ]
    
    lines = content.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    
    return None

def preprocess_and_store(files, vectorstore: HuggingFaceVectorStore):
    docs = []
    for f in files:
        text = extract_text_from_pdf(f)
        chunks = chunk_text(text)
        for chunk in chunks:
            docs.append({"resume_id": f.name, "content": chunk})
    vectorstore.add_documents(docs)
    return docs

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Screening RAG using Hugging Face & Groq", layout="wide")
st.title("📋 Resume Screening RAG (Hugging Face + Groq)")

# Fixed model configuration
st.sidebar.subheader("Model Configuration")
selected_model = "llama-3.3-70b-versatile"  # Best quality model
st.sidebar.info(f"Using: {selected_model}")

# Number of candidates to compare (Top-K)
st.sidebar.subheader("Results Configuration")
st.sidebar.slider("Candidates to compare (Top-K)", min_value=1, max_value=20, value=5, step=1, key="top_k")

if "vs" not in st.session_state:
    try:
        with st.spinner("Initializing model..."):
            st.session_state.vs = HuggingFaceVectorStore()
            st.session_state.rag = ResumeRAG(st.session_state.vs, reasoning_model=selected_model)
        st.success(f"✅ Model '{selected_model}' initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
        st.info("Please make sure you have installed all dependencies and set up your GROQ_API_KEY in .env file")
        st.stop()

st.sidebar.header("Upload Resumes")
uploaded = st.sidebar.file_uploader("Upload resume PDFs", type=["pdf"], accept_multiple_files=True)

# Clear button to reset everything
if st.sidebar.button("🗑️ Clear All Data", help="Clear all uploaded resumes and start fresh"):
    st.session_state.vs = HuggingFaceVectorStore()
    st.session_state.rag = ResumeRAG(st.session_state.vs, reasoning_model=selected_model)
    st.session_state.processed_files = set()
    st.success("✅ All data cleared! You can now upload new resumes.")
    st.rerun()

if uploaded:
    # Initialize separate list for new documents
    new_docs = []
    
    # Check if 'processed_files' exists in session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
        
    with st.spinner("Processing resumes..."):
        for f in uploaded:
            # Create a unique identifier for the file (e.g., name + size)
            file_id = f"{f.name}_{f.size}"
            
            if file_id not in st.session_state.processed_files:
                text = extract_text_from_pdf(f)
                chunks = chunk_text(text)
                for chunk in chunks:
                    new_docs.append({"resume_id": f.name, "content": chunk})
                
                # Mark as processed
                st.session_state.processed_files.add(file_id)
        
        # Only add to vectorstore if there are new documents
        if new_docs:
            if "vs" not in st.session_state or st.session_state.vs is None:
                 st.session_state.vs = HuggingFaceVectorStore()
                 st.session_state.rag = ResumeRAG(st.session_state.vs, reasoning_model=selected_model)
                 
            st.session_state.vs.add_documents(new_docs)
            st.success(f"✅ Processed {len(new_docs)} new chunks from uploaded files.")
        else:
            st.info("Files already processed.")

# Show current status
if "vs" in st.session_state and hasattr(st.session_state.vs, 'vectors') and st.session_state.vs.vectors:
    unique_resumes = set(v['resume_id'] for v in st.session_state.vs.vectors)
    st.info(f"📊 Currently loaded: {len(unique_resumes)} resume(s) with {len(st.session_state.vs.vectors)} chunks")
else:
    st.info("📊 No resumes loaded. Upload some PDFs to get started.")

st.subheader("Enter Job Description or Search Query")

# Pre-written queries
st.write("**Quick Queries:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🐍 Python Dev"):
        st.session_state.job_query = "Python developer with 3+ years experience in web development, Django/Flask, databases"
with col2:
    if st.button("📊 Data Scientist"):
        st.session_state.job_query = "Data scientist with machine learning experience, Python, SQL, statistical analysis"
with col3:
    if st.button("⚛️ Full Stack"):
        st.session_state.job_query = "Full stack developer with React, Node.js, Python, database experience"
with col4:
    if st.button("🤖 ML Engineer"):
        st.session_state.job_query = "Machine learning engineer with TensorFlow, PyTorch, Python, deep learning experience"

col5, col6, col7, col8 = st.columns(4)
with col5:
    if st.button("☁️ DevOps"):
        st.session_state.job_query = "DevOps engineer with AWS, Docker, Kubernetes, CI/CD experience"
with col6:
    if st.button("📱 Mobile Dev"):
        st.session_state.job_query = "Mobile developer with React Native, Flutter, iOS, Android experience"
with col7:
    if st.button("🎨 Frontend"):
        st.session_state.job_query = "Frontend developer with React, JavaScript, HTML, CSS, UI/UX experience"
with col8:
    if st.button("🔧 Backend"):
        st.session_state.job_query = "Backend developer with Node.js, Python, APIs, microservices, database experience"

# Initialize job_query in session state if not exists
if "job_query" not in st.session_state:
    st.session_state.job_query = ""

job_desc = st.text_area("Job Description / Search Query", value=st.session_state.job_query, height=100)

# Create two columns for main content and sidebar
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    if st.button("Find Best Matches"):
        if job_desc.strip() == "":
            st.warning("Please enter some job description or search query.")
        else:
            with st.spinner("Finding best matches..."):
                top_k = st.session_state.get("top_k", 3)
                answer = st.session_state.rag.answer_query(job_desc, top_k=top_k)
                # Store results in session state for sidebar
                st.session_state.last_results = answer
            st.write("### Results")
            st.write(answer)

with col_sidebar:
    st.subheader("📊 Candidates")
    
    if "vs" in st.session_state and hasattr(st.session_state.vs, 'vectors') and st.session_state.vs.vectors:
        # Get unique resume names
        unique_resumes = list(set(v['resume_id'] for v in st.session_state.vs.vectors))
        
        st.write(f"**Loaded Resumes ({len(unique_resumes)}):**")
        
        # Show candidates in order
        for i, resume_id in enumerate(unique_resumes, 1):
            with st.container():
                # Try to extract name from resume content
                resume_content = next((v['content'] for v in st.session_state.vs.vectors if v['resume_id'] == resume_id), "")
                name = extract_name_from_resume(resume_content)
                
                if name:
                    st.write(f"**{i}. 👤 {name}**")
                    st.caption(f"📄 {resume_id}")
                else:
                    st.write(f"**{i}. 📄 {resume_id}**")
                
                st.divider()
    else:
        st.info("No resumes loaded yet")
