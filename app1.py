import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit Page Configuration
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .highlight { color: #4CAF50; font-size:22px; font-weight:bold; }
    .stTextArea textarea { height: 150px !important; }
    </style>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Page Title
st.markdown("<h1 class='highlight'>📄 AI Resume Screening & Candidate Ranking System </h1>", unsafe_allow_html=True)

# Layout: Divide into two columns
col1, col2 = st.columns(2)

# Left Side: Job Description Input
with col1:
    st.subheader("📝 Job Description")
    job_description = st.text_area("Enter the job description here", height=200)

# Right Side: Resume Upload
with col2:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload multiple PDF resumes", type=["pdf"], accept_multiple_files=True)

# Processing
if uploaded_files and job_description.strip():
    st.markdown("<h2 class='highlight'>📊 Ranking Resumes...</h2>", unsafe_allow_html=True)

    resumes = []
    file_names = []

    # Progress Bar
    progress = st.progress(0)
    total_files = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        if text:
            resumes.append(text)
            file_names.append(file.name)
        else:
            st.warning(f"⚠️ Could not extract text from {file.name}. Skipping...")

        # Update progress bar
        progress.progress((i + 1) / total_files)

    if resumes:
        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Display results in a styled DataFrame
        st.success("✅ Resume ranking completed!")

        results = pd.DataFrame({"Resume": file_names, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        # Use Streamlit's dataframe display for better UI
        st.dataframe(results.style.format({"Score": "{:.2f}"}))
    else:
        st.error("🚨 No valid resumes found. Please upload PDFs with readable text.")

