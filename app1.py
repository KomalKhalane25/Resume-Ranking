import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
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

# Predefined skills list (you can expand it)
SKILL_LIST = [
    "Python", "Java", "JavaScript", "C++", "C#", "SQL", "Machine Learning", "Data Science",
    "Deep Learning", "NLP", "Django", "Flask", "Spring Boot", "React", "Angular", "Node.js",
    "HTML", "CSS", "Docker", "Kubernetes", "Cloud Computing", "AWS", "Azure", "GCP",
    "REST API", "Microservices", "Git", "DevOps", "Agile", "Scrum"
]

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# Function to extract skills from a given text
def extract_skills(text):
    found_skills = set()
    for skill in SKILL_LIST:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.add(skill)
    return found_skills

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
st.markdown("<h1 class='highlight'>📄 AI Resume Screening & Candidate Ranking</h1>", unsafe_allow_html=True)

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
    skill_matches = []
    skill_missing = []

    # Extract skills from job description
    required_skills = extract_skills(job_description)

    # Progress Bar
    progress = st.progress(0)
    total_files = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        if text:
            resumes.append(text)
            file_names.append(file.name)

            # Extract skills from resume
            resume_skills = extract_skills(text)
            
            # Find matched and unmatched skills
            matched_skills = resume_skills.intersection(required_skills)
            missing_skills = required_skills - matched_skills

            skill_matches.append(", ".join(matched_skills) if matched_skills else "None")
            skill_missing.append(", ".join(missing_skills) if missing_skills else "None")
        else:
            st.warning(f"⚠️ Could not extract text from {file.name}. Skipping...")

        # Update progress bar
        progress.progress((i + 1) / total_files)

    if resumes:
        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Display results in a styled DataFrame
        st.success("✅ Resume ranking completed!")

        results = pd.DataFrame({
            "Resume": file_names,
            "Score": scores,
            "Matched Skills": skill_matches,   # Skills found in the resume
            "Missing Skills": skill_missing    # Skills required but missing
        })
        results = results.sort_values(by="Score", ascending=False)

        # Use Streamlit's dataframe display for better UI
        st.dataframe(results.style.format({"Score": "{:.2f}"}))
    else:
        st.error("🚨 No valid resumes found. Please upload PDFs with readable text.")
