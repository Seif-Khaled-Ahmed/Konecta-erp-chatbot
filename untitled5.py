"""
AI-Powered CV Parsing & Job Matching System
Extracts CV fields, embeds content, matches with job descriptions
Requirements: pip install streamlit openai PyPDF2 python-docx pandas plotly scikit-learn
"""

import streamlit as st
import openai
import PyPDF2
import docx
import json
import re
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="CV Parser & Job Matcher", page_icon="📄", layout="wide")

# ============================================
# SESSION STATE
# ============================================
if 'parsed_cvs' not in st.session_state:
    st.session_state.parsed_cvs = []
if 'job_descriptions' not in st.session_state:
    st.session_state.job_descriptions = []
if 'matches' not in st.session_state:
    st.session_state.matches = []

# ============================================
# SAMPLE JOB DESCRIPTIONS
# ============================================
SAMPLE_JOBS = {
    "Senior Software Engineer": {
        "description": """
        We are seeking a Senior Software Engineer with 5+ years of experience.
        
        Required Skills:
        - Python, Java, or C++ programming
        - Cloud platforms (AWS, Azure, GCP)
        - Microservices architecture
        - Database design (SQL, NoSQL)
        - RESTful API development
        - Git version control
        
        Nice to Have:
        - Machine learning experience
        - DevOps knowledge
        - Leadership experience
        
        Education: Bachelor's in Computer Science or related field
        """,
        "required_skills": ["Python", "Java", "Cloud", "AWS", "Azure", "Microservices", "SQL", "API", "Git"],
        "experience_years": 5
    },
    "Data Analyst": {
        "description": """
        Looking for a Data Analyst to join our analytics team.
        
        Required Skills:
        - SQL and database querying
        - Python or R for data analysis
        - Data visualization (Tableau, Power BI, or similar)
        - Statistical analysis
        - Excel proficiency
        
        Nice to Have:
        - Machine learning knowledge
        - Business intelligence tools
        - ETL processes
        
        Education: Bachelor's in Statistics, Mathematics, or related field
        Experience: 2-4 years
        """,
        "required_skills": ["SQL", "Python", "R", "Tableau", "Power BI", "Statistics", "Excel", "Data Analysis"],
        "experience_years": 3
    },
    "HR Manager": {
        "description": """
        Seeking an experienced HR Manager to lead our HR department.
        
        Required Skills:
        - HR strategy and planning
        - Recruitment and talent acquisition
        - Performance management
        - Employee relations
        - HRIS systems
        - Labor law knowledge
        
        Nice to Have:
        - Change management
        - Organizational development
        - HR analytics
        
        Education: Bachelor's in HR, Business, or related field
        Experience: 7+ years with 3+ years in management
        """,
        "required_skills": ["HR", "Recruitment", "Performance Management", "HRIS", "Labor Law", "Leadership"],
        "experience_years": 7
    }
}

# ============================================
# FILE READING FUNCTIONS
# ============================================
def read_pdf(file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def read_docx(file):
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

# ============================================
# CV PARSING WITH GPT
# ============================================
def parse_cv_with_gpt(cv_text, api_key):
    """Use GPT to extract structured information from CV"""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Extract the following information from this CV and return it as a JSON object:
        
        {{
            "name": "Full name",
            "email": "Email address",
            "phone": "Phone number",
            "education": ["List of degrees and institutions"],
            "experience_years": "Total years of experience (number only)",
            "skills": ["List of technical and professional skills"],
            "experience": ["List of job titles and companies"],
            "certifications": ["List of certifications if any"],
            "summary": "Brief professional summary (2-3 sentences)"
        }}
        
        CV Text:
        {cv_text[:3000]}
        
        Return ONLY valid JSON, no other text.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a CV parsing expert. Extract structured information and return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        # Clean up markdown if present
        result = result.replace("```json", "").replace("```", "").strip()
        
        parsed_data = json.loads(result)
        return parsed_data
        
    except Exception as e:
        st.error(f"Error parsing CV: {e}")
        return None

# ============================================
# EMBEDDING & MATCHING
# ============================================
def get_embedding(text, api_key):
    """Get OpenAI embedding for text"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text[:8000]  # Token limit
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

def calculate_match_score(cv_data, job_data, cv_embedding, job_embedding):
    """Calculate comprehensive match score"""
    
    # 1. Embedding similarity (40%)
    embedding_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]
    
    # 2. Skills match (30%)
    cv_skills = set([s.lower() for s in cv_data.get('skills', [])])
    required_skills = set([s.lower() for s in job_data['required_skills']])
    
    if required_skills:
        skills_match = len(cv_skills.intersection(required_skills)) / len(required_skills)
    else:
        skills_match = 0
    
    # 3. Experience match (20%)
    cv_exp = cv_data.get('experience_years', 0)
    required_exp = job_data['experience_years']
    
    if cv_exp >= required_exp:
        exp_score = 1.0
    elif cv_exp >= required_exp * 0.7:
        exp_score = 0.7
    else:
        exp_score = cv_exp / required_exp if required_exp > 0 else 0
    
    # 4. Education match (10%)
    education = " ".join(cv_data.get('education', [])).lower()
    has_degree = any(word in education for word in ['bachelor', 'master', 'phd', 'degree'])
    edu_score = 1.0 if has_degree else 0.5
    
    # Weighted total
    total_score = (
        embedding_score * 0.4 +
        skills_match * 0.3 +
        exp_score * 0.2 +
        edu_score * 0.1
    )
    
    return {
        'total_score': total_score,
        'embedding_score': embedding_score,
        'skills_match': skills_match,
        'experience_score': exp_score,
        'education_score': edu_score,
        'matched_skills': list(cv_skills.intersection(required_skills)),
        'missing_skills': list(required_skills - cv_skills)
    }

def generate_match_reasoning(cv_data, job_title, match_details):
    """Generate human-readable match explanation"""
    reasoning = f"**Match Analysis for {job_title}:**\n\n"
    
    score = match_details['total_score']
    if score >= 0.8:
        reasoning += "✅ **Excellent Match** - Strong alignment with requirements\n\n"
    elif score >= 0.6:
        reasoning += "⚠️ **Good Match** - Most requirements met\n\n"
    else:
        reasoning += "❌ **Partial Match** - Some gaps in requirements\n\n"
    
    reasoning += f"**Strengths:**\n"
    if match_details['matched_skills']:
        reasoning += f"- Possesses key skills: {', '.join(match_details['matched_skills'][:5])}\n"
    reasoning += f"- Experience level: {match_details['experience_score']:.0%} match\n"
    reasoning += f"- Overall profile similarity: {match_details['embedding_score']:.0%}\n\n"
    
    if match_details['missing_skills']:
        reasoning += f"**Areas for Development:**\n"
        reasoning += f"- Missing skills: {', '.join(match_details['missing_skills'][:5])}\n"
    
    return reasoning

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("📄 AI-Powered CV Parser & Job Matcher")
    st.markdown("**Generative AI for Recruitment: Extract, Embed, Match, Rank**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        
        st.markdown("---")
        st.header("📊 Statistics")
        st.metric("CVs Parsed", len(st.session_state.parsed_cvs))
        st.metric("Job Positions", len(SAMPLE_JOBS))
        st.metric("Matches Created", len(st.session_state.matches))
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.info("Get your API key at: https://platform.openai.com/api-keys")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload CVs", "💼 Job Descriptions", "🎯 Match & Rank", "📈 Analytics"])
    
    # ============================================
    # TAB 1: UPLOAD CVs
    # ============================================
    with tab1:
        st.header("Upload and Parse CVs")
        
        uploaded_files = st.file_uploader(
            "Upload CV files (PDF or DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🔍 Parse All CVs"):
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    with st.spinner(f"Processing {file.name}..."):
                        # Read file
                        if file.name.endswith('.pdf'):
                            cv_text = read_pdf(file)
                        else:
                            cv_text = read_docx(file)
                        
                        # Parse with GPT
                        parsed_data = parse_cv_with_gpt(cv_text, api_key)
                        
                        if parsed_data:
                            # Get embedding
                            cv_summary = f"{parsed_data.get('summary', '')} {' '.join(parsed_data.get('skills', []))}"
                            embedding = get_embedding(cv_summary, api_key)
                            
                            if embedding:
                                parsed_data['filename'] = file.name
                                parsed_data['embedding'] = embedding
                                parsed_data['raw_text'] = cv_text[:500]  # Store snippet
                                st.session_state.parsed_cvs.append(parsed_data)
                                st.success(f"✅ Parsed: {parsed_data.get('name', 'Unknown')}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"✅ Processed {len(uploaded_files)} CVs!")
        
        # Display parsed CVs
        if st.session_state.parsed_cvs:
            st.subheader("📋 Parsed CVs")
            for idx, cv in enumerate(st.session_state.parsed_cvs):
                with st.expander(f"CV {idx+1}: {cv.get('name', 'Unknown')} - {cv.get('filename', '')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Contact:**")
                        st.write(f"Email: {cv.get('email', 'N/A')}")
                        st.write(f"Phone: {cv.get('phone', 'N/A')}")
                        st.write(f"**Experience:** {cv.get('experience_years', 'N/A')} years")
                    with col2:
                        st.write("**Skills:**")
                        st.write(", ".join(cv.get('skills', [])[:10]))
                    
                    st.write("**Education:**")
                    for edu in cv.get('education', []):
                        st.write(f"- {edu}")
                    
                    st.write("**Summary:**")
                    st.write(cv.get('summary', 'N/A'))
    
    # ============================================
    # TAB 2: JOB DESCRIPTIONS
    # ============================================
    with tab2:
        st.header("Job Descriptions")
        
        st.info("Using sample job descriptions. In production, integrate with your job posting system.")
        
        for job_title, job_data in SAMPLE_JOBS.items():
            with st.expander(f"📌 {job_title}"):
                st.markdown(job_data['description'])
                st.write(f"**Required Skills:** {', '.join(job_data['required_skills'])}")
                st.write(f"**Experience Required:** {job_data['experience_years']}+ years")
        
        # Generate embeddings for jobs (on first load)
        if st.button("🔄 Generate Job Embeddings"):
            with st.spinner("Embedding job descriptions..."):
                for job_title, job_data in SAMPLE_JOBS.items():
                    if 'embedding' not in job_data:
                        embedding = get_embedding(job_data['description'], api_key)
                        if embedding:
                            job_data['embedding'] = embedding
                st.success("✅ Job embeddings generated!")
    
    # ============================================
    # TAB 3: MATCH & RANK
    # ============================================
    with tab3:
        st.header("Match CVs to Jobs")
        
        if not st.session_state.parsed_cvs:
            st.warning("⚠️ Please upload and parse CVs first (Tab 1)")
            return
        
        # Ensure job embeddings exist
        jobs_ready = all('embedding' in job for job in SAMPLE_JOBS.values())
        if not jobs_ready:
            st.warning("⚠️ Please generate job embeddings first (Tab 2)")
            return
        
        selected_job = st.selectbox("Select Job Position", list(SAMPLE_JOBS.keys()))
        
        if st.button("🎯 Match All Candidates"):
            st.session_state.matches = []
            job_data = SAMPLE_JOBS[selected_job]
            job_embedding = job_data['embedding']
            
            with st.spinner("Matching candidates..."):
                for cv in st.session_state.parsed_cvs:
                    match_details = calculate_match_score(
                        cv, job_data, cv['embedding'], job_embedding
                    )
                    
                    reasoning = generate_match_reasoning(cv, selected_job, match_details)
                    
                    st.session_state.matches.append({
                        'name': cv.get('name', 'Unknown'),
                        'email': cv.get('email', 'N/A'),
                        'score': match_details['total_score'],
                        'details': match_details,
                        'reasoning': reasoning,
                        'cv_data': cv
                    })
                
                # Sort by score
                st.session_state.matches.sort(key=lambda x: x['score'], reverse=True)
            
            st.success(f"✅ Matched {len(st.session_state.matches)} candidates!")
        
        # Display matches
        if st.session_state.matches:
            st.subheader(f"🏆 Ranked Candidates for: {selected_job}")
            
            for rank, match in enumerate(st.session_state.matches, 1):
                score = match['score']
                
                # Color code based on score
                if score >= 0.8:
                    badge = "🟢"
                elif score >= 0.6:
                    badge = "🟡"
                else:
                    badge = "🔴"
                
                with st.expander(f"{badge} Rank #{rank}: {match['name']} - Match Score: {score:.1%}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(match['reasoning'])
                        
                        st.write("**Detailed Scores:**")
                        details = match['details']
                        score_df = pd.DataFrame({
                            'Component': ['Embedding Similarity', 'Skills Match', 'Experience', 'Education'],
                            'Score': [
                                details['embedding_score'],
                                details['skills_match'],
                                details['experience_score'],
                                details['education_score']
                            ]
                        })
                        
                        fig = px.bar(score_df, x='Component', y='Score', 
                                    title="Score Breakdown",
                                    color='Score',
                                    color_continuous_scale='RdYlGn')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Contact:**")
                        st.write(f"📧 {match['email']}")
                        
                        st.write("**Quick Stats:**")
                        st.metric("Experience", f"{match['cv_data'].get('experience_years', 0)} yrs")
                        st.metric("Skills Match", f"{details['skills_match']:.0%}")
                        
                        if st.button(f"📄 View Full CV", key=f"view_{rank}"):
                            st.write(match['cv_data'].get('raw_text', 'N/A'))
    
    # ============================================
    # TAB 4: ANALYTICS
    # ============================================
    with tab4:
        st.header("📈 Recruitment Analytics")
        
        if not st.session_state.matches:
            st.info("Run matching first to see analytics")
            return
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        scores = [m['score'] for m in st.session_state.matches]
        with col1:
            st.metric("Avg Match Score", f"{np.mean(scores):.1%}")
        with col2:
            st.metric("Top Candidate", f"{max(scores):.1%}")
        with col3:
            qualified = sum(1 for s in scores if s >= 0.6)
            st.metric("Qualified Candidates", f"{qualified}/{len(scores)}")
        with col4:
            st.metric("Total Applicants", len(scores))
        
        # Score distribution
        st.subheader("Score Distribution")
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Candidate Match Score Distribution",
            labels={'x': 'Match Score', 'y': 'Count'}
        )
        fig.add_vline(x=0.6, line_dash="dash", line_color="red", 
                     annotation_text="Qualification Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        # Skills gap analysis
        st.subheader("Skills Gap Analysis")
        all_missing = {}
        for match in st.session_state.matches:
            for skill in match['details']['missing_skills']:
                all_missing[skill] = all_missing.get(skill, 0) + 1
        
        if all_missing:
            skills_df = pd.DataFrame(
                list(all_missing.items()),
                columns=['Skill', 'Candidates Missing']
            ).sort_values('Candidates Missing', ascending=False).head(10)
            
            fig = px.bar(skills_df, x='Skill', y='Candidates Missing',
                        title="Top 10 Missing Skills Across Candidates")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        st.subheader("📥 Export Results")
        if st.button("Download Ranking Report"):
            export_data = []
            for rank, match in enumerate(st.session_state.matches, 1):
                export_data.append({
                    'Rank': rank,
                    'Name': match['name'],
                    'Email': match['email'],
                    'Overall Score': f"{match['score']:.2%}",
                    'Skills Match': f"{match['details']['skills_match']:.2%}",
                    'Experience Score': f"{match['details']['experience_score']:.2%}",
                    'Matched Skills': ', '.join(match['details']['matched_skills']),
                    'Missing Skills': ', '.join(match['details']['missing_skills'])
                })
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv,
                "candidate_ranking.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()