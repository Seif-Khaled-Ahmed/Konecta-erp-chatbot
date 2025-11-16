"""
ERP RAG Chatbot with Structured Workflows and Analytics
FREE VERSION - Uses Google Gemini API (Free tier: 60 requests/minute)
Get free API key: https://makersuite.google.com/app/apikey

Installation:
pip install streamlit google-generativeai faiss-cpu pandas plotly

Usage:
streamlit run chatbot.py
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import re

import streamlit as st
import requests
import os

# Get API key from secrets
api_key = st.secrets["API_KEY"]
api_url = st.secrets["API_URL"]

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def call_private_api(data):
    response = requests.post(api_url, json=data, headers=headers)
    return response.json()
# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Konecta ERP Chatbot", 
    page_icon="logo.jpeg",  # Use the logo file instead of emoji
    layout="wide"
)
# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analytics' not in st.session_state:
    st.session_state.analytics = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

# ============================================
# SAMPLE ERP DOCUMENTS (Replace with real docs)
# ============================================
from pypdf import PdfReader

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
SAMPLE_DOCUMENTS = {
    "HR_Policy": read_pdf("Konecta Finance document.pdf"),
    "Finance_Policy": read_pdf("Konecta HR document.pdf")
}

# ============================================
# SIMPLE VECTOR STORE (No embeddings needed)
# ============================================
class SimpleKnowledgeBase:
    """Simple keyword-based retrieval (no embeddings required)"""
    
    def __init__(self, documents):
        self.documents = documents
        self.doc_text = "\n\n".join([f"Document: {name}\n{content}" 
                                     for name, content in documents.items()])
    
    def search(self, query, top_k=3):
        """Simple keyword matching"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        results = []
        for doc_name, doc_content in self.documents.items():
            doc_lower = doc_content.lower()
            doc_words = set(re.findall(r'\w+', doc_lower))
            
            # Calculate overlap
            overlap = len(query_words & doc_words)
            if overlap > 0:
                results.append({
                    'name': doc_name,
                    'content': doc_content,
                    'score': overlap / len(query_words)
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

# ============================================
# GEMINI AI FUNCTIONS
# ============================================
def initialize_gemini(api_key):
    """Initialize Gemini API"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        return False

def query_with_rag(query, knowledge_base, api_key):
    """RAG: Retrieve relevant docs + Generate answer"""
    try:
        # Retrieve relevant documents
        relevant_docs = knowledge_base.search(query, top_k=3)
        
        if not relevant_docs:
            context = "No relevant documents found."
        else:
            context = "\n\n".join([
                f"Source: {doc['name']}\n{doc['content'][:500]}"
                for doc in relevant_docs
            ])
        
        # Generate answer with Gemini
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are a helpful ERP assistant for Konecta. Answer the user's question based on the provided context.

Context from knowledge base:
{context}

User Question: {query}

Instructions:
- Answer based on the context provided
- Be concise and helpful
- Cite which document you're referencing when relevant
- You may also converse with the user normally but not fabricate information.
- If you can't answer because it isn't in the context, say "I don't have that information in the knowledge base"

Answer:"""
        
        response = model.generate_content(prompt)
        answer = response.text
        
        # Calculate confidence based on document relevance
        confidence = relevant_docs[0]['score'] if relevant_docs else 0.0
        
        return {
            'answer': answer,
            'sources': relevant_docs,
            'confidence': min(confidence, 1.0)
        }
        
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
        return None

# ============================================
# INTENT DETECTION
# ============================================
def detect_intent(query):
    """Detect if query requires structured workflow"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['apply', 'request', 'submit']) and 'leave' in query_lower:
        return 'leave_request'
    elif any(word in query_lower for word in ['expense', 'reimbursement', 'receipt']):
        return 'expense_submission'
    else:
        return 'general_query'

def log_analytics(query, intent, response, confidence, sources):
    """Log interaction for analytics"""
    st.session_state.analytics.append({
        'timestamp': datetime.now(),
        'query': query,
        'intent': intent,
        'confidence': confidence,
        'sources_count': len(sources) if sources else 0,
        'resolved': confidence > 0.5
    })

# ============================================
# STRUCTURED WORKFLOWS
# ============================================
def leave_request_form():
    """Structured workflow for leave requests"""
    st.subheader("📝 Leave Request Form")
    
    with st.form("leave_form"):
        leave_type = st.selectbox(
            "Leave Type",
            ["Annual Leave", "Sick Leave", "Emergency Leave"]
        )
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        reason = st.text_area("Reason")
        
        submitted = st.form_submit_button("Submit Request")
        
        if submitted:
            st.success(f"✅ Leave request submitted successfully!")
            st.info(f"Type: {leave_type} | Dates: {start_date} to {end_date}")
            st.info("Your manager will be notified and you'll receive approval within 48 hours.")
            return True
    return False

def expense_submission_form():
    """Structured workflow for expense submission"""
    st.subheader("💰 Expense Reimbursement Form")
    
    with st.form("expense_form"):
        expense_type = st.selectbox(
            "Expense Type",
            ["Travel", "Meals", "Office Supplies", "Other"]
        )
        amount = st.number_input("Amount ($)", min_value=0.0, step=0.01)
        description = st.text_area("Description")
        receipt = st.file_uploader("Upload Receipt", type=['pdf', 'jpg', 'png'])
        
        submitted = st.form_submit_button("Submit Expense")
        
        if submitted:
            st.success(f"✅ Expense submitted successfully!")
            st.info(f"Type: {expense_type} | Amount: ${amount:.2f}")
            st.info("Finance team will review within 5 business days.")
            return True
    return False

# ============================================
# MAIN APP
# ============================================
def main():
        # Use get() with fallback
    api_key = st.secrets.get("API_KEY")
    
    if not api_key:
        st.error("API key not found in secrets. Please configure it in Streamlit Cloud.")
        st.stop()
    initialize_gemini(api_key)
    value = api_key
    # Enhanced Custom CSS - Updated with dark blue #2900C8
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* FORCE LIGHT THEME EVERYWHERE */
        .stApp, .main, .block-container, body, html {
            background: linear-gradient(135deg, #F0F4F8 0%, #E8F1F8 100%) !important;
            color: #2C3E50 !important;
        }
        
        /* Remove dark theme from all containers */
        .stApp > header, header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        footer {
            background-color: transparent !important;
            color: #5E4BD4 !important;
        }
        
        /* CRITICAL: Fix bottom chat input dark background */
        .stBottom, [data-testid="stBottom"], 
        .stChatFloatingInputContainer,
        [data-testid="stChatFloatingInputContainer"],
        .stChatInputContainer {
            background-color: #F0F4F8 !important;
            background: #F0F4F8 !important;
            border-top: 3px solid #2900C8 !important;
            padding: 20px !important;
            box-shadow: 0 -4px 20px rgba(41, 0, 200, 0.1) !important;
        }
        
        /* Force remove dark background on bottom area */
        [data-testid="stBottomBlockContainer"],
        .stBottomContainer {
            background-color: #F0F4F8 !important;
            background: #F0F4F8 !important;
        }
        
        /* Chat input box itself */
        .stChatInput {
            background-color: #FFFFFF !important;
        }
        .stTextInput > div > div > input:focus {
    border-color: #2900C8 !important;
    box-shadow: 0 0 0 3px rgba(41, 0, 200, 0.1) !important;
    outline: 2px solid #2900C8 !important;
}
        .stChatInput > div {
            background-color: #FFFFFF !important;
            border: 2px solid #2900C8 !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 12px rgba(41, 0, 200, 0.15) !important;
        }
        
        /* Chat input textarea */
        .stChatInput textarea {
            background-color: #FFFFFF !important;
            color: #1A1A1A !important;
            font-size: 15px !important;
            border: none !important;
        }
        
        .stChatInput textarea::placeholder {
            color: #7E6FD4 !important;
        }
        
        /* Chat input send button */
        .stChatInput button {
            background: linear-gradient(135deg, #2900C8 0%, #1F0096 100%) !important;
            color: #FFFFFF !important;
            border-radius: 12px !important;
        }
        
        .stChatInput button:hover {
            background: linear-gradient(135deg, #1F0096 0%, #150064 100%) !important;
        }
        
        /* LOADING SCREEN - Fix dark background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #F0F4F8 0%, #E8F1F8 100%) !important;
        }
        
        /* Loading spinner container */
        [data-testid="stSpinner"],
        .stSpinner {
            background-color: transparent !important;
        }
        
        /* Spinner itself - Dark Blue theme */
        .stSpinner > div {
            border-top-color: #2900C8 !important;
            border-right-color: #5E4BD4 !important;
            border-bottom-color: #7E6FD4 !important;
            border-left-color: #9E8FE4 !important;
        }
        
        /* Status container during loading */
        .stStatus {
            background-color: #E8E4F9 !important;
            border: 2px solid #2900C8 !important;
            border-radius: 12px !important;
        }
        
        .stStatus > div {
            color: #2900C8 !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #FFFFFF !important;
            border-right: 3px solid #2900C8;
            box-shadow: 2px 0 10px rgba(41, 0, 200, 0.1);
        }
        
        [data-testid="stSidebar"] * {
            color: #1A1A1A !important;
        }
        
        /* Override all text colors to be dark */
        .stApp p, .stApp span, .stApp div, .stApp label, .stMarkdown {
            color: #2C3E50 !important;
        }
        
        /* Title styling */
        h1 {
            color: #2900C8 !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 4px rgba(41, 0, 200, 0.1);
        }
        
        /* Headers */
        h2 {
            color: #2900C8 !important;
            font-weight: 600 !important;
        }
        
        h3, h4 {
            color: #2900C8 !important;
            font-weight: 600 !important;
        }
        
        /* Chat messages container */
        .stChatMessage {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            padding: 20px !important;
            margin: 12px 0 !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08) !important;
            border: 1px solid #E8E4F9 !important;
        }
        
        /* User message - Purple/Blue background */
        [data-testid="stChatMessageContent"] {
            color: #1A1A1A !important;
        }
        
        .stChatMessage[data-testid*="user"] {
            background: linear-gradient(135deg, #E8E4F9 0%, #D4CFEF 100%) !important;
            border-left: 5px solid #2900C8 !important;
        }
        
        /* Assistant message - White background */
        .stChatMessage[data-testid*="assistant"] {
            background-color: #FFFFFF !important;
            border-left: 5px solid #5E4BD4 !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #2900C8 0%, #1F0096 100%) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 10px rgba(41, 0, 200, 0.3) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #1F0096 0%, #150064 100%) !important;
            box-shadow: 0 6px 16px rgba(41, 0, 200, 0.4) !important;
            transform: translateY(-2px);
        }
        
        /* Form submit button */
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #2900C8 0%, #1F0096 100%) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            width: 100% !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #2900C8 !important;
            font-weight: 700 !important;
            font-size: 28px !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #2C3E50 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Info/Alert boxes */
        .stAlert {
            background-color: #E8E4F9 !important;
            border-left: 5px solid #2900C8 !important;
            border-radius: 12px !important;
            color: #2900C8 !important;
            padding: 16px !important;
        }
        
        /* Success box */
        .stSuccess {
            background-color: #E8F5E9 !important;
            border-left: 5px solid #4CAF50 !important;
            color: #2E7D32 !important;
        }
        
        /* Warning box */
        .stWarning {
            background-color: #FFF3E0 !important;
            border-left: 5px solid #FF9800 !important;
            color: #E65100 !important;
        }
        
        /* Error box */
        .stError {
            background-color: #FFEBEE !important;
            border-left: 5px solid #F44336 !important;
            color: #C62828 !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            border-radius: 10px !important;
            border: 2px solid #D4CFEF !important;
            color: #2900C8 !important;
            font-weight: 600 !important;
            padding: 12px !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #E8E4F9 !important;
            border-color: #2900C8 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #F8FBFF !important;
            border: 2px solid #E8E4F9 !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
            padding: 16px !important;
        }
        
        /* Form styling */
        [data-testid="stForm"] {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            padding: 24px !important;
            border: 2px solid #2900C8 !important;
            box-shadow: 0 4px 12px rgba(41, 0, 200, 0.15) !important;
        }
        
        /* Text input */
        .stTextInput > div > div > input {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #D4CFEF !important;
            color: #1A1A1A !important;
            font-size: 15px !important;
            padding: 12px !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #2900C8 !important;
            box-shadow: 0 0 0 3px rgba(41, 0, 200, 0.1) !important;
        }
        
        .stTextInput label {
            color: #2900C8 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #D4CFEF !important;
        }
        
        .stSelectbox label {
            color: #2900C8 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Date input */
        .stDateInput > div > div > input {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #D4CFEF !important;
            color: #1A1A1A !important;
        }
        
        .stDateInput label {
            color: #2900C8 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Text area */
        .stTextArea > div > div > textarea {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #D4CFEF !important;
            color: #1A1A1A !important;
            font-size: 15px !important;
        }
        
        .stTextArea label {
            color: #2900C8 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Number input */
        .stNumberInput > div > div > input {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #D4CFEF !important;
            color: #1A1A1A !important;
        }
        
        .stNumberInput label {
            color: #2900C8 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* File uploader */
        .stFileUploader {
            background-color: #F8FBFF !important;
            border: 2px dashed #D4CFEF !important;
            border-radius: 10px !important;
            padding: 20px !important;
        }
        
        .stFileUploader label {
            color: #2900C8 !important;
            font-weight: 600 !important;
        }
        
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
        }
        
        /* Divider */
        hr {
            border-color: #D4CFEF !important;
            margin: 24px 0 !important;
        }
        
        /* Remove any remaining dark backgrounds */
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            background-color: transparent !important;
        }
        
        /* Markdown code blocks */
        code {
            background-color: #E8E4F9 !important;
            color: #2900C8 !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }
        
        pre {
            background-color: #F8FBFF !important;
            border: 2px solid #D4CFEF !important;
            border-radius: 8px !important;
            padding: 16px !important;
        }
        
        pre code {
            background-color: transparent !important;
            color: #2900C8 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load and encode the logo
    import base64
    try:
        with open("logo.jpeg", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{logo_data}" style="width: 60px; height: 60px; margin-right: 15px; vertical-align: middle;">'
    except:
        # Fallback if logo file not found
        logo_html = ''
    
    # Header with logo
    st.markdown(f"""
        <div style='text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #FFFFFF 0%, #E8E4F9 100%); 
                    border-radius: 20px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(41, 0, 200, 0.15);'>
            <h1 style='margin: 0; color: #2900C8; font-size: 42px; font-weight: 700;'>
                {logo_html}Konecta ERP Chatbot
            </h1>
            <p style='color: #5E4BD4; font-size: 18px; margin-top: 12px; font-weight: 500;'>
                AI-Powered Assistant with RAG + Structured Workflows
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Configuration section header only (no container)
        st.markdown("""
            <h2 style='color: #2900C8 !important; margin: 0 0 20px 0; font-weight: 600; font-size: 20px;'>
                ⚙️ Configuration
            </h2>
        """, unsafe_allow_html=True)
        
        # Set default API key
        default_api_key = "AIzaSyClemJJjKHTuEIMtXy1-WJBJZkGmZK2Pzw"
    
        api_key = st.text_input(
            "🔑 Google Gemini API Key", 
            value=default_api_key,
            type="password",
            help="Enter your Google Gemini API key to start using the chatbot"
        )
    
        if api_key and not st.session_state.knowledge_base:
            with st.spinner("🔄 Initializing knowledge base..."):
                if initialize_gemini(api_key):
                    st.session_state.knowledge_base = SimpleKnowledgeBase(SAMPLE_DOCUMENTS)
                    st.success("✅ Knowledge base loaded successfully!")
        
        st.markdown("<hr style='margin: 30px 0; border: 2px solid #D4CFEF;'>", unsafe_allow_html=True)
        
        # Analytics section - only show when data exists
        if st.session_state.analytics:
            st.markdown("""
                <h2 style='color: #2900C8 !important; margin: 0 0 20px 0; font-weight: 600; font-size: 20px;'>
                    📊 Analytics
                </h2>
            """, unsafe_allow_html=True)
            
            df = pd.DataFrame(st.session_state.analytics)
            
            # Key metrics with cards
            st.markdown("""
                <div style='background-color: #E8E4F9; padding: 16px; border-radius: 10px; 
                            border: 2px solid #D4CFEF; margin-bottom: 20px;'>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 Total Queries", len(df))
                st.metric("🎯 Avg Confidence", f"{df['confidence'].mean():.2%}")
            with col2:
                st.metric("✅ Resolution Rate", f"{(df['resolved'].sum() / len(df)):.2%}")
                st.metric("📚 Avg Sources", f"{df['sources_count'].mean():.1f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Intent distribution
            st.markdown("""
                <h3 style='color: #2900C8 !important; margin: 24px 0 16px 0; font-size: 18px;'>
                    Query Types Distribution
                </h3>
            """, unsafe_allow_html=True)
            
            intent_counts = df['intent'].value_counts()
            
            # Custom colors for pie chart (dark blue theme)
            fig = px.pie(
                values=intent_counts.values, 
                names=intent_counts.index,
                color_discrete_sequence=['#2900C8', '#5E4BD4', '#7E6FD4', '#9E8FE4', '#1F0096']
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
                font=dict(color='#2900C8', size=13, family='Inter'),
                showlegend=True,
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#D4CFEF', borderwidth=2)
            )
            fig.update_traces(textfont=dict(color='white', size=14, family='Inter'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Export analytics button
            if st.button("📥 Export Analytics", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV", 
                    csv, 
                    "analytics.csv", 
                    "text/csv",
                    use_container_width=True
                )
    
    # Main chat interface
    if not api_key:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); 
                        padding: 24px; border-radius: 16px; 
                        border-left: 6px solid #FF9800; margin: 24px 0;
                        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);'>
                <h3 style='color: #E65100 !important; margin-top: 0; font-size: 22px;'>
                    ⚠️ API Key Required
                </h3>
                <p style='color: #E65100 !important; margin-bottom: 0; font-size: 16px; line-height: 1.6;'>
                    Please enter your Google Gemini API key in the sidebar to start using the chatbot.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, #E8E4F9 0%, #D4CFEF 100%); 
                        padding: 24px; border-radius: 16px; 
                        border-left: 6px solid #2900C8;
                        box-shadow: 0 4px 12px rgba(41, 0, 200, 0.2);'>
                <h4 style='color: #2900C8 !important; margin-top: 0; font-size: 20px;'>
                    🔑 How to get a FREE API key:
                </h4>
                <ol style='color: #2900C8 !important; font-size: 16px; line-height: 1.8; margin-bottom: 0;'>
                    <li>Visit <a href='https://makersuite.google.com/app/apikey' 
                        target='_blank' style='color: #2900C8 !important; font-weight: 600; 
                        text-decoration: underline;'>Google AI Studio</a></li>
                    <li>Sign in with your Google account</li>
                    <li>Click 'Create API Key' button</li>
                    <li>Copy the key and paste it in the sidebar</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.knowledge_base:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); 
                        padding: 24px; border-radius: 16px; 
                        border-left: 6px solid #F44336; margin: 24px 0;
                        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);'>
                <h3 style='color: #C62828 !important; margin-top: 0; font-size: 22px;'>
                    ❌ Initialization Error
                </h3>
                <p style='color: #C62828 !important; margin-bottom: 0; font-size: 16px; line-height: 1.6;'>
                    Failed to initialize knowledge base. Please check your API key and try again.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"<div style='color: #2C3E50; font-size: 15px; line-height: 1.6;'>{message['content']}</div>", 
                       unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #F8FBFF 0%, #E8E4F9 100%); 
                                        padding: 16px; border-radius: 12px; margin: 12px 0; 
                                        border-left: 4px solid #5E4BD4;
                                        box-shadow: 0 2px 8px rgba(94, 75, 212, 0.15);'>
                                <strong style='color: #2900C8 !important; font-size: 16px;'>
                                    📄 Source {i}: {source['name']}
                                </strong>
                                <span style='color: #5E4BD4 !important; font-weight: 600; margin-left: 8px;'>
                                    (Relevance: {source['score']:.0%})
                                </span>
                                <p style='color: #424242 !important; margin-top: 12px; 
                                          font-size: 14px; line-height: 1.6;'>
                                    {source['content'][:200]}...
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if query := st.chat_input("💬 Ask me about HR policies, leave requests, expenses, or ERP help..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(f"<div style='color: #1A1A1A; font-size: 15px; line-height: 1.6;'>{query}</div>", 
                       unsafe_allow_html=True)
        
        # Detect intent
        intent = detect_intent(query)
        
        # Handle structured workflows
        if intent == 'leave_request':
            with st.chat_message("assistant"):
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #E8E4F9 0%, #D4CFEF 100%); 
                                padding: 18px; border-radius: 12px; 
                                margin-bottom: 20px; border-left: 5px solid #2900C8;
                                box-shadow: 0 3px 10px rgba(41, 0, 200, 0.15);'>
                        <p style='color: #2900C8 !important; margin: 0; font-weight: 600; 
                                  font-size: 16px;'>
                            ✅ I can help you submit a leave request. Let me open the form for you.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                leave_request_form()
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Leave request form opened. Please fill in the details above."
            })
            
            log_analytics(query, intent, "Form opened", 1.0, [])
            
        elif intent == 'expense_submission':
            with st.chat_message("assistant"):
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #E8E4F9 0%, #D4CFEF 100%); 
                                padding: 18px; border-radius: 12px; 
                                margin-bottom: 20px; border-left: 5px solid #2900C8;
                                box-shadow: 0 3px 10px rgba(41, 0, 200, 0.15);'>
                        <p style='color: #2900C8 !important; margin: 0; font-weight: 600; 
                                  font-size: 16px;'>
                            💰 I'll help you submit an expense. Here's the form:
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                expense_submission_form()
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Expense submission form opened. Please complete the form above."
            })
            
            log_analytics(query, intent, "Form opened", 1.0, [])
            
        else:
            # General RAG query
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base..."):
                    result = query_with_rag(query, st.session_state.knowledge_base, api_key)
                    
                    if result:
                        answer = result['answer']
                        sources = result['sources']
                        confidence = result['confidence']
                        
                        st.markdown(f"<div style='color: #2C3E50; font-size: 15px; line-height: 1.6;'>{answer}</div>", 
                                   unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        log_analytics(query, intent, answer, confidence, sources)
                    else:
                        st.markdown("""
                            <div style='background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); 
                                        padding: 18px; border-radius: 12px; 
                                        border-left: 5px solid #F44336;
                                        box-shadow: 0 3px 10px rgba(244, 67, 54, 0.15);'>
                                <p style='color: #C62828 !important; margin: 0; font-weight: 600; 
                                          font-size: 16px;'>
                                    ❌ Sorry, I encountered an error. Please try again.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


