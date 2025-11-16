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

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(page_title="Konecta ERP Chatbot", page_icon="💼", layout="wide")

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
SAMPLE_DOCUMENTS = {
    "HR_Leave_Policy": """
    KONECTA LEAVE POLICY
    
    Annual Leave: All employees are entitled to 21 days of annual leave per year.
    Leave must be requested at least 2 weeks in advance through the ERP system.
    
    Sick Leave: Employees can take up to 10 days of sick leave per year.
    Medical certificate required for absences exceeding 3 consecutive days.
    
    Emergency Leave: Up to 5 days per year for family emergencies.
    Requires manager approval within 24 hours.
    
    Leave Request Process:
    1. Submit leave request through ERP chatbot or portal
    2. Manager receives notification
    3. Approval/rejection within 48 hours
    4. HR updates leave balance automatically
    """,
    
    "Finance_Expense_Policy": """
    KONECTA EXPENSE REIMBURSEMENT POLICY
    
    Eligible Expenses:
    - Travel: Flight, hotel, transportation (with receipts)
    - Meals: Up to $50/day for business travel
    - Office Supplies: Pre-approved purchases only
    
    Submission Process:
    1. Upload receipts to ERP system
    2. Fill expense form with justification
    3. Finance team reviews within 5 business days
    4. Reimbursement processed in next payroll cycle
    
    Non-reimbursable: Personal expenses, alcohol, entertainment without prior approval.
    """,
    
    "ERP_User_Guide": """
    KONECTA ERP SYSTEM GUIDE
    
    Login: Use your company email and password. MFA required.
    
    Dashboard Access:
    - HR Module: View payslips, request leave, update personal info
    - Finance Module: Submit expenses, view budget reports
    - Analytics: Real-time KPI dashboards
    
    Common Issues:
    - Password reset: Contact IT helpdesk
    - Dashboard not loading: Clear browser cache
    - Missing data: Contact your department admin
    
    Support: erp-support@konecta.com or use chatbot for instant help.
    """,
    
    "HR_Performance_Review": """
    KONECTA PERFORMANCE REVIEW PROCESS
    
    Review Cycle: Bi-annual (June and December)
    
    Process:
    1. Self-assessment submission (2 weeks before review)
    2. Manager evaluation
    3. 1-on-1 review meeting
    4. Goal setting for next period
    
    Criteria:
    - Job Performance (40%)
    - Initiative & Innovation (30%)
    - Teamwork & Collaboration (20%)
    - Attendance & Punctuality (10%)
    
    Ratings: Outstanding, Exceeds Expectations, Meets Expectations, Needs Improvement
    """
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
- If the answer isn't in the context, say "I don't have that information in the knowledge base"
- Be concise and helpful
- Cite which document you're referencing when relevant
-you may also converse with the user normally but not fabricate information.

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
    # Enhanced Custom CSS for white and blue theme with better contrast
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main background - Light blue-grey */
        .stApp {
            background: linear-gradient(135deg, #F0F4F8 0%, #E8F1F8 100%);
        }
        
        /* Remove dark theme defaults */
        .stApp, .stApp > header {
            background-color: transparent !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 3px solid #2196F3;
            box-shadow: 2px 0 10px rgba(33, 150, 243, 0.1);
        }
        
        [data-testid="stSidebar"] * {
            color: #1A1A1A !important;
        }
        
        /* Override all text colors to be dark */
        .stApp p, .stApp span, .stApp div, .stApp label {
            color: #2C3E50 !important;
        }
        
        /* Title styling */
        h1 {
            color: #1565C0 !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 4px rgba(21, 101, 192, 0.1);
        }
        
        /* Headers */
        h2 {
            color: #1976D2 !important;
            font-weight: 600 !important;
        }
        
        h3 {
            color: #2196F3 !important;
            font-weight: 600 !important;
        }
        
        /* Chat messages container */
        .stChatMessage {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            padding: 20px !important;
            margin: 12px 0 !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08) !important;
            border: 1px solid #E3F2FD !important;
        }
        
        /* User message - Blue background */
        [data-testid="stChatMessageContent"] {
            color: #1A1A1A !important;
        }
        
        .stChatMessage[data-testid*="user"] {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
            border-left: 5px solid #2196F3 !important;
        }
        
        /* Assistant message - White background */
        .stChatMessage[data-testid*="assistant"] {
            background-color: #FFFFFF !important;
            border-left: 5px solid #64B5F6 !important;
        }
        
        /* Chat input container */
        .stChatInputContainer {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            border: 2px solid #2196F3 !important;
            padding: 8px !important;
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15) !important;
        }
        
        .stChatInputContainer textarea {
            color: #1A1A1A !important;
            font-size: 15px !important;
        }
        
        .stChatInputContainer textarea::placeholder {
            color: #64B5F6 !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 10px rgba(33, 150, 243, 0.3) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%) !important;
            box-shadow: 0 6px 16px rgba(33, 150, 243, 0.4) !important;
            transform: translateY(-2px);
        }
        
        /* Form submit button */
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            width: 100% !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #1565C0 !important;
            font-weight: 700 !important;
            font-size: 28px !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #2C3E50 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        [data-testid="stMetricDelta"] {
            color: #424242 !important;
        }
        
        /* Info/Alert boxes */
        .stAlert {
            background-color: #E3F2FD !important;
            border-left: 5px solid #2196F3 !important;
            border-radius: 12px !important;
            color: #1565C0 !important;
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
            border: 2px solid #BBDEFB !important;
            color: #1976D2 !important;
            font-weight: 600 !important;
            padding: 12px !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #E3F2FD !important;
            border-color: #2196F3 !important;
        }
        
        .streamlit-expanderContent {
            background-color: #F8FBFF !important;
            border: 2px solid #E3F2FD !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
            padding: 16px !important;
        }
        
        /* Form styling */
        [data-testid="stForm"] {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            padding: 24px !important;
            border: 2px solid #2196F3 !important;
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15) !important;
        }
        
        /* Text input */
        .stTextInput > div > div > input {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #BBDEFB !important;
            color: #1A1A1A !important;
            font-size: 15px !important;
            padding: 12px !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #2196F3 !important;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1) !important;
        }
        
        .stTextInput label {
            color: #1976D2 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #BBDEFB !important;
        }
        
        .stSelectbox label {
            color: #1976D2 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Date input */
        .stDateInput > div > div > input {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #BBDEFB !important;
            color: #1A1A1A !important;
        }
        
        .stDateInput label {
            color: #1976D2 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Text area */
        .stTextArea > div > div > textarea {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #BBDEFB !important;
            color: #1A1A1A !important;
            font-size: 15px !important;
        }
        
        .stTextArea label {
            color: #1976D2 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Number input */
        .stNumberInput > div > div > input {
            background-color: #F8FBFF !important;
            border-radius: 10px !important;
            border: 2px solid #BBDEFB !important;
            color: #1A1A1A !important;
        }
        
        .stNumberInput label {
            color: #1976D2 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* File uploader */
        .stFileUploader label {
            color: #1976D2 !important;
            font-weight: 600 !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #2196F3 !important;
        }
        
        /* Divider */
        hr {
            border-color: #BBDEFB !important;
            margin: 24px 0 !important;
        }
        
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
        }
        
        /* Markdown text */
        .stMarkdown {
            color: #2C3E50 !important;
        }
        
        /* Remove default dark theme for main content */
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Header area */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* Footer */
        footer {
            background-color: transparent !important;
            color: #64B5F6 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with icon and better styling
    st.markdown("""
        <div style='text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #FFFFFF 0%, #E3F2FD 100%); 
                    border-radius: 20px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(33, 150, 243, 0.15);'>
            <h1 style='margin: 0; color: #1565C0; font-size: 42px; font-weight: 700;'>
                💼 Konecta ERP Chatbot
            </h1>
            <p style='color: #2196F3; font-size: 18px; margin-top: 12px; font-weight: 500;'>
                AI-Powered Assistant with RAG + Structured Workflows
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0; margin-bottom: 24px; 
                        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                        border-radius: 12px; box-shadow: 0 4px 10px rgba(33, 150, 243, 0.3);'>
                <h2 style='color: #FFFFFF !important; margin: 0; font-size: 24px;'>⚙️ Configuration</h2>
            </div>
        """, unsafe_allow_html=True)
    
        # API Key section with better styling
        st.markdown("""
            <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                        padding: 18px; border-radius: 12px; margin-bottom: 24px; 
                        border: 2px solid #2196F3;'>
                <p style='margin: 0; color: #1565C0 !important; font-weight: 700; font-size: 15px;'>
                    🆓 Using FREE Google Gemini API
                </p>
            </div>
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
        
        st.markdown("<hr style='margin: 30px 0; border: 2px solid #2196F3;'>", unsafe_allow_html=True)
        
        # Analytics Dashboard
        st.markdown("""
            <div style='text-align: center; padding: 20px 0; margin-bottom: 24px; 
                        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); 
                        border-radius: 12px; box-shadow: 0 4px 10px rgba(33, 150, 243, 0.3);'>
                <h2 style='color: #FFFFFF !important; margin: 0; font-size: 24px;'>📊 Analytics Dashboard</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.analytics:
            df = pd.DataFrame(st.session_state.analytics)
            
            # Key metrics with cards
            st.markdown("""
                <div style='background-color: #FFFFFF; padding: 20px; border-radius: 12px; 
                            border: 2px solid #2196F3; margin-bottom: 20px; 
                            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);'>
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
                <h3 style='color: #1976D2 !important; margin: 24px 0 16px 0; font-size: 18px;'>
                    📊 Query Types Distribution
                </h3>
            """, unsafe_allow_html=True)
            
            intent_counts = df['intent'].value_counts()
            
            # Custom colors for pie chart (blue theme)
            fig = px.pie(
                values=intent_counts.values, 
                names=intent_counts.index,
                color_discrete_sequence=['#2196F3', '#64B5F6', '#90CAF9', '#BBDEFB', '#1976D2']
            )
            fig.update_layout(
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
                font=dict(color='#1565C0', size=13, family='Inter'),
                showlegend=True,
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#BBDEFB', borderwidth=2)
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
        else:
            st.markdown("""
                <div style='background-color: #E3F2FD; padding: 16px; border-radius: 10px; 
                            text-align: center; border: 2px solid #BBDEFB;'>
                    <p style='color: #1976D2 !important; margin: 0; font-weight: 600;'>
                        📊 No analytics data yet. Start chatting to see insights!
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
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
            <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                        padding: 24px; border-radius: 16px; 
                        border-left: 6px solid #2196F3;
                        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);'>
                <h4 style='color: #1565C0 !important; margin-top: 0; font-size: 20px;'>
                    🔑 How to get a FREE API key:
                </h4>
                <ol style='color: #1976D2 !important; font-size: 16px; line-height: 1.8; margin-bottom: 0;'>
                    <li>Visit <a href='https://makersuite.google.com/app/apikey' 
                        target='_blank' style='color: #2196F3 !important; font-weight: 600; 
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
                            <div style='background: linear-gradient(135deg, #F8FBFF 0%, #E3F2FD 100%); 
                                        padding: 16px; border-radius: 12px; margin: 12px 0; 
                                        border-left: 4px solid #64B5F6;
                                        box-shadow: 0 2px 8px rgba(100, 181, 246, 0.15);'>
                                <strong style='color: #1976D2 !important; font-size: 16px;'>
                                    📄 Source {i}: {source['name']}
                                </strong>
                                <span style='color: #64B5F6 !important; font-weight: 600; margin-left: 8px;'>
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
                    <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                                padding: 18px; border-radius: 12px; 
                                margin-bottom: 20px; border-left: 5px solid #2196F3;
                                box-shadow: 0 3px 10px rgba(33, 150, 243, 0.15);'>
                        <p style='color: #1565C0 !important; margin: 0; font-weight: 600; 
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
                    <div style='background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                                padding: 18px; border-radius: 12px; 
                                margin-bottom: 20px; border-left: 5px solid #2196F3;
                                box-shadow: 0 3px 10px rgba(33, 150, 243, 0.15);'>
                        <p style='color: #1565C0 !important; margin: 0; font-weight: 600; 
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