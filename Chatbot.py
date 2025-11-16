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
    st.title("💼 Konecta ERP Chatbot")
    st.markdown("**AI-Powered Assistant with RAG + Structured Workflows**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
    
        st.info("🆓 **Using FREE Google Gemini API**")    
        # Set default API key
        default_api_key = "AIzaSyClemJJjKHTuEIMtXy1-WJBJZkGmZK2Pzw"
    
        api_key = st.text_input("Google Gemini API Key", 
                            value=default_api_key,
                            type="password")
    
    if api_key and not st.session_state.knowledge_base:
        with st.spinner("Initializing knowledge base..."):
            if initialize_gemini(api_key):
                st.session_state.knowledge_base = SimpleKnowledgeBase(SAMPLE_DOCUMENTS)
                st.success("✅ Knowledge base loaded!")
        st.markdown("---")
        st.header("📊 Analytics Dashboard")
        
        if st.session_state.analytics:
            df = pd.DataFrame(st.session_state.analytics)
            
            # Key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", len(df))
                st.metric("Avg Confidence", f"{df['confidence'].mean():.2%}")
            with col2:
                st.metric("Resolution Rate", f"{(df['resolved'].sum() / len(df)):.2%}")
                st.metric("Avg Sources", f"{df['sources_count'].mean():.1f}")
            
            # Intent distribution
            st.subheader("Query Types")
            intent_counts = df['intent'].value_counts()
            fig = px.pie(values=intent_counts.values, names=intent_counts.index)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export analytics
            if st.button("📥 Export Analytics"):
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "analytics.csv", "text/csv")
    
    # Main chat interface
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to start.")
        st.info("**How to get a FREE API key:**\n1. Visit https://makersuite.google.com/app/apikey\n2. Sign in with Google\n3. Click 'Create API Key'\n4. Copy and paste here")
        return
    
    if not st.session_state.knowledge_base:
        st.error("Failed to initialize knowledge base. Check your API key.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}: {source['name']}** (Relevance: {source['score']:.0%})")
                        st.text(source['content'][:200] + "...")
    
    # Chat input
    if query := st.chat_input("Ask me about HR policies, leave requests, expenses, or ERP help..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Detect intent
        intent = detect_intent(query)
        
        # Handle structured workflows
        if intent == 'leave_request':
            with st.chat_message("assistant"):
                st.markdown("I can help you submit a leave request. Let me open the form for you.")
                leave_request_form()
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Leave request form opened. Please fill in the details above."
            })
            
            log_analytics(query, intent, "Form opened", 1.0, [])
            
        elif intent == 'expense_submission':
            with st.chat_message("assistant"):
                st.markdown("I'll help you submit an expense. Here's the form:")
                expense_submission_form()
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Expense submission form opened. Please complete the form above."
            })
            
            log_analytics(query, intent, "Form opened", 1.0, [])
            
        else:
            # General RAG query
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    result = query_with_rag(query, st.session_state.knowledge_base, api_key)
                    
                    if result:
                        answer = result['answer']
                        sources = result['sources']
                        confidence = result['confidence']
                        
                        st.markdown(answer)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        log_analytics(query, intent, answer, confidence, sources)
                    else:
                        st.error("Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    main()