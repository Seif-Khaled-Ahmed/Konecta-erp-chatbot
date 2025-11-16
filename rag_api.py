"""
ERP RAG Chatbot API
REST API for RAG system with structured workflows and analytics
Uses Google Gemini API (Free tier: 60 requests/minute)

Installation:
pip install fastapi uvicorn google-generativeai pandas pydantic python-multipart

Usage:
uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000

API Documentation: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import google.generativeai as genai
import json
import re
from enum import Enum

# ============================================
# CONFIGURATION
# ============================================
app = FastAPI(
    title="Konecta ERP RAG API",
    description="AI-Powered RAG system for ERP assistance with structured workflows",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (In production, use Redis or database)
KNOWLEDGE_BASE = None
ANALYTICS_STORE = []

# ============================================
# SAMPLE ERP DOCUMENTS
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
# PYDANTIC MODELS
# ============================================
class IntentType(str, Enum):
    LEAVE_REQUEST = "leave_request"
    EXPENSE_SUBMISSION = "expense_submission"
    GENERAL_QUERY = "general_query"

class LeaveType(str, Enum):
    ANNUAL = "Annual Leave"
    SICK = "Sick Leave"
    EMERGENCY = "Emergency Leave"

class ExpenseType(str, Enum):
    TRAVEL = "Travel"
    MEALS = "Meals"
    OFFICE_SUPPLIES = "Office Supplies"
    OTHER = "Other"

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    api_key: str = Field(..., description="Google Gemini API key")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")

class DocumentSource(BaseModel):
    name: str
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    intent: IntentType
    confidence: float
    sources: List[DocumentSource]
    timestamp: datetime
    session_id: Optional[str] = None

class LeaveRequest(BaseModel):
    leave_type: LeaveType
    start_date: date
    end_date: date
    reason: str
    api_key: str

class LeaveResponse(BaseModel):
    status: str
    message: str
    request_id: str
    leave_type: str
    start_date: date
    end_date: date
    timestamp: datetime

class ExpenseRequest(BaseModel):
    expense_type: ExpenseType
    amount: float = Field(..., gt=0, description="Expense amount in USD")
    description: str
    api_key: str
    receipt_filename: Optional[str] = None

class ExpenseResponse(BaseModel):
    status: str
    message: str
    request_id: str
    expense_type: str
    amount: float
    timestamp: datetime

class AnalyticsQuery(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    intent_filter: Optional[IntentType] = None

class AnalyticsResponse(BaseModel):
    total_queries: int
    avg_confidence: float
    resolution_rate: float
    avg_sources: float
    intent_distribution: Dict[str, int]
    queries: List[Dict[str, Any]]

class ConfigRequest(BaseModel):
    api_key: str

class ConfigResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int

# ============================================
# SIMPLE KNOWLEDGE BASE
# ============================================
class SimpleKnowledgeBase:
    """Simple keyword-based retrieval (no embeddings required)"""
    
    def __init__(self, documents):
        self.documents = documents
        self.doc_text = "\n\n".join([f"Document: {name}\n{content}" 
                                     for name, content in documents.items()])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
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
# CORE FUNCTIONS
# ============================================
def initialize_gemini(api_key: str) -> bool:
    """Initialize Gemini API"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Gemini: {str(e)}")

def detect_intent(query: str) -> IntentType:
    """Detect if query requires structured workflow"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['apply', 'request', 'submit']) and 'leave' in query_lower:
        return IntentType.LEAVE_REQUEST
    elif any(word in query_lower for word in ['expense', 'reimbursement', 'receipt']):
        return IntentType.EXPENSE_SUBMISSION
    else:
        return IntentType.GENERAL_QUERY

def query_with_rag(query: str, knowledge_base: SimpleKnowledgeBase, api_key: str) -> Dict[str, Any]:
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
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""You are a helpful ERP assistant for Konecta. Answer the user's question based on the provided context.

Context from knowledge base:
{context}

User Question: {query}

Instructions:
- Answer based on the context provided
- If the answer isn't in the context, say "I don't have that information in the knowledge base"
- Be concise and helpful
- Cite which document you're referencing when relevant

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
        raise HTTPException(status_code=500, detail=f"Error querying Gemini: {str(e)}")

def log_analytics(query: str, intent: IntentType, confidence: float, sources: List[Dict], session_id: Optional[str] = None):
    """Log interaction for analytics"""
    ANALYTICS_STORE.append({
        'timestamp': datetime.now(),
        'query': query,
        'intent': intent.value,
        'confidence': confidence,
        'sources_count': len(sources) if sources else 0,
        'resolved': confidence > 0.5,
        'session_id': session_id
    })

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Konecta ERP RAG API",
        "version": "1.0.0",
        "timestamp": datetime.now()
    }

@app.post("/api/configure", response_model=ConfigResponse)
async def configure_system(config: ConfigRequest):
    """Initialize the knowledge base with API key"""
    global KNOWLEDGE_BASE
    
    try:
        initialize_gemini(config.api_key)
        KNOWLEDGE_BASE = SimpleKnowledgeBase(SAMPLE_DOCUMENTS)
        
        return ConfigResponse(
            status="success",
            message="Knowledge base initialized successfully",
            documents_loaded=len(SAMPLE_DOCUMENTS)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """Main query endpoint for RAG chatbot"""
    global KNOWLEDGE_BASE
    
    # Initialize if needed
    if KNOWLEDGE_BASE is None:
        initialize_gemini(request.api_key)
        KNOWLEDGE_BASE = SimpleKnowledgeBase(SAMPLE_DOCUMENTS)
    
    # Detect intent
    intent = detect_intent(request.query)
    
    # For structured workflows, return intent with guidance
    if intent in [IntentType.LEAVE_REQUEST, IntentType.EXPENSE_SUBMISSION]:
        answer = f"I detected you want to submit a {intent.value.replace('_', ' ')}. Please use the dedicated endpoint: /api/{intent.value}"
        sources = []
        confidence = 1.0
    else:
        # General RAG query
        result = query_with_rag(request.query, KNOWLEDGE_BASE, request.api_key)
        answer = result['answer']
        sources = result['sources']
        confidence = result['confidence']
    
    # Log analytics
    log_analytics(request.query, intent, confidence, sources, request.session_id)
    
    return QueryResponse(
        answer=answer,
        intent=intent,
        confidence=confidence,
        sources=[DocumentSource(**source) for source in sources],
        timestamp=datetime.now(),
        session_id=request.session_id
    )

@app.post("/api/leave_request", response_model=LeaveResponse)
async def submit_leave_request(request: LeaveRequest):
    """Structured endpoint for leave requests"""
    # Validate dates
    if request.end_date < request.start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")
    
    # Generate request ID
    request_id = f"LR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Log analytics
    log_analytics(
        f"Leave request: {request.leave_type}",
        IntentType.LEAVE_REQUEST,
        1.0,
        []
    )
    
    return LeaveResponse(
        status="submitted",
        message="Leave request submitted successfully. Your manager will be notified and you'll receive approval within 48 hours.",
        request_id=request_id,
        leave_type=request.leave_type.value,
        start_date=request.start_date,
        end_date=request.end_date,
        timestamp=datetime.now()
    )

@app.post("/api/expense_submission", response_model=ExpenseResponse)
async def submit_expense(request: ExpenseRequest):
    """Structured endpoint for expense submissions"""
    # Validate amount
    if request.expense_type == ExpenseType.MEALS and request.amount > 50:
        raise HTTPException(
            status_code=400,
            detail="Meal expenses are limited to $50/day according to policy"
        )
    
    # Generate request ID
    request_id = f"EX-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Log analytics
    log_analytics(
        f"Expense submission: {request.expense_type}",
        IntentType.EXPENSE_SUBMISSION,
        1.0,
        []
    )
    
    return ExpenseResponse(
        status="submitted",
        message="Expense submitted successfully. Finance team will review within 5 business days.",
        request_id=request_id,
        expense_type=request.expense_type.value,
        amount=request.amount,
        timestamp=datetime.now()
    )

@app.post("/api/expense_submission/upload")
async def upload_receipt(
    file: UploadFile = File(...),
    expense_type: str = None,
    amount: float = None,
    description: str = None
):
    """Upload receipt for expense submission"""
    # Validate file type
    allowed_types = ['application/pdf', 'image/jpeg', 'image/png']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF, JPG, and PNG are allowed"
        )
    
    # In production, save to cloud storage (S3, GCS, etc.)
    file_location = f"/tmp/receipts/{file.filename}"
    
    return {
        "status": "uploaded",
        "filename": file.filename,
        "message": "Receipt uploaded successfully",
        "file_size": file.size if hasattr(file, 'size') else None
    }

@app.post("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics(query: AnalyticsQuery = None):
    """Get analytics data"""
    if not ANALYTICS_STORE:
        return AnalyticsResponse(
            total_queries=0,
            avg_confidence=0.0,
            resolution_rate=0.0,
            avg_sources=0.0,
            intent_distribution={},
            queries=[]
        )
    
    # Filter by date range if provided
    filtered_data = ANALYTICS_STORE
    if query and query.start_date:
        filtered_data = [a for a in filtered_data if a['timestamp'] >= query.start_date]
    if query and query.end_date:
        filtered_data = [a for a in filtered_data if a['timestamp'] <= query.end_date]
    if query and query.intent_filter:
        filtered_data = [a for a in filtered_data if a['intent'] == query.intent_filter.value]
    
    if not filtered_data:
        return AnalyticsResponse(
            total_queries=0,
            avg_confidence=0.0,
            resolution_rate=0.0,
            avg_sources=0.0,
            intent_distribution={},
            queries=[]
        )
    
    # Calculate metrics
    total = len(filtered_data)
    avg_conf = sum(a['confidence'] for a in filtered_data) / total
    resolution_rate = sum(a['resolved'] for a in filtered_data) / total
    avg_sources = sum(a['sources_count'] for a in filtered_data) / total
    
    # Intent distribution
    intent_dist = {}
    for item in filtered_data:
        intent = item['intent']
        intent_dist[intent] = intent_dist.get(intent, 0) + 1
    
    return AnalyticsResponse(
        total_queries=total,
        avg_confidence=avg_conf,
        resolution_rate=resolution_rate,
        avg_sources=avg_sources,
        intent_distribution=intent_dist,
        queries=filtered_data
    )

@app.get("/api/analytics/export")
async def export_analytics():
    """Export analytics as JSON"""
    return {
        "status": "success",
        "count": len(ANALYTICS_STORE),
        "data": ANALYTICS_STORE
    }

@app.delete("/api/analytics")
async def clear_analytics():
    """Clear analytics data"""
    global ANALYTICS_STORE
    count = len(ANALYTICS_STORE)
    ANALYTICS_STORE = []
    return {
        "status": "success",
        "message": f"Cleared {count} analytics records"
    }

@app.get("/api/documents")
async def list_documents():
    """List available documents in knowledge base"""
    return {
        "status": "success",
        "count": len(SAMPLE_DOCUMENTS),
        "documents": list(SAMPLE_DOCUMENTS.keys())
    }

@app.get("/api/documents/{doc_name}")
async def get_document(doc_name: str):
    """Retrieve a specific document"""
    if doc_name not in SAMPLE_DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "status": "success",
        "name": doc_name,
        "content": SAMPLE_DOCUMENTS[doc_name]
    }

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
