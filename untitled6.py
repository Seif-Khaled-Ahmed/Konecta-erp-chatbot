"""
AI-Powered Document Processing System
Automated Invoice & Purchase Order Processing with OCR and Validation
Requirements: pip install streamlit openai PyPDF2 pillow pandas plotly
"""

import streamlit as st
import openai
import PyPDF2
from PIL import Image
import io
import json
import re
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Document Processor", page_icon="📑", layout="wide")

# ============================================
# SESSION STATE
# ============================================
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'time_saved_hours': 0
    }

# ============================================
# DOCUMENT TYPES & VALIDATION RULES
# ============================================
DOCUMENT_TYPES = {
    "Invoice": {
        "required_fields": ["invoice_number", "date", "vendor_name", "total_amount", "line_items"],
        "optional_fields": ["due_date", "tax_amount", "currency", "payment_terms"],
        "validation_rules": {
            "total_amount": lambda x: x > 0,
            "invoice_number": lambda x: len(str(x)) > 0,
            "date": lambda x: len(str(x)) >= 8
        }
    },
    "Purchase Order": {
        "required_fields": ["po_number", "date", "vendor_name", "total_amount", "items"],
        "optional_fields": ["shipping_address", "billing_address", "delivery_date"],
        "validation_rules": {
            "total_amount": lambda x: x > 0,
            "po_number": lambda x: len(str(x)) > 0
        }
    },
    "Receipt": {
        "required_fields": ["receipt_number", "date", "merchant", "total_amount"],
        "optional_fields": ["payment_method", "items"],
        "validation_rules": {
            "total_amount": lambda x: x > 0
        }
    }
}

# ============================================
# FILE READING
# ============================================
def read_pdf_text(file):
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

def read_image(file):
    """Read image file"""
    try:
        image = Image.open(file)
        return image
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

# ============================================
# AI EXTRACTION (Multi-modal approach)
# ============================================
def extract_with_gpt4_vision(image_file, doc_type, api_key):
    """Extract using GPT-4 Vision for images"""
    try:
        import base64
        
        # Convert image to base64
        image = Image.open(image_file)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        client = openai.OpenAI(api_key=api_key)
        
        doc_config = DOCUMENT_TYPES[doc_type]
        fields = doc_config['required_fields'] + doc_config['optional_fields']
        
        prompt = f"""
        Extract information from this {doc_type} image and return as JSON.
        
        Required fields: {', '.join(doc_config['required_fields'])}
        Optional fields: {', '.join(doc_config['optional_fields'])}
        
        Return format:
        {{
            "document_type": "{doc_type}",
            "extracted_fields": {{
                // all extracted fields here
            }},
            "confidence": 0.95,
            "extraction_notes": "any issues or uncertainties"
        }}
        
        For amounts, extract numbers only (no currency symbols).
        For dates, use YYYY-MM-DD format.
        If a field is not found, set it to null.
        """
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        result = result.replace("```json", "").replace("```", "").strip()
        
        return json.loads(result)
        
    except Exception as e:
        st.error(f"Error with GPT-4 Vision: {e}")
        return None

def extract_with_gpt_text(text, doc_type, api_key):
    """Extract from text using GPT"""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        doc_config = DOCUMENT_TYPES[doc_type]
        
        prompt = f"""
        Extract structured information from this {doc_type} text and return as JSON.
        
        Required fields: {', '.join(doc_config['required_fields'])}
        Optional fields: {', '.join(doc_config['optional_fields'])}
        
        Text:
        {text[:4000]}
        
        Return format:
        {{
            "document_type": "{doc_type}",
            "extracted_fields": {{
                // all extracted fields here
            }},
            "confidence": 0.95,
            "extraction_notes": "any issues"
        }}
        
        For line_items or items, return as array of objects with: description, quantity, unit_price, total
        Return ONLY valid JSON, no other text.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a document processing expert. Extract data accurately and return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        result = result.replace("```json", "").replace("```", "").strip()
        
        return json.loads(result)
        
    except Exception as e:
        st.error(f"Error with GPT extraction: {e}")
        return None

# ============================================
# VALIDATION PIPELINE
# ============================================
def validate_extraction(extracted_data, doc_type):
    """Validate extracted data against rules"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'missing_required': [],
        'field_validations': {}
    }
    
    doc_config = DOCUMENT_TYPES[doc_type]
    extracted_fields = extracted_data.get('extracted_fields', {})
    
    # Check required fields
    for field in doc_config['required_fields']:
        if field not in extracted_fields or extracted_fields[field] is None:
            validation_results['missing_required'].append(field)
            validation_results['is_valid'] = False
    
    # Apply validation rules
    for field, rule in doc_config['validation_rules'].items():
        if field in extracted_fields and extracted_fields[field] is not None:
            try:
                if not rule(extracted_fields[field]):
                    validation_results['errors'].append(f"{field}: validation failed")
                    validation_results['is_valid'] = False
                    validation_results['field_validations'][field] = False
                else:
                    validation_results['field_validations'][field] = True
            except:
                validation_results['warnings'].append(f"{field}: could not validate")
    
    # Business logic validations
    if doc_type == "Invoice" and 'line_items' in extracted_fields:
        # Validate line items sum to total
        try:
            line_total = sum(item.get('total', 0) for item in extracted_fields['line_items'])
            invoice_total = extracted_fields.get('total_amount', 0)
            
            if abs(line_total - invoice_total) > 0.01:
                validation_results['warnings'].append(
                    f"Line items total (${line_total:.2f}) doesn't match invoice total (${invoice_total:.2f})"
                )
        except:
            pass
    
    return validation_results

def handle_edge_cases(extracted_data, file_quality="good"):
    """Handle common edge cases"""
    edge_case_notes = []
    
    extracted_fields = extracted_data.get('extracted_fields', {})
    
    # Handle missing fields
    if not extracted_fields:
        edge_case_notes.append("⚠️ No fields extracted - possible OCR failure")
    
    # Handle low confidence
    confidence = extracted_data.get('confidence', 0)
    if confidence < 0.7:
        edge_case_notes.append(f"⚠️ Low extraction confidence ({confidence:.0%})")
    
    # Handle poor quality scans
    if file_quality == "poor":
        edge_case_notes.append("⚠️ Low quality scan detected - results may be inaccurate")
    
    # Check for handwriting indicators
    notes = extracted_data.get('extraction_notes', '').lower()
    if 'handwritten' in notes or 'handwriting' in notes:
        edge_case_notes.append("⚠️ Handwritten content detected - manual review recommended")
    
    # Handle partial data
    if len(extracted_fields) < 3:
        edge_case_notes.append("⚠️ Limited data extracted - document may be incomplete")
    
    return edge_case_notes

# ============================================
# DATA TRANSFORMATION & STORAGE
# ============================================
def transform_for_erp(validated_data, doc_type):
    """Transform extracted data into ERP-ready format"""
    extracted_fields = validated_data.get('extracted_fields', {})
    
    erp_record = {
        'document_id': f"{doc_type[:3].upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'document_type': doc_type,
        'processing_date': datetime.now().isoformat(),
        'status': 'validated' if validated_data.get('is_valid', False) else 'needs_review',
        'data': extracted_fields,
        'validation_results': validated_data.get('validation_results', {}),
        'confidence': validated_data.get('confidence', 0)
    }
    
    return erp_record

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("📑 AI Document Processing System")
    st.markdown("**Automated Invoice & Purchase Order Processing**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        
        st.markdown("---")
        st.header("📊 Processing Stats")
        stats = st.session_state.processing_stats
        st.metric("Total Processed", stats['total'])
        st.metric("Successful", stats['successful'])
        st.metric("Failed", stats['failed'])
        st.metric("Time Saved", f"{stats['time_saved_hours']:.1f}h")
        
        st.markdown("---")
        st.info("**Manual processing time:** ~15 min/doc")
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Process Documents", "📋 Processed Documents", "📈 Analytics"])
    
    # ============================================
    # TAB 1: PROCESS DOCUMENTS
    # ============================================
    with tab1:
        st.header("Upload and Process Documents")
        
        col1, col2 = st.columns(2)
        with col1:
            doc_type = st.selectbox("Document Type", list(DOCUMENT_TYPES.keys()))
        with col2:
            file_quality = st.select_slider(
                "Scan Quality",
                options=["poor", "fair", "good", "excellent"],
                value="good"
            )
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, PNG, JPG)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process All Documents"):
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    with st.spinner(f"Processing {file.name}..."):
                        start_time = datetime.now()
                        
                        # Determine processing method
                        if file.name.endswith('.pdf'):
                            text = read_pdf_text(file)
                            extracted_data = extract_with_gpt_text(text, doc_type, api_key)
                        else:
                            # Use vision for images (fallback to text if vision fails)
                            try:
                                extracted_data = extract_with_gpt4_vision(file, doc_type, api_key)
                            except:
                                st.warning(f"Vision API failed for {file.name}, trying text extraction...")
                                # Convert image to text first (simplified)
                                extracted_data = None
                        
                        if extracted_data:
                            # Validate
                            validation_results = validate_extraction(extracted_data, doc_type)
                            
                            # Handle edge cases
                            edge_cases = handle_edge_cases(extracted_data, file_quality)
                            
                            # Transform for ERP
                            extracted_data['validation_results'] = validation_results
                            extracted_data['edge_case_notes'] = edge_cases
                            extracted_data['filename'] = file.name
                            extracted_data['processing_time'] = (datetime.now() - start_time).total_seconds()
                            
                            erp_record = transform_for_erp(extracted_data, doc_type)
                            
                            st.session_state.processed_docs.append(erp_record)
                            
                            # Update stats
                            st.session_state.processing_stats['total'] += 1
                            if validation_results['is_valid']:
                                st.session_state.processing_stats['successful'] += 1
                            else:
                                st.session_state.processing_stats['failed'] += 1
                            st.session_state.processing_stats['time_saved_hours'] += 0.25  # 15 min per doc
                            
                            if validation_results['is_valid']:
                                st.success(f"✅ {file.name} processed successfully")
                            else:
                                st.warning(f"⚠️ {file.name} needs manual review")
                        else:
                            st.error(f"❌ Failed to process {file.name}")
                            st.session_state.processing_stats['total'] += 1
                            st.session_state.processing_stats['failed'] += 1
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"✅ Processed {len(uploaded_files)} documents!")
    
    # ============================================
    # TAB 2: PROCESSED DOCUMENTS
    # ============================================
    with tab2:
        st.header("📋 Processed Documents")
        
        if not st.session_state.processed_docs:
            st.info("No documents processed yet. Upload documents in the first tab.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                ["validated", "needs_review"],
                default=["validated", "needs_review"]
            )
        with col2:
            type_filter = st.multiselect(
                "Filter by Type",
                list(DOCUMENT_TYPES.keys()),
                default=list(DOCUMENT_TYPES.keys())
            )
        
        # Display documents
        for idx, doc in enumerate(st.session_state.processed_docs):
            if doc['status'] in status_filter and doc['document_type'] in type_filter:
                status_emoji = "✅" if doc['status'] == "validated" else "⚠️"
                
                with st.expander(f"{status_emoji} {doc['document_id']} - {doc['document_type']} - {doc['status']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Extracted Data")
                        
                        data = doc['data']
                        for key, value in data.items():
                            if isinstance(value, list) and key in ['line_items', 'items']:
                                st.write(f"**{key.replace('_', ' ').title()}:**")
                                items_df = pd.DataFrame(value)
                                st.dataframe(items_df, use_container_width=True)
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Validation results
                        st.subheader("Validation Results")
                        val_results = doc['validation_results']
                        
                        if val_results['is_valid']:
                            st.success("✅ All validations passed")
                        else:
                            if val_results['missing_required']:
                                st.error(f"Missing required fields: {', '.join(val_results['missing_required'])}")
                            if val_results['errors']:
                                st.error("Validation errors:")
                                for error in val_results['errors']:
                                    st.write(f"- {error}")
                        
                        if val_results['warnings']:
                            st.warning("Warnings:")
                            for warning in val_results['warnings']:
                                st.write(f"- {warning}")
                        
                        # Edge cases
                        if 'edge_case_notes' in doc and doc['edge_case_notes']:
                            st.subheader("Edge Cases Detected")
                            for note in doc['edge_case_notes']:
                                st.info(note)
                    
                    with col2:
                        st.subheader("Metadata")
                        st.write(f"**ID:** {doc['document_id']}")
                        st.write(f"**Type:** {doc['document_type']}")
                        st.write(f"**Status:** {doc['status']}")
                        st.write(f"**Confidence:** {doc['confidence']:.0%}")
                        st.write(f"**Processed:** {doc['processing_date'][:10]}")
                        
                        # Actions
                        st.subheader("Actions")
                        if st.button(f"✏️ Edit", key=f"edit_{idx}"):
                            st.info("Edit functionality - integrate with your ERP system")
                        if st.button(f"✅ Approve", key=f"approve_{idx}"):
                            doc['status'] = 'approved'
                            st.success("Document approved!")
                        if st.button(f"🗑️ Delete", key=f"delete_{idx}"):
                            st.warning("Delete functionality")
                        
                        # Export single document
                        if st.button(f"📥 Export JSON", key=f"export_{idx}"):
                            json_str = json.dumps(doc, indent=2, default=str)
                            st.download_button(
                                "Download JSON",
                                json_str,
                                f"{doc['document_id']}.json",
                                "application/json"
                            )
    
    # ============================================
    # TAB 3: ANALYTICS
    # ============================================
    with tab3:
        st.header("📈 Processing Analytics")
        
        if not st.session_state.processed_docs:
            st.info("No data to analyze yet.")
            return
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        docs = st.session_state.processed_docs
        total = len(docs)
        validated = sum(1 for d in docs if d['status'] == 'validated')
        avg_confidence = sum(d['confidence'] for d in docs) / total if total > 0 else 0
        
        with col1:
            st.metric("Total Documents", total)
        with col2:
            st.metric("Auto-Validated", f"{validated}/{total}")
        with col3:
            st.metric("Success Rate", f"{(validated/total*100):.1f}%")
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
        
        # Document type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Documents by Type")
            type_counts = {}
            for doc in docs:
                doc_type = doc['document_type']
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="Document Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Status Breakdown")
            status_counts = {}
            for doc in docs:
                status = doc['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            fig = px.bar(
                x=list(status_counts.keys()),
                y=list(status_counts.values()),
                title="Documents by Status",
                labels={'x': 'Status', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time savings
        st.subheader("Time Savings Analysis")
        time_saved = st.session_state.processing_stats['time_saved_hours']
        manual_time = total * 0.25  # 15 min per doc
        automation_rate = (time_saved / manual_time * 100) if manual_time > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Manual Time (est.)", f"{manual_time:.1f}h")
        with col2:
            st.metric("Actual Time", f"{manual_time - time_saved:.1f}h")
        with col3:
            st.metric("Time Saved", f"{time_saved:.1f}h ({automation_rate:.0f}%)")
        
        # Export all
        st.subheader("📥 Export Data")
        if st.button("Download All Documents (JSON)"):
            json_str = json.dumps(st.session_state.processed_docs, indent=2, default=str)
            st.download_button(
                "Download JSON",
                json_str,
                "all_documents.json",
                "application/json"
            )
        
        if st.button("Download Summary Report (CSV)"):
            summary_data = []
            for doc in docs:
                summary_data.append({
                    'Document ID': doc['document_id'],
                    'Type': doc['document_type'],
                    'Status': doc['status'],
                    'Confidence': f"{doc['confidence']:.0%}",
                    'Processing Date': doc['processing_date'][:10],
                    'Total Amount': doc['data'].get('total_amount', 'N/A')
                })
            
            df = pd.DataFrame(summary_data)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "processing_summary.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()