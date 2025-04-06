import os
from dotenv import load_dotenv
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
# If the Google Generative AI package supports an actual call, import and use it.
# (This example assumes that the GenerativeModel can be initialized with an API key.)
import google.generativeai as genai
# Load environment variables and API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# File paths and model names
PDF_PATH = (
    "/Users/satyamjadhav/Base/Codebases/Counsultadd-HK/data/pdf/ELIGIBLE RFP - 1.pdf"
)
DOCX_PATH = (
    "/Users/satyamjadhav/Base/Codebases/Counsultadd-HK/data/docs/Company_Data.docx"
)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "./chroma_db"
# Configure the API key once at the start of your program
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
LLM_MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(model_name='gemini-2.0-flash')
# Hardcoded company data and requirements for simulation
COMPANY_DATA = """
COMPANY DATA
Field: Company Legal Name
Data: FirstStaff Workforce Solutions, LLC
Field: Principal Business Address
Data: 3105 Maple Avenue, Suite 1200, Dallas, TX 75201
Field: Phone Number
Data: (214) 832-4455
Field: Fax Number
Data: (214) 832-4460
Field: Email Address
Data: proposals@firststaffsolutions.com
Field: Authorized Representative
Data: Meredith Chan, Director of Contracts
Field: Authorized Representative Phone
Data: (212) 555-0199
Field: Signature
Data: Meredith Chan (signed manually)
Field: Company Length of Existence
Data: 9 years
Field: Years of Experience in Temporary Staffing
Data: 7 years
Field: DUNS Number
Data: 07-842-1490
Field: CAGE Code
Data: 8J4T7
Field: SAM.gov Registration Date
Data: 03/01/2022
Field: NAICS Codes
Data: 561320 – Temporary Help Services; 541611 – Admin Management
Field: State of Incorporation
Data: Delaware
Field: Bank Letter of Creditworthiness
Data: Not Available.
Field: State Registration Number
Data: SRN-DE-0923847
Field: Services Provided
Data: Administrative, IT, Legal & Credentialing Staffing
Field: Business Structure
Data: Limited Liability Company (LLC)
Field: W-9 Form
Data: Attached (TIN: 47-6392011)
Field: Certificate of Insurance
Data: Travelers Insurance, Policy #TX-884529-A; includes Workers' Comp, Liability, and Auto
Field: Licenses
Data: Texas Employment Agency License #TXEA-34892
Field: Historically Underutilized Business/DBE Status
Data: Not certified.
Field: Key Personnel – Project Manager
Data: Ramesh Iyer
Field: Key Personnel – Technical Lead
Data: Sarah Collins
Field: Key Personnel – Security Auditor
Data: James Wu
Field: NO MBE Certification
Data: True
"""
INELIGIBLE_RFP_REQUIREMENTS = """
*Eligibility Criteria Found (Scenario: Ineligible PDF):*
* If eligible for Native American Preference, provide proof of enrollment/membership (such as a tribal enrollment card) in a federally recognized Tribe or proof of certification as an Indian-owned business (Native American Ownership must be 51% or more).
* If eligible for WBE/DVBE preference, provide proof of certification, which may consist of business registered to the WBE/DVBE owner and a copy of your certification from a government agency in the state the contractor is incorporated or has its principal business.
* If eligible for MBE/VBE preference, provide proof of certification, which may consist of business registered to the MBE/VBE owner and a copy of your certification from a government agency in the state the contractor is incorporated or has its principal business.
* Provide a brief summary of you/your firm's experience in this field and overall qualifications.
* Provide a minimum of three (3) business references of existing clients which should include all contact information.
* The Native American Preference Employment Plan requires the successful General Contractor, and all sub-contractors subsequently hired by the General Contractor, to employ at least fifty percent (50%) qualified Native American workers per division for the project, i.e. electricians, carpenters, plumbers, roofers, etc.
* The General Contractor and the Subcontractors shall be required to employ at least sixty percent (60%) unskilled positions with qualified Native Americans.
* The General Contractor and all sub-contractors shall maintain the established 50% and 60% Native employment benchmarks throughout the duration of the project and, wherever possible, the General Contractor and sub-contractors shall endeavor to exceed these benchmarks.
"""
ELIGIBLE_RFP_REQUIREMENTS = """
*Eligibility Criteria Found (Scenario: Eligible PDF):*
* The successful bidder must have a minimum of three (3) years of business in Temporary Staffing.
* Submit company background information including principal place of business, length of existence, breadth of experience and expertise, management structure, and any other information that demonstrates relative qualifications and experience.
* W-9 and Certificate of Insurance as required herein. If not, then: A formal letter from your CPA is an acceptable alternative for Non-Public companies but must include a financial solvency statement confirming financial adequacy to meet expenditures for a minimum of one year.
* Please include name of the company, name of representative, physical address, telephone number, and email address. [References Requirement]
* Proof of insurance MUST be submitted with the proposal; The awarded company must submit a certificate of insurance as proof of coverage to the Purchasing Director at 3840 Hulen St, Suite 538, Fort Worth, Texas 76107.
* The following are prohibited as defined by the Texas Government Code: "Scrutinized business operations in Sudan;" "Scrutinized business operations in Iran;" and "Scrutinized business operations with designated foreign terrorist organizations."
* Required Project manager or Account Administrative.
* Please submit proof of Historically Underutilized Business "HUB" state certificate. If not certified, but you intend to subcontract services, please Submit Attachment A. If you do not intend to subcontract services, write "None" on Attachment A and submit it.
* Please submit proof of M/W/DBE certificate.
"""
# --- Helper Functions ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip() or "No text extracted from PDF."
def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file using python-docx."""
    try:
        doc = Document(docx_path)
        full_text = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    # Fallback to hardcoded data if reading fails
    return COMPANY_DATA
def chunk_text(text, method="structural", chunk_size=500, overlap=50):
    """Chunks text using different strategies."""
    print(f"Chunking text using '{method}' method...")
    chunks = []
    if method == "structural":
        # Use double newlines as paragraph boundaries and adjust headings heuristically.
        current_section = "Introduction"
        paragraphs = text.split("\n\n")
        for i, para in enumerate(paragraphs):
            para_stripped = para.strip()
            if len(para_stripped) > 10:
                # If the line is in all caps, assume it is a section heading.
                if para_stripped.isupper() and len(para_stripped.split()) < 10:
                    current_section = para_stripped
                chunk_data = {
                    "text": para_stripped,
                    "metadata": {
                        "chunk_id": f"struct_{i}",
                        "section": current_section,
                        "source": "RFP/CompanyDoc",
                    },
                }
                chunks.append(chunk_data)
    elif method == "semantic":
        # A simple fixed-size semantic split as placeholder.
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text_str = " ".join(words[i: i + chunk_size])
            chunk_data = {
                "text": chunk_text_str,
                "metadata": {
                    "chunk_id": f"semantic_{i}",
                    "source": "RFP/CompanyDoc",
                },
            }
            chunks.append(chunk_data)
    else:
        # Simple fixed-size chunking.
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text_str = " ".join(words[i: i + chunk_size])
            chunk_data = {
                "text": chunk_text_str,
                "metadata": {
                    "chunk_id": f"simple_{i}",
                    "source": "RFP/CompanyDoc",
                },
            }
            chunks.append(chunk_data)
    print(f"Split text into {len(chunks)} chunks.")
    tokenized_chunks = [chunk["text"].lower().split() for chunk in chunks]
    return chunks, tokenized_chunks
def setup_vector_store(text_chunks_with_metadata, embedding_model_name):
    """
    Creates embeddings and indexes them in ChromaDB.
    Also sets up a BM25 keyword index.
    """
    print(f"Loading embedding model '{embedding_model_name}'...")
    embedding_model = SentenceTransformer(embedding_model_name)
    texts = [chunk["text"] for chunk in text_chunks_with_metadata]
    print("Generating embeddings for chunks...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    # Initialize or load the ChromaDB client and collection
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection_name = "rfp_analysis_docs"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection '{collection_name}'.")
    except Exception as e:
        print(f"Creating new collection '{collection_name}'.")
        collection = client.create_collection(name=collection_name)
    print("Adding documents, embeddings, and metadata to ChromaDB...")
    ids = [chunk["metadata"]["chunk_id"] for chunk in text_chunks_with_metadata]
    collection.add(embeddings=embeddings, documents=texts, metadatas=[chunk["metadata"] for chunk in text_chunks_with_metadata], ids=ids)
    vector_store = {
        "client": client,
        "collection": collection,
        "embeddings": embeddings,
        "chunks_with_metadata": text_chunks_with_metadata,
    }
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    keyword_index = {"bm25": bm25_index, "tokenized_corpus": tokenized_corpus}
    print("Vector store and keyword index created.")
    return vector_store, keyword_index
def retrieve_relevant_chunks(query, vector_store, keyword_index, embedding_model_name, top_k=5):
    """
    Retrieves relevant text chunks using a hybrid search:
    semantic search via ChromaDB (if available) and keyword search via BM25.
    Combines the results using Reciprocal Rank Fusion (RRF).
    """
    print(f"Retrieving chunks relevant to query: '{query}'")
    all_chunks = vector_store["chunks_with_metadata"]
    # --- Semantic Search via ChromaDB ---
    collection = vector_store.get("collection")
    semantic_chunks = []
    if collection:
        try:
            # Generate query embedding using the same embedding model.
            embedding_model = SentenceTransformer(embedding_model_name)
            query_embedding = embedding_model.encode([query]).tolist()[0]
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            # Parse the returned results to form a list of chunks.
            retrieved_docs = semantic_results.get("documents", [[]])[0] # changed
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    semantic_chunks.append(next((chunk for chunk in all_chunks if chunk["text"] == doc), None))
            print(f"Semantic search returned {len(semantic_chunks)} chunks.")
        except Exception as e:
            print(f"Semantic search error: {e}")
    else:
        print("No collection available for semantic search.")
    # --- Keyword Search using BM25 ---
    bm25 = keyword_index.get("bm25")
    tokenized_corpus = keyword_index.get("tokenized_corpus")
    query_tokens = query.lower().split()
    keyword_scores = bm25.get_scores(query_tokens)
    # Get indices of top scoring chunks
    keyword_indices = sorted(range(len(keyword_scores)), key=lambda i: keyword_scores[i], reverse=True)[:top_k]
    keyword_chunks = [all_chunks[i] for i in keyword_indices]
    print(f"Keyword search returned {len(keyword_chunks)} chunks.")
    # --- Combine Results with Reciprocal Rank Fusion (RRF) ---
    combined_scores = {}
    k_rrf = 60
    for rank, chunk in enumerate(semantic_chunks):
        if chunk:
            chunk_id = chunk["metadata"].get("chunk_id", chunk["text"])
            combined_scores.setdefault(chunk_id, {"score": 0, "chunk": chunk})
            combined_scores[chunk_id]["score"] += 1 / (k_rrf + rank + 1)
    for rank, chunk in enumerate(keyword_chunks):
        chunk_id = chunk["metadata"].get("chunk_id", chunk["text"])
        combined_scores.setdefault(chunk_id, {"score": 0, "chunk": chunk})
        combined_scores[chunk_id]["score"] += 1 / (k_rrf + rank + 1)
    sorted_chunks = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
    final_chunks = [item["chunk"] for item in sorted_chunks][: int(top_k * 1.5)]
    # Format the results as a context string for LLM prompting.
    context_str = ""
    for chunk in final_chunks:
        meta = chunk["metadata"]
        metadata_str = f"Source: {meta.get('source', 'N/A')}, Section: {meta.get('section', 'N/A')}, Chunk ID: {meta.get('chunk_id', 'N/A')}"
        context_str += f"--- Chunk Metadata: {metadata_str} ---\n{chunk['text']}\n---\n"
    return context_str
def query_llm(prompt, model_name=LLM_MODEL_NAME):
    """
    Queries the LLM. If the API key is available and the model call works,
    it will call the real model. Otherwise, it falls back to simulation.
    """
    print(f"Querying LLM model '{model_name}' with prompt length {len(prompt)}...")
    try:
        # Use the generate_content function instead of instantiating a model class.
        response = model.generate_content(contents=prompt)
        return response.text  # Adjust based on the response structure.
    except Exception as e:
        print(f"Error calling LLM model: {e}. Falling back to simulation.")
    
    # --- Fallback: Simulation Logic ---
    if (
        "compare the company information against each requirement" in prompt.lower()
        and "here is information about our company" in prompt.lower()
    ):
        # [Simulation logic as before...]
        return "LLM Response (Simulated): Request processed."
    elif (
        "synthesize a single, comprehensive list of all requirements and guidelines" in prompt.lower()
        and "formatting and submission" in prompt.lower()
    ):
        return """
*Formatting and Submission Guidelines (Simulated Synthesis):*
- Table of Contents required; follow RFP Section order.
- Maximum 50 pages (excluding attachments); Times New Roman, 12pt, double-spaced.
- Attach required documents (Form B, Certificate of Insurance, W-9, HUB/M/W/DBE if applicable).
- Submission via State Procurement Portal by July 15, 2024, 5:00 PM CT.
File naming convention: CompanyName_Proposal_YYYYMMDD.pdf
"""
    elif (
        "synthesize a comprehensive analysis of potential risks and unfavorable clauses" in prompt.lower()
        and "contractual terms" in prompt.lower()
    ):
        return """
*Risk Analysis and Suggestions (Simulated Synthesis):*
- Liability: Unlimited exposure; suggest negotiating a cap.
- Termination: One-sided; recommend mutual termination rights.
- Intellectual Property: All rights to the state; negotiate retention of pre-existing IP.
- Payment Terms: Net 60; suggest reducing to Net 30 or adding late payment interest.
- Confidentiality: Indefinite survival; propose a limited duration.
"""
    else:
        return "LLM Response (Simulated): Request processed."
# --- Main Workflow ---
if __name__ == "__main__":
    # 1. Load Models
    print("Initializing embedding model and LLM client...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # LLM client is initialized on-demand in query_llm
    # 2. Parse Documents
    rfp_text_from_pdf = extract_text_from_pdf(PDF_PATH)
    company_text_from_doc = extract_text_from_docx(DOCX_PATH)
    # Prefer the DOCX extraction if available; otherwise, fallback to COMPANY_DATA
    company_text = company_text_from_doc or COMPANY_DATA
    # 3. Process RFP Text for RAG
    rfp_text_placeholder = """
REQUEST FOR PROPOSAL (RFP) - Temporary Staffing Services
SECTION 1: INTRODUCTION
... overview ...
SECTION 2: ELIGIBILITY REQUIREMENTS
The successful bidder must have a minimum of three (3) years of business in Temporary Staffing.
Submit company background information... W-9 and Certificate of Insurance... References Requirement...
Proof of insurance MUST be submitted... Prohibited Operations... Required Project Manager...
Please submit proof of Historically Underutilized Business "HUB" state certificate. If not certified... Submit Attachment A.
Please submit proof of M/W/DBE certificate.
SECTION 3: SCOPE OF WORK
... details on services needed ...
SECTION 4: FORMATTING AND SUBMISSION GUIDELINES
Proposals must not exceed 50 pages, excluding required attachments.
Use Times New Roman, 12pt font, double-spaced, with 1-inch margins.
Include a Table of Contents. Structure must follow RFP Section order.
Required Attachments: Form B (Signed), Certificate of Insurance, W-9. Submit HUB/Attachment A and M/W/DBE if applicable.
File Naming: CompanyName_Proposal_YYYYMMDD.pdf
Submission: Upload via State Procurement Portal by July 15, 2024, 5:00 PM CT. Single PDF file.
SECTION 5: CONTRACT TERMS AND CONDITIONS
5.1 Liability: Contractor assumes unlimited liability... Contractor shall indemnify State...
5.2 Termination: State may terminate for convenience (30 days notice). Contractor termination only for cause (60 day cure).
5.3 Intellectual Property: All work product is sole property of the State.
5.4 Payment Terms: Net 60 upon acceptance. No late payment interest mentioned.
5.5 Confidentiality: Broad definition, survives indefinitely.
... other standard clauses ...
"""
    print("\n--- Setting up RAG ---")
    rfp_chunks, _ = chunk_text(rfp_text_placeholder, method="structural")
    vector_store, keyword_index = setup_vector_store(rfp_chunks, EMBEDDING_MODEL_NAME)
    # 4. Perform Analysis Tasks
    # Task 2: Comparison using Hardcoded Requirements and Company Data
    print("\n--- Task 2 (Scenario 1): Comparing 'Ineligible' Requirements ---")
    prompt2_ineligible = f"""
Analyze the following request carefully.
Here are the eligibility requirements extracted from an RFP:
{INELIGIBLE_RFP_REQUIREMENTS}
Here is information about our company:
---
{company_text}
---
Task: Compare the company information against each specific eligibility requirement listed above. For each requirement, determine if the company data indicates 'MET', 'NOT MET', or 'ACTION REQUIRED'. Provide a brief justification based only on the provided company data. Do not include any information or points that are not present in the company data or requirements. Finally, give an overall assessment of eligibility based on these requirements.
Comparison:
"""
    comparison_summary_ineligible = query_llm(prompt2_ineligible)
    print("\nEligibility Comparison Summary (Ineligible Scenario):")
    print(comparison_summary_ineligible)
    print("\n--- Task 2 (Scenario 2): Comparing 'Eligible' Requirements ---")
    prompt2_eligible = f"""
Analyze the following request carefully.
Here are the eligibility requirements extracted from an RFP:
{ELIGIBLE_RFP_REQUIREMENTS}
Here is information about our company:
---
{company_text}
---
Task: Compare the company information against each specific eligibility requirement listed above. For each requirement, determine if the company data indicates 'MET', 'NOT MET', or 'ACTION REQUIRED'. Provide a brief justification based only on the provided company data. Do not include any information or points that are not present in the company data or requirements. Finally, give an overall assessment of eligibility based on these requirements.
Comparison:
"""
    comparison_summary_eligible = query_llm(prompt2_eligible)
    print("\nEligibility Comparison Summary (Eligible Scenario):")
    print(comparison_summary_eligible)
    # Task 3: Formatting Specifications Synthesis via RAG
    print("\n--- Task 3: Extracting Formatting Specifications ---")
    formatting_queries = [
        "What are the requirements for document structure like Table of Contents or sections?",
        "Find all rules about physical formatting: page limits, font type, font size, line spacing, margins.",
        "List all required attachments, forms, or certificates for the proposal submission.",
        "Describe the required submission method and the deadline.",
        "Are there specific naming conventions for files or sections?",
    ]
    all_formatting_context = ""
    for query_text in formatting_queries:
        context = retrieve_relevant_chunks(query_text, vector_store, keyword_index, EMBEDDING_MODEL_NAME, top_k=3)
        all_formatting_context += f"--- Context for query: '{query_text}' ---\n{context}\n\n"
    prompt3_synth = f"""
Analyze the following text snippets extracted from the RFP:
---
{all_formatting_context}
---
Based only on these snippets, synthesize a single comprehensive list of all formatting and submission requirements and guidelines. Organize the list logically (by structure, formatting details, attachments, and submission instructions). Do not include any points that are not present in the given RFP snippets.
Formatting and Submission Guidelines (Synthesized):
"""
    formatting_specs = query_llm(prompt3_synth)
    print("\nExtracted Formatting Specifications:")
    print(formatting_specs)
    # Task 4: Contract Risk Analysis via RAG
    print("\n--- Task 4: Analyzing Contract Risks ---")
    risk_queries = [
        "What are the clauses related to liability and indemnification?",
        "Find terms regarding termination rights for both parties.",
        "What are the warranty periods and obligations?",
        "Describe the payment terms and invoicing process.",
        "Identify clauses about intellectual property ownership.",
        "What are the confidentiality requirements?",
        "Find information on governing law and dispute resolution.",
        "Are there specific data security or privacy requirements?",
    ]
    all_risk_context = ""
    for query_text in risk_queries:
        context = retrieve_relevant_chunks(query_text, vector_store, keyword_index, EMBEDDING_MODEL_NAME, top_k=3)
        all_risk_context += f"--- Context for query: '{query_text}' ---\n{context}\n\n"
    prompt4_synth = f"""
Analyze the following text snippets extracted from the RFP:
---
{all_risk_context}
---
Based only on these snippets, synthesize a comprehensive analysis of potential risks and unfavorable clauses for the contractor. For each risk category (e.g. Liability, Termination, IP, Payment), summarize the clause, explain the risk, and suggest a modification or negotiation point. Do not include any information or points that are not present in the given RFPsnippets.
Risk Analysis and Suggestions (Synthesized):
"""
    risk_analysis = query_llm(prompt4_synth)
    print("\nRisk Analysis Summary:")
    print(risk_analysis)
    print("\n--- RFP Analysis Complete ---")

