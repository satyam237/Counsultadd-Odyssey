from typing import Annotated
import re
from pathlib import Path
from IPython.display import Image, display  # noqa: F401
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

# Initialize embeddings and vector stores
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Define paths and load vector stores
data_dir = Path("/Users/satyamjadhav/Base/Codebases/Counsultadd-HK/data")
vector_stores_dir = data_dir / "vector_stores"

def load_vector_store(store_name: str) -> Chroma:
    """Load a vector store."""
    store_path = vector_stores_dir / store_name

    try:
        # Initialize ChromaDB with proper configuration
        return Chroma(
            persist_directory=str(store_path),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},  # Add default metadata
            collection_name=store_name,  # Explicitly set collection name
        )
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        print("Attempting to recreate the collection...")

        # If loading fails, try to create a new collection
        client = chromadb.PersistentClient(path=str(store_path))

        # Delete existing collection if it exists
        client.delete_collection(store_name)


        # Create new collection with proper configuration
        return Chroma(
            persist_directory=str(store_path),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name=store_name,
        )


# Load only eligible_rfp1 vector store
eligible_rfp_store = load_vector_store("eligible_rfp1")

# Define analysis prompt
ANALYSIS_PROMPT = """
Extracting Key Mandatory Requirements from RFP Document

Fetched Context from RFP:
{context}

Goal:
Extract and list ONLY the following key mandatory requirements exactly as stated in the document:

1. Years of experience in Temporary Staffing
2. Company background submission requirements
3. W-9 and Certificate of Insurance requirements (including CPA letter alternative)
4. Company contact information requirements
5. Insurance proof submission requirements
6. Prohibited business operations (Sudan, Iran, terrorist organizations)
7. Project Manager/Account Administrator requirements
8. HUB certification and M/W/DBE requirements (including Attachment A)

Instructions:
- Extract ONLY the exact requirements as stated in the document
- Include specific submission details where mentioned
- Use bullet points for clarity
- Include exact quotes where available
"""

analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)

# First, let's add the new format requirements prompt and function
FORMAT_REQUIREMENTS_PROMPT = """
Extract ALL formatting specifications and required attachments from the RFP document with extreme precision.

Fetched Context from RFP:
{context}

Analyze and categorize the following with EXACT quotes from the document:

1. Document Formatting Requirements:
   - Page limits (overall and per section)
   - Font specifications (type, size)
   - Margins and spacing
   - Headers/footers format
   - Page numbering requirements
   - Section organization rules

2. Required (Mandatory) Attachments:
   - Forms that MUST be submitted
   - Certificates and licenses
   - Financial documents
   - Legal documents
   - Insurance documents
   - Any other mandatory submissions

3. Preferred/Optional Attachments:
   - Additional supporting documents
   - Optional certifications
   - Supplementary materials
   - Reference documents

For each requirement:
- Quote the EXACT text from RFP
- Include page number reference if available
- Mark as [MANDATORY] or [OPTIONAL]
- Include any specific formatting instructions for attachments

Format the response in clear sections with bullet points.
If a requirement is not explicitly stated, mark as [NOT SPECIFIED].
"""

analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    analysis_results: dict
    requirement_check: dict | None
    format_requirements: dict | None


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=os.environ.get("GEMINI_API_KEY")
)


def hybrid_retrieval(
    store: Chroma, query: str, k: int = 30, keyword_filter: list[str] = None
) -> str:
    """Perform hybrid retrieval using MMR with optional keyword filtering."""
    try:
        mmr_results = store.max_marginal_relevance_search(query, k=k)
    except Exception:
        mmr_results = store.similarity_search(query, k=k)

    if keyword_filter:
        filtered = [
            doc
            for doc in mmr_results
            if any(kw.lower() in doc.page_content.lower() for kw in keyword_filter)
        ]
        results = filtered if filtered else mmr_results
    else:
        results = mmr_results

    formatted_sections = []
    for i, doc in enumerate(results):
        section = f"Section {i + 1}:"
        if "page" in doc.metadata:
            section += f" (Page {doc.metadata['page']})"
        section += f"\n{doc.page_content}"
        formatted_sections.append(section)

    return "\n\n".join(formatted_sections)


def get_relevant_content(query: str, k: int = 30) -> str:
    """Get relevant content from all eligible RFP vector stores."""
    all_results = []

    # Search in ELIGIBLE RFP - 1
    try:
        results1 = eligible_rfp_store.similarity_search(query, k=k // 2)
        all_results.extend(results1)
    except Exception as e:
        print(f"Error searching ELIGIBLE RFP - 1: {str(e)}")

    # Search in ELIGIBLE RFP - 2
    try:
        rfp2_store = load_vector_store("eligible_rfp2")
        results2 = rfp2_store.similarity_search(query, k=k // 2)
        all_results.extend(results2)
    except Exception as e:
        print(f"Error searching ELIGIBLE RFP - 2: {str(e)}")

    # Format results with section numbers and metadata
    formatted_sections = []
    for i, doc in enumerate(all_results):
        section = f"Section {i + 1}:"
        if "page" in doc.metadata:
            section += f" (Page {doc.metadata['page']})"
        if "source" in doc.metadata:
            section += f" (Source: {doc.metadata['source']})"
        section += f"\n{doc.page_content}"
        formatted_sections.append(section)

    return "\n\n".join(formatted_sections)


def format_markdown_text(text: str) -> str:
    """Format markdown text for better readability."""
    # Convert **text** to bold
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Add proper spacing around sections
    text = re.sub(r"\n(?=[A-Z][a-z]+ Requirements:)", r"\n\n", text)
    text = re.sub(r"\n(?=Overall Summary)", r"\n\n", text)

    # Improve bullet point formatting
    text = re.sub(r"\n\s*-\s*", r"\n• ", text)

    # Add spacing after section headers
    text = re.sub(r"(:\n)", r":\n\n", text)

    return text


def print_section(title: str, content: str, char: str = "=") -> None:
    """Print a formatted section with title and content."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")
    print(format_markdown_text(content))


def print_format_requirements(reqs: dict):
    """Print format requirements in a clean, organized way."""
    print("\nDOCUMENT FORMAT REQUIREMENTS AND ATTACHMENTS")
    print("=" * 80)
    print()

    if "raw_analysis" in reqs:
        # Split the raw analysis into lines and process each line
        lines = reqs["raw_analysis"].split("\n")
        current_indent = 0

        for line in lines:
            # Skip empty lines
            if not line.strip():
                print()
                continue

            # Detect section headers and adjust indentation
            if (
                "DOCUMENT FORMAT REQUIREMENTS:" in line
                or "REQUIRED ATTACHMENTS:" in line
                or "OPTIONAL ATTACHMENTS:" in line
            ):
                current_indent = 0
                print("\n" + line.strip())
                print("-" * len(line.strip()))
            # Detect subsections and adjust indentation
            elif line.strip().endswith(":"):
                current_indent = 2
                print("\n" + " " * current_indent + line.strip())
            # Format list items with proper indentation
            elif line.strip().startswith("-") or line.strip().startswith("*"):
                current_indent = 4
                print(" " * current_indent + line.strip())
            else:
                print(" " * current_indent + line.strip())
    else:
        print("No raw analysis available")

    print("\n" + "=" * 80)


def run_analysis():
    """Run the RFP analysis pipeline."""
    print("\n" + "=" * 80)
    print("RFP ANALYSIS PIPELINE".center(80))
    print("=" * 80 + "\n")

    # Initialize state with empty messages and analysis results
    initial_state = {"messages": [], "analysis_results": {}}

    # Run the graph
    print("Step 1: Analyzing RFP Requirements...")
    print("-" * 40 + "\n")

    for event in graph.stream(initial_state):
        for key, value in event.items():
            if key == "rfp_analysis":
                print("RFP Requirements Analysis:")
                print("=" * 25 + "\n")
                analysis_content = value["analysis_results"]["analysis"]
                print(format_markdown_text(analysis_content))
                print("\nSource Sections Used:")
                print("-" * 20 + "\n")
                print(format_markdown_text(value["analysis_results"]["context"]))
                print("\n" + "=" * 80 + "\n")

            elif key == "chatbot":
                summary_content = value["messages"][-1]["content"]
                print_section("Summary and Recommendations", summary_content)

            elif key == "check_requirements_node":
                print("\nStep 2: Checking Company Compliance...")
                print("-" * 40 + "\n")
                if "requirement_check" in value:
                    results = value["requirement_check"]

                    print("Detailed Analysis by Category:")
                    print("=" * 30 + "\n")

                    # Print details for each category
                    for category, details in results["details"].items():
                        print(f"{category}:")
                        print("-" * len(category))
                        print(format_markdown_text(details))
                        print()

                    print("\nOverall Status:")
                    print("=" * 15 + "\n")
                    print("✓ Satisfied Requirements:")
                    for req in results["satisfied"]:
                        print(f"  • {req}")

                    print("\n⚠ Partially Met Requirements:")
                    for req in results["partial"]:
                        print(f"  • {req}")

                    print("\n✗ Missing Requirements:")
                    for req in results["missing"]:
                        print(f"  • {req}")

                    print("\n" + "=" * 80)
                    print("FINAL RECOMMENDATION".center(80))
                    print("=" * 80 + "\n")
                    print(
                        format_markdown_text(
                            results["full_analysis"]
                            .split("RECOMMENDATION:")[-1]
                            .strip()
                        )
                    )
                    print("\n" + "=" * 80)

            elif key == "format_analysis":
                print("\nStep 3: Document Format Requirements and Attachments")
                print("-" * 50)

                if "format_requirements" in value["analysis_results"]:
                    reqs = value["analysis_results"]["format_requirements"]
                    print_format_requirements(reqs)
                    print("\n" + "=" * 80)

            elif key == "risk_analysis":
                print("\nStep 4: Risk Analysis")
                print("-" * 50)

                if "risk_analysis" in value["analysis_results"]:
                    risk_results = value["analysis_results"]["risk_analysis"]
                    print_risk_analysis(risk_results)
                    print("\n" + "=" * 80)


def format_analysis_output(analysis_text: str) -> str:
    """Format the analysis output for better readability."""
    sections = {
        "Mandatory Requirements": [],
        "Technical Requirements": [],
        "Compliance Criteria": [],
        "Key Deadlines or Timelines": [],
        "Overall Summary": [],
    }

    current_section = None
    lines = analysis_text.split("\n")

    for line in lines:
        # Check if this is a section header
        for section in sections.keys():
            if section in line:
                current_section = section
                break

        # Add line to current section if we have one
        if current_section and line.strip():
            sections[current_section].append(line.strip())

    # Format the output
    formatted_output = []
    for section, content in sections.items():
        if content:
            formatted_output.append(f"\n{section}:")
            formatted_output.extend(
                [f"  {line}" for line in content if line != section + ":"]
            )

    return "\n".join(formatted_output)


def analyze_rfp(state: State):
    query = """
Find ALL sections that include ANY of the following requirements:

1. Experience Requirements:
- Minimum years of experience in Temporary Staffing
- Business existence duration
- Company background requirements

2. Documentation Requirements:
- W-9 submission
- Certificate of Insurance
- CPA solvency letter requirements
- Company contact information requirements
- Physical address requirements
- Insurance proof submission

3. Compliance Requirements:
- Prohibited business operations (Sudan, Iran, terrorist organizations)
- Texas Government Code requirements
- Business operation restrictions

4. Certification Requirements:
- HUB certification
- M/W/DBE certification
- Subcontracting requirements (Attachment A)

5. Personnel Requirements:
- Project Manager requirements
- Account Administrator requirements
- Management team requirements

6. Company Information:
- Principal place of business
- Management structure
- Contact details
- Physical address
- Representative information
"""

    # Updated keywords list
    keywords = [
        "years of experience",
        "years of business",
        "Temporary Staffing",
        "company background",
        "length of existence",
        "breadth of experience",
        "management structure",
        "company name",
        "representative name",
        "physical address",
        "telephone number",
        "email address",
        "principal place of business",
        "W-9",
        "Certificate of Insurance",
        "CPA letter",
        "financial solvency statement",
        "insurance coverage",
        "proof of insurance",
        "Purchasing Director",
        "Scrutinized business operations",
        "Sudan",
        "Iran",
        "foreign terrorist organizations",
        "Texas Government Code",
        "Project Manager",
        "Account Administrator",
        "required personnel",
        "HUB certificate",
        "Historically Underutilized Business",
        "M/W/DBE certificate",
        "minority business enterprise",
        "Attachment A",
        "subcontract services",
        "proof of coverage",
        "financial adequacy",
        "minimum of three years",
        "business operations",
        "management team",
        "organization chart",
    ]

    content = hybrid_retrieval(eligible_rfp_store, query, k=30, keyword_filter=keywords)
    analysis_result = llm.invoke(analysis_prompt.format(context=content))

    # Format the analysis result
    formatted_analysis = format_analysis_output(analysis_result.content)

    return {
        "messages": [analysis_result],
        "analysis_results": {"context": content, "analysis": formatted_analysis},
    }


def chatbot(state: State):
    """Regular chatbot function."""
    analysis = state["analysis_results"]["analysis"]
    context = state["analysis_results"]["context"]

    chatbot_prompt = f"""
Based on the RFP analysis below, extract the key mandatory requirements using the most relevant text or quotes you can find. For each of the following items, try to include the text verbatim if available. If not, include the closest matching statement:

1. Minimum years requirement for Temporary Staffing experience.
2. Company background information submission requirements.
3. W-9 and Certificate of Insurance requirements (or CPA letter alternative if specified).
4. Company contact information requirements.
5. Insurance proof submission requirements.
6. Prohibited business operations statement.
7. Project Manager and Account Administrator requirements.
8. HUB/M/W/DBE certification and Attachment A requirements.

Your response should contain one bullet point per requirement in the following format:
- [Extracted text]

Do not include any additional commentary. Use the exact wording from the document when available. If you cannot find an exact quote for an item, provide the most accurate paraphrase while indicating that it is paraphrased.

RFP Analysis Summary:
{analysis}

Original Context:
{context}
"""


    chat_response = llm.invoke(chatbot_prompt)
    formatted_response = format_markdown_text(chat_response.content)

    return {"messages": [{"role": "assistant", "content": formatted_response}]}



def check_requirements(state: State):
    """Check if company data meets RFP requirements."""
    analysis = state["analysis_results"]["analysis"]

    try:
        # Read company data using Path for proper path handling
        company_data_path = Path("/Users/satyamjadhav/Base/Codebases/Counsultadd-HK/data/docs/company_data.txt")
        with open(company_data_path, "r", encoding="utf-8") as f:
            company_text = f.read().strip()

        # Normalize text: lower-case and collapse whitespace
        normalized_text = re.sub(r'\s+', ' ', company_text.lower())

        # Validate that we have all the required sections using the normalized text
        required_sections = [
            "company information",
            "company details",
            "experience and services",
            "documentation and compliance",
            "certifications",
            "key personnel",
        ]
        
        missing_sections = [
            section for section in required_sections if section not in normalized_text
        ]

        if missing_sections:
            raise ValueError(f"Missing required sections in company data: {missing_sections}")

        # Ensure the data is properly formatted (adjust the threshold if necessary)
        if not company_text or len(company_text.split("\n")) < 10:
            raise ValueError("Company data appears to be incomplete or malformed")

    except Exception as e:
        print(f"Error loading company data: {str(e)}")
        return {
            "messages": state["messages"],
            "analysis_results": state["analysis_results"],
            "requirement_check": {
                "satisfied": [],
                "missing": ["ALL - Company data not found or incomplete"],
                "partial": [],
                "details": {},
                "full_analysis": "",
                "summary": f"ERROR: Could not load company data file. Details: {str(e)}",
            },
        }

    # Continue with further processing if needed...


    # If all checks pass, you would continue with further analysis...
    # (This part of the code is not shown here.)





    # Direct comparison prompt with dynamic company data extraction
    verification_prompt = f"""
    Task: Compare the RFP requirements with the company information and determine if each requirement is satisfied.

    RFP Requirements:
    {analysis}

    Company Information (organized by category):
    {company_text}

    For each of these categories, analyze if the company information EXACTLY matches the RFP requirements.
    Pay special attention to:

    1. Experience Requirements:
       - Extract and verify years of Temporary Staffing experience from company data
       - Extract and verify company's total years of existence from company data

    2. Documentation Requirements:
       - Check if W-9 is mentioned as attached
       - Verify Certificate of Insurance details from company data
       - Check for Bank Letter of Creditworthiness status
       - Verify any licensing information

    3. Company Information:
       - Verify all contact details are present (address, phone, email)
       - Extract and verify physical address location
       - Check management structure details

    4. Certifications:
       - Check HUB certification status from company data
       - Check MBE/DBE certification status from company data

    5. Personnel:
       - Verify Project Manager assignment from company data
       - Check other key personnel roles and assignments

    Provide your analysis in this exact format for each category:

    CATEGORY: [category name]
    STATUS: [SATISFIED/MISSING/PARTIAL]
    EVIDENCE: [Quote the exact text from company information that satisfies the requirement]
    MISSING: [List any specific items not found or requirements not met]
    GAP ANALYSIS: [Explain any gaps between requirements and provided information]
    
    After analyzing each category, provide a final recommendation:
    "RECOMMENDATION: Proceed with bid submission." (if all critical requirements are met)
    OR
    "RECOMMENDATION: Hold bid submission until missing requirements are addressed." (if critical requirements are missing)
    
    Include specific details about what needs to be addressed if recommending a hold.
    """

    try:
        # Get detailed analysis from LLM with error handling
        result = llm.invoke(verification_prompt)
        analysis_text = result.content

        if not analysis_text or len(analysis_text.split()) < 50:
            raise ValueError("LLM response appears to be incomplete or malformed")

        # Parse the analysis to extract results
        categories = [
            "Experience Requirements",
            "Documentation Requirements",
            "Company Information",
            "Certifications",
            "Personnel",
        ]

        results = {"satisfied": [], "missing": [], "partial": [], "details": {}}

        # Parse the analysis text to categorize results with improved error handling
        for category in categories:
            try:
                # Split by category headers and find the relevant section
                sections = analysis_text.split("CATEGORY:")
                category_section = next(
                    (s for s in sections if category in s.strip().split("\n")[0]), None
                )

                if category_section:
                    section_text = category_section.strip()

                    # Validate section content
                    if len(section_text.split("\n")) < 3:
                        raise ValueError(
                            f"Incomplete analysis for category: {category}"
                        )

                    # Determine status with validation
                    if "STATUS: SATISFIED" in section_text:
                        if "EVIDENCE:" not in section_text:
                            raise ValueError(
                                f"Missing evidence for satisfied status in {category}"
                            )
                        results["satisfied"].append(category)
                    elif "STATUS: MISSING" in section_text:
                        results["missing"].append(category)
                    elif "STATUS: PARTIAL" in section_text:
                        results["partial"].append(category)
                    else:
                        raise ValueError(f"Invalid status for category: {category}")

                    # Store the detailed analysis
                    results["details"][category] = section_text
                else:
                    results["missing"].append(category)
                    results["details"][category] = f"Analysis not found for {category}"

            except Exception as e:
                print(f"Error processing category {category}: {str(e)}")
                results["missing"].append(category)
                results["details"][category] = f"Error processing {category}: {str(e)}"

        # Extract and validate recommendation
        if "RECOMMENDATION:" not in analysis_text:
            raise ValueError("Missing recommendation in analysis")

        recommendation = analysis_text.split("RECOMMENDATION:")[-1].strip()

        if not recommendation or len(recommendation.split()) < 5:
            raise ValueError("Recommendation appears to be incomplete")

        # Format the results
        output = {
            "messages": state["messages"],
            "analysis_results": state["analysis_results"],
            "requirement_check": {
                "satisfied": results["satisfied"],
                "missing": results["missing"],
                "partial": results["partial"],
                "details": results["details"],
                "full_analysis": analysis_text,
                "summary": recommendation,
            },
        }

        return output

    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return {
            "messages": state["messages"],
            "analysis_results": state["analysis_results"],
            "requirement_check": {
                "satisfied": [],
                "missing": ["Error in analysis"],
                "partial": [],
                "summary": f"ERROR: Failed to complete analysis. Details: {str(e)}",
            },
        }


def analyze_format_requirements(state: State):
    """Analyze document format requirements and required attachments."""
    try:
        # Basic search queries with added focus on attachments
        search_queries = [
            "format requirements",
            "document formatting",
            "submission requirements",
            "required attachments",
            "mandatory documents",
            "HUB certification requirements",
            "required forms certificates",
            "required documentation",
        ]

        all_content = []
        for query in search_queries:
            try:
                content = get_relevant_content(query, k=20)
                if content:
                    all_content.append(content)
            except Exception as e:
                print(f"Search error for '{query}': {str(e)}")
                continue

        if not all_content:
            try:
                content = get_relevant_content(
                    "Find any requirements or instructions about documents and formatting",
                    k=40,
                )
                if content:
                    all_content = [content]
            except Exception as e:
                print(f"Fallback search error: {str(e)}")

        combined_content = "\n\n".join(all_content)

        if not combined_content:
            raise ValueError("No content found in vector stores")

        format_prompt = """
        Analyze the RFP document and provide two sections: formatting requirements and a complete list of all mentioned documents.

        Text to analyze:
        {context}

        Part 1 - DOCUMENT FORMAT REQUIREMENTS:
        1. Page Limits:
           - List any requirements about page numbers, total length, section lengths
        2. Font Requirements:
           - List any font type, size requirements
        3. Margins and Spacing:
           - List any margin, indent, line spacing requirements
        4. Headers and Footers:
           - List any header/footer requirements
        5. Organization:
           - List any section organization, tab requirements, binding requirements
        6. Submission Format:
           - Digital/physical copy requirements
           - Number of copies needed
           - Any specific file formats (.pdf, .doc, etc.)

        Part 2 - COMPLETE DOCUMENTS LIST:
        List ALL items that need to be submitted, searching for these keywords:
        - Attachments (e.g., "Attachment A", "Attachment 1")
        - Forms (e.g., submission forms, response forms)
        - Documents (any referenced documents)
        - Papers (any required papers)
        - Resumes (staff/personnel resumes)
        - Letters (reference letters, commitment letters)
        - Proposals (technical/cost proposals)
        - Screenshots (system screenshots, examples)
        - Permits (business permits, operating permits)
        - Certificates (any certifications required)
        - Licenses (business licenses, professional licenses)
        - References (reference documents, case studies)
        - Reports (financial reports, audit reports)
        - Statements (financial statements, bank statements)
        - Proofs (proof of insurance, proof of certification)
        - Samples (work samples, writing samples)
        - Exhibits (any referenced exhibits)

        Format your response exactly like this:

        DOCUMENT FORMAT REQUIREMENTS:
        [List all formatting requirements under the categories above]

        COMPLETE DOCUMENTS LIST:
        - [document name/title]
        - [document name/title]
        ...

        Rules:
        - Use EXACT names/titles as mentioned in the RFP
        - List each document only once
        - Include ALL documents regardless of their mandatory/optional status
        - Do NOT add any explanations or comments to the documents list
        - Do NOT skip any document mentioned in the RFP
        - Include document numbers/letters if specified (e.g., "Attachment A" not just "Attachment")
        """

        format_analysis = llm.invoke(format_prompt.format(context=combined_content))

        if not format_analysis.content:
            raise ValueError("No response from LLM")

        # Initialize the requirements dictionary
        formatted_requirements = {
            "document_format": {
                "page_limits": [],
                "font_specs": [],
                "margins_spacing": [],
                "headers_footers": [],
                "organization": [],
            },
            "mandatory_attachments": [],
            "optional_attachments": [],
            "raw_analysis": format_analysis.content,
        }

        # Parse the LLM response
        current_section = None
        current_subsection = None

        for line in format_analysis.content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for main sections
            if "DOCUMENT FORMAT REQUIREMENTS:" in line:
                current_section = "document_format"
                continue
            elif "REQUIRED ATTACHMENTS:" in line:
                current_section = "required"
                continue
            elif "OPTIONAL ATTACHMENTS:" in line:
                current_section = "optional"
                continue

            # Handle document format subsections
            if current_section == "document_format":
                if "Page Limits:" in line:
                    current_subsection = "page_limits"
                elif "Font Requirements:" in line:
                    current_subsection = "font_specs"
                elif "Margins and Spacing:" in line:
                    current_subsection = "margins_spacing"
                elif "Headers and Footers:" in line:
                    current_subsection = "headers_footers"
                elif "Organization:" in line:
                    current_subsection = "organization"
                elif line.startswith("-") or line.startswith("*"):
                    if current_subsection:
                        requirement = line[1:].strip()
                        # Accept requirements even without explicit tags
                        if requirement:
                            formatted_requirements["document_format"][
                                current_subsection
                            ].append(requirement)

            # Handle attachments - now more inclusive in what we capture
            elif current_section == "required" and (
                line.startswith("-") or line.startswith("*")
            ):
                requirement = line[1:].strip()
                if requirement:
                    # Check for key terms that indicate mandatory requirements
                    is_mandatory = (
                        any(
                            term.lower() in requirement.lower()
                            for term in [
                                "must",
                                "shall",
                                "required",
                                "mandatory",
                                "hub",
                                "certificate",
                                "certification",
                                "insurance",
                                "form",
                                "attachment",
                                "documentation",
                            ]
                        )
                        or "[MANDATORY]" in requirement
                    )
                    if is_mandatory:
                        formatted_requirements["mandatory_attachments"].append(
                            requirement
                        )

            elif current_section == "optional" and (
                line.startswith("-") or line.startswith("*")
            ):
                requirement = line[1:].strip()
                if requirement:
                    # Check for key terms that indicate optional requirements
                    is_optional = (
                        any(
                            term.lower() in requirement.lower()
                            for term in [
                                "may",
                                "optional",
                                "preferred",
                                "if applicable",
                                "if needed",
                                "recommended",
                                "suggested",
                            ]
                        )
                        or "[OPTIONAL]" in requirement
                    )
                    if is_optional:
                        formatted_requirements["optional_attachments"].append(
                            requirement
                        )
                    else:
                        # If not clearly optional, treat as mandatory
                        formatted_requirements["mandatory_attachments"].append(
                            requirement
                        )

        return {
            "messages": state["messages"],
            "analysis_results": {
                **state["analysis_results"],
                "format_requirements": formatted_requirements,
                "format_analysis_text": format_analysis.content,
                "format_context": combined_content,
            },
        }

    except Exception as e:
        print(f"\nDEBUG - Error in format requirements analysis: {str(e)}")
        traceback.print_exc()
        return {
            "messages": state["messages"],
            "analysis_results": {
                **state["analysis_results"],
                "format_requirements": {
                    "document_format": {
                        "page_limits": [],
                        "font_specs": [],
                        "margins_spacing": [],
                        "headers_footers": [],
                        "organization": [],
                    },
                    "mandatory_attachments": [],
                    "optional_attachments": [],
                },
                "format_analysis_text": f"Error occurred: {str(e)}",
                "format_context": "",
            },
        }


def analyze_risks(state: State) -> State:
    """Analyze RFP for risky clauses and contract terms."""
    print("\nStep 4: Performing Risk Analysis...")
    print("-" * 60)

    # Risk-indicating keywords
    risk_keywords = [
        "indemnification",
        "liability",
        "termination",
        "penalty",
        "damages",
        "warranty",
        "compliance",
        "breach",
        "default",
        "force majeure",
        "confidentiality",
        "intellectual property",
        "governing law",
        "jurisdiction",
        "dispute",
        "arbitration",
        "limitation of liability",
        "liquidated damages",
        "insurance requirements",
        "performance bond",
    ]

    risky_statements = []

    print("\nStep 4.1: Searching for statements with risk-indicating keywords...")
    for keyword in risk_keywords:
        try:
            # Get context around risk keywords
            context = hybrid_retrieval(
                eligible_rfp_store,
                f"Find sections containing {keyword}",
                k=5,
                keyword_filter=[keyword],
            )

            if context:
                # Extract statements containing the keyword
                paragraphs = re.split(r"\n\s*\n", context)
                for para in paragraphs:
                    if keyword.lower() in para.lower():
                        risky_statements.append(
                            {
                                "type": "keyword_match",
                                "keyword": keyword,
                                "statement": para.strip(),
                                "context": context,
                            }
                        )

        except Exception as e:
            print(f"Error processing keyword '{keyword}': {str(e)}")

    print("\nStep 4.2: Retrieving additional risk-related content from vector store...")
    risk_queries = [
        "Find sections describing vendor obligations and responsibilities",
        "Find sections about legal requirements and compliance",
        "Find sections discussing penalties or consequences",
        "Find sections about project timeline and deadlines",
        "Find sections about payment terms and financial obligations",
    ]

    for query in risk_queries:
        try:
            context = hybrid_retrieval(eligible_rfp_store, query, k=3)
            if context:
                risky_statements.append(
                    {
                        "type": "vector_store_retrieval",
                        "keyword": query,
                        "statement": context.split("\n\n")[
                            0
                        ].strip(),  # First relevant paragraph
                        "context": context,
                    }
                )
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")

    print(f"\nFound {len(risky_statements)} potentially risky statements:")
    print(
        f"- {sum(1 for s in risky_statements if s['type'] == 'keyword_match')} from keyword matches"
    )
    print(
        f"- {sum(1 for s in risky_statements if s['type'] == 'vector_store_retrieval')} from vector store retrieval"
    )

    print("\nStep 4.3: Analyzing identified statements for risks...")
    print("-" * 60)

    # Format the statements for LLM analysis
    formatted_statements = "\n\n".join(
        [
            f"Statement {i + 1} (Source: {s['type']}, Related to: {s['keyword']}):\n"
            f"Text: {s['statement']}\n"
            f"Context: {s['context'][:200]}..."  # Include some context
            for i, s in enumerate(risky_statements)
        ]
    )

    risk_analysis_prompt = """
    You are an expert legal and contract analyst. Analyze these RFP statements and explain the key risks and threats that a vendor should be aware of before bidding.

    Statements to analyze:
    {statements}

    Provide a clear and direct analysis in this format:

    EXECUTIVE SUMMARY:
    - Brief overview of major risks found (2-3 sentences)
    - Total number of significant risks
    - Risk severity breakdown (High/Medium/Low)

    KEY RISKS AND THREATS:
    [For each major risk found, provide:]

    Risk #[number]:
    THREAT: [One sentence description of the threat]
    SEVERITY: [High/Medium/Low]
    WHY IT MATTERS: [2-3 bullet points explaining the business impact]
    MITIGATION: [1-2 bullet points suggesting how to address it]

    BOTTOM LINE:
    - Clear yes/no recommendation on whether to bid
    - Top 3 risks that need immediate attention
    - Quick actions needed before bidding

    Keep explanations direct and business-focused. Avoid legal jargon. Focus on practical implications for the vendor.
    """

    try:
        # Split statements into smaller chunks if needed
        max_statements_per_chunk = 5
        statement_chunks = [
            risky_statements[i : i + max_statements_per_chunk]
            for i in range(0, len(risky_statements), max_statements_per_chunk)
        ]

        all_analyses = []

        for chunk in statement_chunks:
            chunk_statements = "\n\n".join(
                [
                    f"Statement {i + 1} (Source: {s['type']}, Related to: {s['keyword']}):\n"
                    f"Text: {s['statement']}\n"
                    f"Context: {s['context'][:200]}..."
                    for i, s in enumerate(chunk)
                ]
            )

            risk_result = llm.invoke(
                risk_analysis_prompt.format(statements=chunk_statements)
            )

            if not risk_result.content:
                raise ValueError(f"No response from LLM for chunk analysis")

            all_analyses.append(risk_result.content)

        # Combine all analyses
        combined_analysis = "\n\n".join(all_analyses)

        return {
            "messages": state["messages"],
            "analysis_results": {
                **state["analysis_results"],
                "risk_analysis": {
                    "risky_statements": risky_statements,
                    "raw_analysis": combined_analysis,
                    "statements_analyzed": formatted_statements,
                },
            },
        }

    except Exception as e:
        error_msg = f"Error in risk analysis: {str(e)}\n{traceback.format_exc()}"
        print(f"\nDEBUG - {error_msg}")

        return {
            "messages": state["messages"],
            "analysis_results": {
                **state["analysis_results"],
                "risk_analysis": {
                    "risky_statements": risky_statements,
                    "raw_analysis": f"Error occurred: {error_msg}",
                    "statements_analyzed": formatted_statements,
                },
            },
        }


def print_risk_analysis(risk_analysis_results: dict):
    """Print risk analysis results in a clean, organized way."""
    print("\nRISK ANALYSIS RESULTS")
    print("=" * 80)
    print()

    if not risk_analysis_results:
        print("No risk analysis results available.")
        return

    # Print the LLM analysis
    if "raw_analysis" in risk_analysis_results:
        analysis_text = risk_analysis_results["raw_analysis"]

        # Print Executive Summary
        if "EXECUTIVE SUMMARY:" in analysis_text:
            print("EXECUTIVE SUMMARY")
            print("-" * 20)
            summary_section = analysis_text.split("EXECUTIVE SUMMARY:")[1].split(
                "KEY RISKS AND THREATS:"
            )[0]
            print(summary_section.strip())
            print("\n" + "=" * 80)

        # Print Key Risks
        if "KEY RISKS AND THREATS:" in analysis_text:
            print("\nKEY RISKS AND THREATS")
            print("-" * 20)
            risks_section = analysis_text.split("KEY RISKS AND THREATS:")[1].split(
                "BOTTOM LINE:"
            )[0]
            print(risks_section.strip())
            print("\n" + "=" * 80)

        # Print Bottom Line
        if "BOTTOM LINE:" in analysis_text:
            print("\nBOTTOM LINE")
            print("-" * 20)
            bottom_line = analysis_text.split("BOTTOM LINE:")[1].strip()
            print(bottom_line)

    print("\n" + "=" * 80)


# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("rfp_analysis", analyze_rfp)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("check_requirements_node", check_requirements)
graph_builder.add_node("format_analysis", analyze_format_requirements)
graph_builder.add_node("risk_analysis", analyze_risks)

# Add edges
graph_builder.add_edge(START, "rfp_analysis")
graph_builder.add_edge("rfp_analysis", "chatbot")
graph_builder.add_edge("chatbot", "check_requirements_node")
graph_builder.add_edge("check_requirements_node", "format_analysis")
graph_builder.add_edge("format_analysis", "risk_analysis")
graph_builder.add_edge("risk_analysis", END)

# Compile the graph
graph = graph_builder.compile()


if __name__ == "__main__":
    run_analysis()
