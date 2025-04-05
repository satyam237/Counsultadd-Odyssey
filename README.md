# Counsultadd-Odyssey

## RFP Analysis and Bid Eligibility Checker

A powerful tool designed to analyze Request for Proposal (RFP) documents and assess company eligibility for bid submission. This tool uses advanced language models and semantic search to extract requirements, analyze formatting specifications, and evaluate company compliance.

## Features

- **RFP Document Analysis**
  - Extracts all formatting requirements
  - Identifies required documents and attachments
  - Lists all certificates, forms, and submissions needed
  - Provides detailed formatting specifications

- **Company Eligibility Check**
  - Compares company data against RFP requirements
  - Analyzes experience requirements
  - Verifies documentation compliance
  - Checks personnel qualifications
  - Validates certifications and credentials

- **Comprehensive Reporting**
  - Detailed analysis of met and unmet requirements
  - Clear recommendations for bid submission
  - Gap analysis for missing requirements
  - Formatted output for easy review

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google Cloud API credentials for Gemini model
- Sufficient storage for vector databases

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/satyam237/Counsultadd-Odyssey.git
   cd Counsultadd-Odyssey
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

## Project Structure

```
Counsultadd-Odyssey/
├── src/
│   ├── llm_analysis.py      # Main analysis logic
│   └── utils/               # Utility functions
├── data/
│   ├── docs/               # Company documents
│   └── vector_stores/      # Vector databases
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Usage

1. Place your RFP document in the appropriate directory
2. Add your company data in `data/docs/company_data.txt`
3. Run the analysis:
   ```python
   from src.llm_analysis import run_analysis
   run_analysis()
   ```

The tool will:
1. Analyze the RFP document
2. Extract requirements and formatting specifications
3. Compare with company data
4. Provide detailed analysis and recommendations

## Output Format

The analysis provides structured output including:

1. **Document Format Requirements**
   - Page limits
   - Font specifications
   - Margins and spacing
   - Headers and footers
   - Organization requirements

2. **Required Documents List**
   - All required attachments
   - Forms and certificates
   - Supporting documents

3. **Eligibility Analysis**
   - Requirements met
   - Requirements not met
   - Partial matches
   - Final recommendation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- Built with LangChain, LangGraph and Google's Gemini model
- Uses ChromaDB for vector storage
- Implements semantic search for accurate requirement matching
