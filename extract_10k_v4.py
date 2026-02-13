"""
10-K Filing Extraction Pipeline (v2)
=====================================
Extracts structured data from SEC 10-K filings using Claude API.
Schema validated against equity research prompt v4 requirements.

Supports both HTML (preferred) and PDF input files.

Usage:
    python extract_10k.py path/to/filing.htm
    python extract_10k.py path/to/filing.pdf
    python extract_10k.py path/to/filings_folder/

Output:
    Creates a JSON extraction file in the ./extractions/ directory.

Setup:
    1. pip install anthropic pymupdf beautifulsoup4 lxml
    2. Set your API key:  set ANTHROPIC_API_KEY=your-key-here  (Windows)
       Or create a .env file with:  ANTHROPIC_API_KEY=your-key-here
"""

import os
import sys
import json
import re
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    print("Missing dependency: anthropic")
    print("Run: pip install anthropic")
    sys.exit(1)

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None  # PDF support optional if using HTML

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # HTML support optional if using PDF


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 16000
CONCURRENT_LIMIT = 5

OUTPUT_DIR = Path("./extractions")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_run_output_dir() -> Path:
    """Create a unique output directory for each extraction run."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


RUN_OUTPUT_DIR = OUTPUT_DIR


# ---------------------------------------------------------------------------
# API Key Loading
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Load API key from environment or .env file."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key

    print("ERROR: No API key found.")
    print("Set it with:  set ANTHROPIC_API_KEY=your-key-here")
    print("Or create a .env file with:  ANTHROPIC_API_KEY=your-key-here")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Text Extraction (PDF and HTML)
# ---------------------------------------------------------------------------
def extract_text_from_pdf(filepath: Path) -> str:
    """Extract text from a PDF file using pymupdf."""
    if fitz is None:
        print("ERROR: pymupdf not installed. Run: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(str(filepath))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def extract_text_from_html(filepath: Path) -> str:
    """Extract text from an EDGAR HTML filing."""
    if BeautifulSoup is None:
        print("ERROR: beautifulsoup4 not installed. Run: pip install beautifulsoup4 lxml")
        sys.exit(1)

    raw = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def extract_text(filepath: Path) -> str:
    """Extract text from either PDF or HTML."""
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(filepath)
    elif suffix in (".htm", ".html"):
        return extract_text_from_html(filepath)
    else:
        return filepath.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Section Splitting
# ---------------------------------------------------------------------------
ITEM_PATTERNS = {
    "item1_business": [
        r'(?i)\bitem\s*1[\.\s]*[-–—]?\s*business\b',
    ],
    "item1a_risk_factors": [
        r'(?i)\bitem\s*1a[\.\s]*[-–—]?\s*risk\s+factors\b',
    ],
    "item1b_unresolved": [
        r'(?i)\bitem\s*1b[\.\s]*[-–—]?\s*unresolved\b',
    ],
    "item1c_cybersecurity": [
        r'(?i)\bitem\s*1c[\.\s]*[-–—]?\s*cyber\s*security\b',
    ],
    "item2_properties": [
        r'(?i)\bitem\s*2[\.\s]*[-–—]?\s*properties\b',
    ],
    "item3_legal": [
        r'(?i)\bitem\s*3[\.\s]*[-–—]?\s*legal\b',
    ],
    "item5_market": [
        r'(?i)\bitem\s*5[\.\s]*[-–—]?\s*market\s+for\b',
    ],
    "item6_reserved": [
        r'(?i)\bitem\s*6[\.\s]*[-–—]?\s*(\[reserved\]|selected|reserved)\b',
    ],
    "item7_mda": [
        r'(?i)\bitem\s*7[\.\s]*[-–—]?\s*management\W*s?\s*discussion\b',
    ],
    "item7a_market_risk": [
        r'(?i)\bitem\s*7a[\.\s]*[-–—]?\s*quantitative\b',
    ],
    "item8_financials": [
        r'(?i)\bitem\s*8[\.\s]*[-–—]?\s*(consolidated\s+)?financial\s+statements\b',
        r'(?i)\bitem\s*8[\.\s]*[-–—]?\s*[\w\s]{0,30}financial\s+statements\b',
    ],
    "item9_disagreements": [
        r'(?i)\bitem\s*9[\.\s]*[-–—]?\s*changes\s+in\b',
    ],
    "item10_directors": [
        r'(?i)\bitem\s*10[\.\s]*[-–—]?\s*directors\b',
    ],
    "item11_compensation": [
        r'(?i)\bitem\s*11[\.\s]*[-–—]?\s*executive\s+compensation\b',
    ],
}

TARGET_SECTIONS = [
    "item1_business",
    "item1a_risk_factors",
    "item7_mda",
    "item8_financials",
]


def find_section_boundaries(text: str) -> list[tuple[str, int]]:
    """Find character positions where each Item section begins."""
    boundaries = []

    for section_name, patterns in ITEM_PATTERNS.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if not matches:
                continue

            if len(matches) == 1:
                boundaries.append((section_name, matches[0].start()))
            else:
                threshold = len(text) * 0.10
                body_matches = [m for m in matches if m.start() > threshold]
                if body_matches:
                    boundaries.append((section_name, body_matches[0].start()))
                else:
                    boundaries.append((section_name, matches[-1].start()))
            break

    boundaries.sort(key=lambda x: x[1])
    return boundaries


def is_cross_reference_stub(section_text: str) -> bool:
    """Detect if a section is just a short cross-reference/pointer to another location.
    
    Some 10-Ks have Items that say things like:
      "The information required by this Item is set forth on pages 45 through 89"
      "See the Consolidated Financial Statements beginning on page F-1"
      "The response to this item is incorporated by reference..."
    """
    # Strip the Item header line itself to look at the body
    lines = [l.strip() for l in section_text.split('\n') if l.strip()]
    # Skip lines that are just the Item header/title
    body_lines = []
    past_header = False
    for line in lines:
        if not past_header:
            # Skip the item heading lines (e.g., "Item 8.", "Financial Statements and Supplementary Data")
            if re.match(r'(?i)^\s*item\s*\d', line):
                past_header = True
                continue
            if past_header is False and len(line) < 120:
                continue
        past_header = True
        body_lines.append(line)

    body_text = ' '.join(body_lines).strip()
    
    # If the body is very short, check for cross-reference language
    if len(body_text) < 500:
        cross_ref_patterns = [
            r'(?i)set\s+forth\s+on\s+pages?\b',
            r'(?i)incorporated\s+by\s+reference',
            r'(?i)included\s+(elsewhere|herein|in\s+this)',
            r'(?i)see\s+(the\s+)?consolidated\s+financial',
            r'(?i)beginning\s+on\s+page',
            r'(?i)appears?\s+(on|at)\s+page',
            r'(?i)financial\s+statements\s+.*\s+follow',
            r'(?i)are\s+included\s+in\s+part\s+',
            r'(?i)information\s+required\s+by\s+this\s+item',
        ]
        for pattern in cross_ref_patterns:
            if re.search(pattern, body_text):
                return True

    return False


# Patterns to locate the actual financial statements content when Item 8 is a stub
FINANCIAL_STATEMENTS_ANCHOR_PATTERNS = [
    r'(?i)report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm',
    r'(?i)consolidated\s+balance\s+sheets?\b',
    r'(?i)consolidated\s+statements?\s+of\s+(income|operations|earnings)',
    r'(?i)notes?\s+to\s+(the\s+)?consolidated\s+financial\s+statements',
]


def find_financial_statements_content(text: str, boundaries: list[tuple[str, int]]) -> str | None:
    """Find the actual financial statements content when Item 8 is a cross-reference stub.
    
    The financial statements typically appear after the last numbered Item section
    in the filing, identifiable by headers like 'Report of Independent Registered
    Public Accounting Firm', 'Consolidated Balance Sheets', etc.
    """
    # Find the earliest anchor for financial statements content
    earliest_anchor = None
    for pattern in FINANCIAL_STATEMENTS_ANCHOR_PATTERNS:
        match = re.search(pattern, text)
        if match:
            if earliest_anchor is None or match.start() < earliest_anchor:
                earliest_anchor = match.start()

    if earliest_anchor is None:
        return None

    # Determine where the financial statements end.
    # They typically end at the start of Part III items (Item 10+) or at the
    # signatures section, or at the end of the document.
    end_markers = [
        r'(?i)\bitem\s*10[\.\s]*[-–—]?\s*directors\b',
        r'(?i)\bitem\s*11[\.\s]*[-–—]?\s*executive\s+compensation\b',
        r'(?i)\bPART\s+III\b',
        r'(?i)\bSIGNATURES\b',
        r'(?i)\bEXHIBIT\s+INDEX\b',
    ]

    end_pos = len(text)
    for pattern in end_markers:
        match = re.search(pattern, text[earliest_anchor:])
        if match:
            candidate = earliest_anchor + match.start()
            if candidate < end_pos and candidate > earliest_anchor:
                end_pos = candidate

    # Also check: if any known Item boundary falls between our anchor and end,
    # don't overshoot into another Item's content. But financial statements
    # are typically outside the Item boundary structure, so this is just safety.
    for bname, bpos in boundaries:
        if bpos > earliest_anchor and bpos < end_pos:
            # Only trim if this is a non-financial-statements item
            if bname not in ("item8_financials",):
                end_pos = bpos

    content = text[earliest_anchor:end_pos].strip()

    # Sanity check: the financial statements section should be substantial
    if len(content) < 5000:
        return None

    return content


def split_into_sections(text: str) -> dict[str, str]:
    """Split full 10-K text into target sections."""
    boundaries = find_section_boundaries(text)

    if not boundaries:
        print("  WARNING: Could not identify any Item boundaries.")
        print("  Falling back to full-text extraction (single API call).")
        return {"full_text": text}

    sections = {}
    for i, (name, start) in enumerate(boundaries):
        if name not in TARGET_SECTIONS:
            continue

        if i + 1 < len(boundaries):
            end = boundaries[i + 1][1]
        else:
            end = len(text)

        section_text = text[start:end].strip()

        # Check for cross-reference stubs — the Item header exists but just
        # points to content located elsewhere in the filing
        if is_cross_reference_stub(section_text):
            print(f"  NOTE: {name} appears to be a cross-reference stub, searching for actual content...")
            actual_content = find_financial_statements_content(text, boundaries)
            if actual_content:
                print(f"  ✓ Found actual content for {name} ({len(actual_content):,} chars)")
                section_text = section_text + "\n\n" + actual_content
            else:
                print(f"  WARNING: {name} is a cross-reference stub and actual content could not be located")

        sections[name] = section_text

    found = list(sections.keys())
    missing = [s for s in TARGET_SECTIONS if s not in sections]
    print(f"  Sections found: {', '.join(found)}")
    if missing:
        print(f"  Sections missing: {', '.join(missing)}")

    return sections


# ---------------------------------------------------------------------------
# Extraction Prompts (v2 — validated against equity research prompt v4)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial document extraction engine. Your task is to read a section of a 10-K filing and produce a structured JSON extraction.

CRITICAL RULES:
1. Extract only. Do not analyze, interpret, or editorialize. Do not assess whether something is good or bad.
2. Preserve the company's own language for key disclosures — risk factors, accounting policies, contractual obligations, revenue recognition.
3. If information is not present in the provided text, use "NOT_PROVIDED". Do not fabricate, infer, or substitute.
4. All numerical values must appear verbatim in the filing. Do not compute, estimate, or round.
5. Cite the Item number and section header for every extraction.
6. Output valid JSON only. No markdown fences, no preamble, no commentary before or after the JSON."""


ITEM1_PROMPT = """Extract the following from this Item 1 — Business section of a 10-K filing.

Output this exact JSON structure. Populate all fields from the text. Use "NOT_PROVIDED" for anything not found.

{
  "business_description": {
    "source": "Item 1 — Business",
    "business_model_summary": "2-3 sentence summary of how the company makes money, in the company's own framing",
    "revenue_model": "subscription / transaction / licensing / advertising / mixed — as described by the company",
    "products_and_services": [
      "list each product or service category as the company defines them"
    ],
    "geographic_markets": [
      "list geographic regions mentioned"
    ],
    "seasonality": "company's description of seasonal patterns, or NOT_PROVIDED",
    "key_customers_disclosed": [
      "any named customers"
    ],
    "customer_concentration": {
      "any_customer_above_10pct": true or false,
      "details": ["specifics if disclosed — customer name, percentage, segment"]
    },
    "competitive_landscape_as_described": "the company's own characterization of its competitive environment — preserve their language",
    "regulatory_environment": "key regulatory disclosures — what regulations govern the business, any licensing requirements",
    "employees": {
      "total": null,
      "breakdown_if_provided": "by segment, geography, or function if disclosed"
    },
    "competitive_advantage_evidence": {
      "switching_costs": {
        "evidence_found": true or false,
        "supporting_text": ["verbatim passages from the filing that suggest switching costs — e.g., integration depth, data migration barriers, training requirements, contractual lock-in, proprietary formats"],
        "source_sections": ["which part of Item 1 this came from"]
      },
      "brand_mindshare": {
        "evidence_found": true or false,
        "supporting_text": ["verbatim passages suggesting brand strength, reputation, or mindshare advantages"],
        "source_sections": []
      },
      "cost_scale_advantages": {
        "evidence_found": true or false,
        "supporting_text": ["verbatim passages suggesting cost advantages from scale, infrastructure, distribution, or operational efficiency"],
        "source_sections": []
      },
      "network_effects": {
        "evidence_found": true or false,
        "supporting_text": ["verbatim passages suggesting the product becomes more valuable as more users adopt it"],
        "source_sections": []
      },
      "regulatory_licensing_barriers": {
        "evidence_found": true or false,
        "supporting_text": ["verbatim passages about regulatory approvals, licenses, permits, or compliance requirements that create barriers to entry"],
        "source_sections": []
      }
    },
    "revenue_durability_indicators": {
      "recurring_revenue_disclosed": true or false,
      "recurring_revenue_percentage_if_stated": "verbatim figure or NOT_PROVIDED",
      "contract_terms_disclosed": "description of typical contract lengths, renewal terms — verbatim language",
      "renewal_rates_disclosed": "verbatim figure or NOT_PROVIDED",
      "backlog_or_remaining_performance_obligations": "verbatim figure or NOT_PROVIDED",
      "long_term_customer_agreements_mentioned": true or false,
      "details": "any additional detail on revenue durability from the filing"
    }
  }
}

IMPORTANT:
- For competitive_advantage_evidence, you are NOT assessing whether advantages exist. You are extracting verbatim text passages that a downstream analyst would use to make that assessment. If there is no evidence for a moat type, set evidence_found to false and leave supporting_text empty.
- For revenue_durability_indicators, look for disclosures about recurring revenue, subscription models, contract structures, remaining performance obligations (ASC 606), and customer retention language.
- Customer concentration may also appear in the Notes to Financial Statements. If this section mentions it, capture it. If not, note it here and it will be cross-checked against Item 8 Notes.

FILING TEXT:
"""


ITEM1A_PROMPT = """Extract the following from this Item 1A — Risk Factors section of a 10-K filing.

Extract EVERY risk factor. Preserve the original ordering — this ordering is deliberate and signals management's view of severity.

Output this exact JSON structure:

{
  "risk_factors": {
    "source": "Item 1A — Risk Factors",
    "total_risk_factors_count": null,
    "risks": [
      {
        "rank_order": 1,
        "heading": "exact heading text from filing",
        "summary": "1-2 sentence summary of the risk",
        "verbatim_key_sentence": "the single sentence from this risk factor that most directly states the potential impact on the business",
        "category": "one of: regulatory, competitive, operational, financial, macroeconomic, technology, legal, customer_concentration, key_personnel, cybersecurity, supply_chain, geopolitical, esg, other"
      }
    ]
  }
}

RULES:
- The rank_order must reflect the order the risk appears in the filing, starting at 1.
- The heading must be the exact text of the risk factor heading/title as written.
- The verbatim_key_sentence must be a single sentence copied exactly from the filing — the one that most clearly states the potential business impact.
- Categorize each risk into exactly one category. If a risk spans multiple categories, choose the primary one.
- Do NOT skip any risk factors. Extract all of them, even if there are 30+.
- Set total_risk_factors_count to the actual number of risk factors extracted.

FILING TEXT:
"""


ITEM7_PROMPT = """Extract the following from this Item 7 — MD&A section of a 10-K filing.

Output this exact JSON structure. Populate all fields from the text. Use "NOT_PROVIDED" for anything not found.

{
  "mda": {
    "source": "Item 7 — MD&A",
    "management_narrative_summary": "3-5 sentence summary of management's overall narrative about the year's performance",
    "key_performance_drivers": [
      {
        "driver": "description of the performance driver",
        "direction": "positive or negative",
        "magnitude_if_stated": "verbatim number/percentage from filing, or NOT_PROVIDED"
      }
    ],
    "known_trends_and_uncertainties": [
      "verbatim or near-verbatim language about each forward-looking trend, known uncertainty, or condition management expects to affect future results"
    ],
    "segment_discussion": [
      {
        "segment_name": "",
        "revenue": "verbatim figure or NOT_PROVIDED",
        "operating_income": "verbatim figure or NOT_PROVIDED",
        "management_commentary": "key points from management's discussion of this segment's performance"
      }
    ],
    "margin_commentary": {
      "gross_margin_discussion": "what management says about gross margin trends and drivers — preserve their attribution language",
      "operating_margin_discussion": "what management says about operating margin trends and drivers",
      "factors_cited_for_margin_changes": [
        "each specific factor management cites for margin expansion or compression"
      ]
    },
    "pricing_commentary": {
      "pricing_actions_discussed": true or false,
      "details": "management's description of any pricing changes, pricing environment, or pricing strategy discussed in MD&A"
    },
    "management_emphasized_kpis": [
      "list each metric or KPI that management highlights as important or uses to frame the business performance discussion — e.g., ARR, net retention rate, organic growth, adjusted EBITDA, free cash flow conversion"
    ],
    "liquidity_and_capital_resources": {
      "cash_position_discussed": "management's characterization of the company's liquidity position",
      "capital_allocation_priorities_stated": [
        "each priority in the order management mentions them"
      ],
      "share_repurchase_discussion": "what management says about buybacks in the narrative",
      "dividend_discussion": "what management says about dividends in the narrative"
    },
    "share_repurchase_data": {
      "shares_repurchased_current_year": "verbatim figure or NOT_PROVIDED",
      "amount_spent_on_repurchases": "verbatim dollar amount or NOT_PROVIDED",
      "remaining_authorization": "verbatim figure or NOT_PROVIDED",
      "source_section": "where in MD&A this was disclosed"
    },
    "dividend_data": {
      "dividends_per_share": "verbatim figure or NOT_PROVIDED",
      "total_dividends_paid": "verbatim figure or NOT_PROVIDED",
      "source_section": "where in MD&A this was disclosed"
    },
    "capital_expenditure_discussion": {
      "total_capex_discussed": "verbatim figure or NOT_PROVIDED",
      "maintenance_vs_growth_split_if_disclosed": "verbatim or NOT_PROVIDED — many companies do not break this out",
      "capex_by_segment_if_disclosed": [
        {
          "segment": "",
          "amount": "verbatim"
        }
      ],
      "management_capex_outlook": "any forward-looking capex guidance or commentary"
    },
    "critical_accounting_estimates": [
      {
        "estimate": "name of the critical accounting estimate",
        "description": "brief description of the judgment involved and why management considers it critical"
      }
    ],
    "forward_looking_statements": [
      "verbatim or near-verbatim forward-looking language — management expectations, outlook, guidance"
    ]
  }
}

IMPORTANT:
- For share_repurchase_data and dividend_data, extract the actual dollar amounts and share counts, not just the narrative discussion. These are often in the Liquidity and Capital Resources subsection.
- For capital_expenditure_discussion, look for both the total capex figure and any qualitative breakdown (maintenance vs growth, by segment). The total figure is usually in Liquidity and Capital Resources.
- For management_emphasized_kpis, capture the metrics management uses to frame success — these reveal what management thinks matters most. Include non-GAAP metrics they highlight.
- For pricing_commentary, look for any discussion of pricing actions, pricing environment, ability to pass through cost increases, or competitive pricing pressure.
- For known_trends_and_uncertainties, this is a required SEC disclosure. Management must discuss known trends, events, or uncertainties that could materially affect results. Capture this language verbatim.

FILING TEXT:
"""


ITEM8_PROMPT = """Extract the following from this Item 8 — Financial Statements and Notes section of a 10-K filing.

Focus on the NOTES to financial statements, not the face financial statements (income statement, balance sheet, cash flow statement). Extract all numerical values verbatim.

Output this exact JSON structure. Populate all fields. Use "NOT_PROVIDED" for anything not found.

{
  "financial_statements_and_notes": {
    "source": "Item 8 — Financial Statements and Supplementary Data",

    "revenue_recognition": {
      "policy_summary": "2-3 sentence summary of the revenue recognition approach",
      "verbatim_key_language": "the most important paragraph(s) of the revenue recognition policy, copied verbatim — this is auditor-reviewed language, do not paraphrase",
      "performance_obligations_description": "how the company identifies and satisfies performance obligations",
      "note_reference": "Note number (e.g., Note 2)"
    },

    "debt_and_borrowings": {
      "total_long_term_debt": "verbatim figure or NOT_PROVIDED",
      "total_short_term_debt": "verbatim figure or NOT_PROVIDED",
      "instruments": [
        {
          "description": "name/type of instrument (e.g., 4.25% Senior Notes due 2028)",
          "principal_amount": "verbatim",
          "interest_rate": "verbatim",
          "maturity_date": "verbatim",
          "covenants_noted": "any covenant language associated with this instrument"
        }
      ],
      "maturity_schedule": [
        {
          "year": "",
          "amount": "verbatim"
        }
      ],
      "credit_facility": {
        "total_capacity": "verbatim or NOT_PROVIDED",
        "drawn_amount": "verbatim or NOT_PROVIDED",
        "available": "verbatim or NOT_PROVIDED",
        "expiration": "",
        "key_covenants": [
          "each financial covenant — e.g., maximum leverage ratio, minimum interest coverage"
        ]
      },
      "note_reference": "Note number"
    },

    "lease_obligations": {
      "operating_lease_liability_total": "verbatim or NOT_PROVIDED",
      "finance_lease_liability_total": "verbatim or NOT_PROVIDED",
      "operating_lease_rou_asset": "verbatim or NOT_PROVIDED",
      "maturity_schedule_if_provided": [
        {
          "year": "",
          "amount": "verbatim"
        }
      ],
      "note_reference": "Note number"
    },

    "goodwill_and_intangibles": {
      "goodwill_balance": "verbatim or NOT_PROVIDED",
      "total_intangibles": "verbatim or NOT_PROVIDED",
      "goodwill_by_segment_if_provided": [
        {
          "segment": "",
          "amount": "verbatim"
        }
      ],
      "impairment_charges_current_year": "verbatim amount or NOT_PROVIDED",
      "impairment_history": [
        {
          "year": "",
          "amount": "verbatim",
          "description": "what was impaired and why, if disclosed"
        }
      ],
      "impairment_testing_methodology": "brief description of how the company tests for impairment — qualitative vs quantitative, discount rate if disclosed",
      "note_reference": "Note number"
    },

    "contractual_obligations": {
      "table_provided": true or false,
      "obligations": [
        {
          "category": "e.g., Long-term debt, Operating leases, Purchase obligations",
          "total": "verbatim",
          "less_than_1yr": "verbatim",
          "1_to_3yr": "verbatim",
          "3_to_5yr": "verbatim",
          "more_than_5yr": "verbatim"
        }
      ],
      "source_location": "where in the filing this table appears (e.g., MD&A or Notes)"
    },

    "pension_and_postretirement": {
      "has_defined_benefit_plan": true or false,
      "funded_status": "verbatim or NOT_PROVIDED",
      "benefit_obligation": "verbatim or NOT_PROVIDED",
      "plan_assets": "verbatim or NOT_PROVIDED",
      "note_reference": "Note number"
    },

    "contingencies_and_litigation": {
      "material_matters_disclosed": [
        {
          "description": "brief description of the matter",
          "status": "pending / settled / ongoing / other",
          "potential_exposure_if_stated": "verbatim amount or range, or NOT_PROVIDED",
          "accrual_recorded": "verbatim amount or NOT_PROVIDED"
        }
      ],
      "note_reference": "Note number"
    },

    "segment_financial_data": {
      "number_of_segments": null,
      "segments": [
        {
          "name": "",
          "revenue": "verbatim",
          "operating_income": "verbatim",
          "total_assets": "verbatim or NOT_PROVIDED",
          "depreciation_amortization": "verbatim or NOT_PROVIDED",
          "capital_expenditures": "verbatim or NOT_PROVIDED"
        }
      ],
      "note_reference": "Note number"
    },

    "customer_concentration_from_notes": {
      "any_customer_above_10pct": true or false,
      "details": [
        "each customer disclosed with their revenue percentage or description — verbatim"
      ],
      "note_reference": "Note number — often in Revenue note or Segment note"
    },

    "stock_based_compensation": {
      "total_expense_current_year": "verbatim or NOT_PROVIDED",
      "by_type_if_disclosed": {
        "stock_options": "verbatim or NOT_PROVIDED",
        "rsus": "verbatim or NOT_PROVIDED",
        "psus": "verbatim or NOT_PROVIDED",
        "espp": "verbatim or NOT_PROVIDED"
      },
      "note_reference": "Note number"
    },

    "acquisitions_current_year": [
      {
        "target": "name of acquired company/business",
        "date": "closing date",
        "purchase_price": "verbatim total consideration",
        "goodwill_recognized": "verbatim or NOT_PROVIDED",
        "intangibles_recognized": "verbatim or NOT_PROVIDED",
        "description": "brief description of what was acquired and strategic rationale if stated"
      }
    ],

    "acquisition_related_charges": [
      {
        "description": "what the charge relates to",
        "amount": "verbatim",
        "type": "impairment / integration / write-down / restructuring / transaction_costs",
        "related_acquisition_if_identifiable": "name of the acquisition this charge relates to, or NOT_PROVIDED",
        "source_section": "which Note or section this came from"
      }
    ],

    "divestitures_current_year": [
      {
        "description": "what was divested",
        "date": "",
        "proceeds": "verbatim or NOT_PROVIDED",
        "gain_loss": "verbatim or NOT_PROVIDED"
      }
    ],

    "restructuring_charges": {
      "amount": "verbatim or NOT_PROVIDED",
      "description": "nature of the restructuring — what is being restructured and why",
      "note_reference": "Note number"
    },

    "related_party_transactions": {
      "any_disclosed": true or false,
      "details": "description of any related party transactions, or NOT_PROVIDED",
      "note_reference": "Note number"
    },

    "subsequent_events": "any post-period-end events disclosed in the Notes, verbatim, or NOT_PROVIDED"
  }
}

IMPORTANT:
- For revenue_recognition, copy the policy language VERBATIM. This is auditor-reviewed and paraphrasing can alter its substance. Include the full description of performance obligations.
- For customer_concentration_from_notes, check the Revenue note and the Segment note. Customer concentration is often disclosed here even if not in Item 1.
- For acquisition_related_charges, look for integration costs, transaction costs, restructuring charges tied to acquisitions, and any write-downs of acquired assets. These are often scattered across multiple Notes.
- For goodwill impairment_history, extract any mentions of prior-year impairments, even if discussed briefly. The synthesis stage needs this to track value destruction from acquisitions over time.
- For debt instruments, capture EVERY instrument with its rate, maturity, and amount. The synthesis stage needs the full maturity schedule to identify refinancing cliffs.
- For contingencies_and_litigation, include the potential exposure range if disclosed — companies often state a range of possible loss.

FILING TEXT:
"""


FULL_TEXT_PROMPT = """Extract the following from this 10-K filing. The filing could not be automatically split into individual Items, so you are receiving the full text.

Identify and extract from these sections if you can find them:
- Item 1 (Business)
- Item 1A (Risk Factors)
- Item 7 (MD&A)
- Item 8 (Financial Statements & Notes)

Output valid JSON with this top-level structure:

{
  "metadata": {
    "company_name": "",
    "ticker": "",
    "fiscal_year_end": "",
    "filing_date": "",
    "filing_type": "10-K",
    "extraction_mode": "full_text",
    "notes": "Filing could not be automatically split into sections"
  },
  "business_description": {
    "source": "Item 1 — Business",
    "business_model_summary": "",
    "revenue_model": "",
    "products_and_services": [],
    "geographic_markets": [],
    "seasonality": "NOT_PROVIDED",
    "key_customers_disclosed": [],
    "customer_concentration": { "any_customer_above_10pct": false, "details": [] },
    "competitive_landscape_as_described": "",
    "regulatory_environment": "",
    "employees": { "total": null, "breakdown_if_provided": "" },
    "competitive_advantage_evidence": {
      "switching_costs": { "evidence_found": false, "supporting_text": [], "source_sections": [] },
      "brand_mindshare": { "evidence_found": false, "supporting_text": [], "source_sections": [] },
      "cost_scale_advantages": { "evidence_found": false, "supporting_text": [], "source_sections": [] },
      "network_effects": { "evidence_found": false, "supporting_text": [], "source_sections": [] },
      "regulatory_licensing_barriers": { "evidence_found": false, "supporting_text": [], "source_sections": [] }
    },
    "revenue_durability_indicators": {
      "recurring_revenue_disclosed": false,
      "recurring_revenue_percentage_if_stated": "NOT_PROVIDED",
      "contract_terms_disclosed": "NOT_PROVIDED",
      "renewal_rates_disclosed": "NOT_PROVIDED",
      "backlog_or_remaining_performance_obligations": "NOT_PROVIDED",
      "long_term_customer_agreements_mentioned": false,
      "details": "NOT_PROVIDED"
    }
  },
  "risk_factors": {
    "source": "Item 1A — Risk Factors",
    "total_risk_factors_count": null,
    "risks": []
  },
  "mda": {
    "source": "Item 7 — MD&A",
    "management_narrative_summary": "",
    "key_performance_drivers": [],
    "known_trends_and_uncertainties": [],
    "segment_discussion": [],
    "margin_commentary": { "gross_margin_discussion": "", "operating_margin_discussion": "", "factors_cited_for_margin_changes": [] },
    "pricing_commentary": { "pricing_actions_discussed": false, "details": "" },
    "management_emphasized_kpis": [],
    "liquidity_and_capital_resources": { "cash_position_discussed": "", "capital_allocation_priorities_stated": [], "share_repurchase_discussion": "", "dividend_discussion": "" },
    "share_repurchase_data": { "shares_repurchased_current_year": "NOT_PROVIDED", "amount_spent_on_repurchases": "NOT_PROVIDED", "remaining_authorization": "NOT_PROVIDED", "source_section": "" },
    "dividend_data": { "dividends_per_share": "NOT_PROVIDED", "total_dividends_paid": "NOT_PROVIDED", "source_section": "" },
    "capital_expenditure_discussion": { "total_capex_discussed": "NOT_PROVIDED", "maintenance_vs_growth_split_if_disclosed": "NOT_PROVIDED", "capex_by_segment_if_disclosed": [], "management_capex_outlook": "" },
    "critical_accounting_estimates": [],
    "forward_looking_statements": []
  },
  "financial_statements_and_notes": {
    "source": "Item 8 — Financial Statements and Supplementary Data",
    "revenue_recognition": { "policy_summary": "", "verbatim_key_language": "", "performance_obligations_description": "", "note_reference": "" },
    "debt_and_borrowings": { "total_long_term_debt": "NOT_PROVIDED", "total_short_term_debt": "NOT_PROVIDED", "instruments": [], "maturity_schedule": [], "credit_facility": { "total_capacity": "NOT_PROVIDED", "drawn_amount": "NOT_PROVIDED", "available": "NOT_PROVIDED", "expiration": "", "key_covenants": [] }, "note_reference": "" },
    "lease_obligations": { "operating_lease_liability_total": "NOT_PROVIDED", "finance_lease_liability_total": "NOT_PROVIDED", "operating_lease_rou_asset": "NOT_PROVIDED", "maturity_schedule_if_provided": [], "note_reference": "" },
    "goodwill_and_intangibles": { "goodwill_balance": "NOT_PROVIDED", "total_intangibles": "NOT_PROVIDED", "goodwill_by_segment_if_provided": [], "impairment_charges_current_year": "NOT_PROVIDED", "impairment_history": [], "impairment_testing_methodology": "", "note_reference": "" },
    "contractual_obligations": { "table_provided": false, "obligations": [], "source_location": "" },
    "pension_and_postretirement": { "has_defined_benefit_plan": false, "funded_status": "NOT_PROVIDED", "benefit_obligation": "NOT_PROVIDED", "plan_assets": "NOT_PROVIDED", "note_reference": "" },
    "contingencies_and_litigation": { "material_matters_disclosed": [], "note_reference": "" },
    "segment_financial_data": { "number_of_segments": null, "segments": [], "note_reference": "" },
    "customer_concentration_from_notes": { "any_customer_above_10pct": false, "details": [], "note_reference": "" },
    "stock_based_compensation": { "total_expense_current_year": "NOT_PROVIDED", "by_type_if_disclosed": {}, "note_reference": "" },
    "acquisitions_current_year": [],
    "acquisition_related_charges": [],
    "divestitures_current_year": [],
    "restructuring_charges": { "amount": "NOT_PROVIDED", "description": "", "note_reference": "" },
    "related_party_transactions": { "any_disclosed": false, "details": "NOT_PROVIDED", "note_reference": "" },
    "subsequent_events": "NOT_PROVIDED"
  }
}

Populate every field you can find in the text. Use "NOT_PROVIDED" for anything you cannot locate.

FILING TEXT:
"""


SECTION_PROMPTS = {
    "item1_business": ITEM1_PROMPT,
    "item1a_risk_factors": ITEM1A_PROMPT,
    "item7_mda": ITEM7_PROMPT,
    "item8_financials": ITEM8_PROMPT,
    "full_text": FULL_TEXT_PROMPT,
}


# ---------------------------------------------------------------------------
# API Calls
# ---------------------------------------------------------------------------
async def extract_section(
    client: anthropic.AsyncAnthropic,
    section_name: str,
    section_text: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict | None]:
    """Send a single section to Claude for extraction."""

    prompt_template = SECTION_PROMPTS.get(section_name)
    if not prompt_template:
        print(f"  WARNING: No prompt template for section '{section_name}', skipping.")
        return section_name, None

    max_chars = 400_000
    if len(section_text) > max_chars:
        print(f"  WARNING: {section_name} is very long ({len(section_text):,} chars), truncating to {max_chars:,}")
        section_text = section_text[:max_chars]

    user_message = prompt_template + section_text

    async with semaphore:
        print(f"  Extracting {section_name} ({len(section_text):,} chars)...")
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_text = response.content[0].text.strip()

            # Clean up common issues
            raw_text = re.sub(r'^```json\s*', '', raw_text)
            raw_text = re.sub(r'\s*```$', '', raw_text)

            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group()

            result = json.loads(raw_text)

            # Log token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            print(f"  ✓ {section_name} complete ({input_tokens:,} in / {output_tokens:,} out)")

            return section_name, result

        except json.JSONDecodeError as e:
            print(f"  ✗ {section_name}: Failed to parse JSON response")
            print(f"    Error: {e}")
            debug_path = RUN_OUTPUT_DIR / f"debug_{section_name}.txt"
            debug_path.write_text(raw_text, encoding="utf-8")
            print(f"    Raw response saved to {debug_path}")
            return section_name, None

        except anthropic.APIError as e:
            print(f"  ✗ {section_name}: API error — {e}")
            return section_name, None


async def run_extractions(sections: dict[str, str], api_key: str) -> dict:
    """Run all section extractions concurrently."""
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    tasks = [
        extract_section(client, name, text, semaphore)
        for name, text in sections.items()
    ]

    results = await asyncio.gather(*tasks)
    return {name: data for name, data in results if data is not None}


# ---------------------------------------------------------------------------
# Assembly & Validation
# ---------------------------------------------------------------------------
def assemble_extraction(section_results: dict, filepath: Path) -> dict:
    """Merge per-section extractions into a single 10-K extraction."""

    if "full_text" in section_results:
        extraction = section_results["full_text"]
        extraction.setdefault("metadata", {})
        extraction["metadata"]["source_file"] = filepath.name
        extraction["metadata"]["extraction_timestamp"] = datetime.now().isoformat()
        extraction["metadata"]["extraction_mode"] = "full_text"
        return extraction

    extraction = {
        "metadata": {
            "source_file": filepath.name,
            "filing_type": "10-K",
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_mode": "per_section",
            "sections_extracted": list(section_results.keys()),
            "sections_failed": [
                s for s in TARGET_SECTIONS if s not in section_results
            ],
        }
    }

    section_key_map = {
        "item1_business": "business_description",
        "item1a_risk_factors": "risk_factors",
        "item7_mda": "mda",
        "item8_financials": "financial_statements_and_notes",
    }

    for section_name, json_key in section_key_map.items():
        if section_name in section_results:
            result = section_results[section_name]
            if json_key in result:
                extraction[json_key] = result[json_key]
            else:
                extraction[json_key] = result
        else:
            extraction[json_key] = "NOT_PROVIDED"

    return extraction


def validate_extraction(extraction: dict) -> list[str]:
    """Run quality checks on the assembled extraction."""
    warnings = []

    meta = extraction.get("metadata", {})
    if meta.get("sections_failed"):
        warnings.append(
            f"Failed to extract sections: {', '.join(meta['sections_failed'])}"
        )

    # --- Business Description checks ---
    biz = extraction.get("business_description", {})
    if biz == "NOT_PROVIDED":
        warnings.append("Business description section entirely missing")
    elif isinstance(biz, dict):
        if not biz.get("business_model_summary") or biz["business_model_summary"] == "NOT_PROVIDED":
            warnings.append("No business model summary extracted")

        # Check competitive advantage evidence
        cae = biz.get("competitive_advantage_evidence", {})
        if isinstance(cae, dict):
            moat_types_with_evidence = sum(
                1 for v in cae.values()
                if isinstance(v, dict) and v.get("evidence_found") is True
            )
            if moat_types_with_evidence == 0:
                warnings.append("No competitive advantage evidence found for any moat type — verify against filing")

        # Check revenue durability
        rdi = biz.get("revenue_durability_indicators", {})
        if isinstance(rdi, dict):
            all_not_provided = all(
                v == "NOT_PROVIDED" or v is False
                for k, v in rdi.items()
                if k not in ("details",)
            )
            if all_not_provided:
                warnings.append("No revenue durability indicators extracted — verify against filing")

    # --- Risk Factors checks ---
    risks = extraction.get("risk_factors", {})
    if risks == "NOT_PROVIDED":
        warnings.append("Risk factors section entirely missing")
    elif isinstance(risks, dict):
        risk_list = risks.get("risks", [])
        if len(risk_list) == 0:
            warnings.append("No individual risk factors extracted")
        elif len(risk_list) < 5:
            warnings.append(f"Only {len(risk_list)} risk factors extracted — unusually low, verify completeness")

        # Check ordering integrity
        if risk_list:
            orders = [r.get("rank_order", 0) for r in risk_list]
            if orders != sorted(orders):
                warnings.append("Risk factor rank_order is not sequential — ordering may be corrupted")

    # --- MD&A checks ---
    mda = extraction.get("mda", {})
    if mda == "NOT_PROVIDED":
        warnings.append("MD&A section entirely missing")
    elif isinstance(mda, dict):
        if not mda.get("key_performance_drivers"):
            warnings.append("No key performance drivers extracted from MD&A")
        if not mda.get("known_trends_and_uncertainties"):
            warnings.append("No known trends and uncertainties extracted — this is a required SEC disclosure")
        if not mda.get("management_emphasized_kpis"):
            warnings.append("No management-emphasized KPIs extracted from MD&A")

        # Check capital allocation data
        srd = mda.get("share_repurchase_data", {})
        if isinstance(srd, dict):
            if srd.get("amount_spent_on_repurchases") == "NOT_PROVIDED":
                warnings.append("Share repurchase dollar amount not extracted — verify if company has buyback program")

        ced = mda.get("capital_expenditure_discussion", {})
        if isinstance(ced, dict):
            if ced.get("total_capex_discussed") == "NOT_PROVIDED":
                warnings.append("Total capex figure not extracted from MD&A — check Liquidity and Capital Resources section")

    # --- Financial Statements checks ---
    fin = extraction.get("financial_statements_and_notes", {})
    if fin == "NOT_PROVIDED":
        warnings.append("Financial statements & notes section entirely missing")
    elif isinstance(fin, dict):
        # Debt
        debt = fin.get("debt_and_borrowings", {})
        if isinstance(debt, dict):
            if not debt.get("instruments"):
                warnings.append("No debt instruments extracted — verify if company has debt")
            if not debt.get("maturity_schedule"):
                warnings.append("No debt maturity schedule extracted — critical for balance sheet assessment")

        # Revenue recognition
        rev = fin.get("revenue_recognition", {})
        if isinstance(rev, dict):
            vkl = rev.get("verbatim_key_language", "")
            if not vkl or vkl == "NOT_PROVIDED":
                warnings.append("Revenue recognition verbatim language not extracted — this should always be present")

        # Goodwill
        gw = fin.get("goodwill_and_intangibles", {})
        if isinstance(gw, dict):
            if gw.get("goodwill_balance") == "NOT_PROVIDED":
                warnings.append("Goodwill balance not extracted — verify if company has goodwill")

        # Customer concentration cross-check
        cc = fin.get("customer_concentration_from_notes", {})
        if isinstance(cc, dict):
            biz_cc = biz.get("customer_concentration", {}) if isinstance(biz, dict) else {}
            if (isinstance(biz_cc, dict) and biz_cc.get("any_customer_above_10pct") is True
                    and cc.get("any_customer_above_10pct") is not True):
                warnings.append("Item 1 discloses customer concentration but Notes extraction did not — possible extraction gap")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_single_file(filepath: Path, api_key: str) -> Path | None:
    """Process a single 10-K filing."""
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name}")
    print(f"{'='*60}")

    # Step 1: Extract text
    print("\n[Step 1] Extracting text from file...")
    try:
        full_text = extract_text(filepath)
    except Exception as e:
        print(f"  ERROR: Failed to extract text — {e}")
        return None

    print(f"  Extracted {len(full_text):,} characters")

    if len(full_text) < 1000:
        print("  ERROR: Extracted text is too short. File may be corrupted or not a 10-K.")
        return None

    # Step 2: Split into sections
    print("\n[Step 2] Identifying section boundaries...")
    sections = split_into_sections(full_text)

    # Step 3: Run extractions
    print("\n[Step 3] Running Claude API extractions...")
    section_results = asyncio.run(run_extractions(sections, api_key))

    if not section_results:
        print("  ERROR: All extractions failed.")
        return None

    # Step 4: Assemble
    print("\n[Step 4] Assembling extraction...")
    extraction = assemble_extraction(section_results, filepath)

    # Step 5: Validate
    print("\n[Step 5] Validating extraction...")
    warnings = validate_extraction(extraction)
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    ⚠ {w}")
        extraction["metadata"]["validation_warnings"] = warnings
    else:
        print("  ✓ All checks passed")

    # Step 6: Save
    output_name = filepath.stem + "_extraction.json"
    output_path = RUN_OUTPUT_DIR / output_name
    output_path.write_text(
        json.dumps(extraction, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  ✓ Saved to {output_path}")

    # Print summary stats
    if isinstance(extraction.get("risk_factors"), dict):
        rc = len(extraction["risk_factors"].get("risks", []))
        print(f"  Risk factors extracted: {rc}")
    if isinstance(extraction.get("financial_statements_and_notes"), dict):
        di = len(extraction["financial_statements_and_notes"].get("debt_and_borrowings", {}).get("instruments", []))
        print(f"  Debt instruments extracted: {di}")
        ai = len(extraction["financial_statements_and_notes"].get("acquisitions_current_year", []))
        if ai > 0:
            print(f"  Acquisitions extracted: {ai}")

    return output_path


def main():
    global RUN_OUTPUT_DIR
    RUN_OUTPUT_DIR = create_run_output_dir()

    parser = argparse.ArgumentParser(
        description="Extract structured data from 10-K filings using Claude API (v2)"
    )
    parser.add_argument(
        "path",
        help="Path to a single filing (PDF/HTML) or a folder of filings",
    )
    args = parser.parse_args()

    api_key = load_api_key()
    target = Path(args.path)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(
            f for f in target.iterdir()
            if f.suffix.lower() in (".pdf", ".htm", ".html", ".txt")
        )
        if not files:
            print(f"No supported files found in {target}")
            sys.exit(1)
        print(f"Found {len(files)} filing(s) to process")
    else:
        print(f"Path not found: {target}")
        sys.exit(1)

    results = []
    for filepath in files:
        output_path = process_single_file(filepath, api_key)
        results.append((filepath.name, output_path))

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    succeeded = sum(1 for _, p in results if p is not None)
    failed = len(results) - succeeded
    print(f"  Processed: {len(results)} file(s)")
    print(f"  Succeeded: {succeeded}")
    if failed > 0:
        print(f"  Failed:    {failed}")
    print(f"  Output:    {RUN_OUTPUT_DIR.resolve()}")
    for name, path in results:
        status = f"✓ {path.name}" if path else "✗ FAILED"
        print(f"    {name} → {status}")


if __name__ == "__main__":
    main()
