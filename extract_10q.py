"""
10-Q Filing Extraction Pipeline
=================================
Extracts structured data from SEC 10-Q filings using Claude API.
Schema validated against equity research prompt v4 requirements.

The 10-Q extraction is narrower than the 10-K. The v4 prompt uses 10-Q data for:
  1. Quarterly MD&A narrative — emerging trends, updated guidance, quarterly performance drivers
  2. Updated risk factors — new or modified risks since the last 10-K
  3. Material interim updates — debt changes, acquisitions, litigation, guidance revisions

Supports both HTML (preferred) and PDF input files.

Usage:
    python extract_10q.py path/to/filing.htm
    python extract_10q.py path/to/filing.pdf
    python extract_10q.py path/to/filings_folder/

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
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 12000
CONCURRENT_LIMIT = 5

OUTPUT_DIR = Path("./extractions")
OUTPUT_DIR.mkdir(exist_ok=True)


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

# 10-Q Item patterns — different numbering than 10-K
# Part I is Financial Information, Part II is Other Information
ITEM_PATTERNS_10Q = {
    # Part I
    "part1_item1_financials": [
        r'(?i)\bitem\s*1[\.\s]*[-–—]?\s*financial\s+statements\b',
    ],
    "part1_item2_mda": [
        r'(?i)\bitem\s*2[\.\s]*[-–—]?\s*management\W*s?\s*discussion\b',
    ],
    "part1_item3_market_risk": [
        r'(?i)\bitem\s*3[\.\s]*[-–—]?\s*quantitative\b',
    ],
    "part1_item4_controls": [
        r'(?i)\bitem\s*4[\.\s]*[-–—]?\s*controls\b',
    ],
    # Part II
    "part2_item1_legal": [
        r'(?i)part\s*ii.*?\bitem\s*1[\.\s]*[-–—]?\s*legal\b',
        r'(?i)\bitem\s*1[\.\s]*[-–—]?\s*legal\s+proceedings\b',
    ],
    "part2_item1a_risk_factors": [
        r'(?i)\bitem\s*1a[\.\s]*[-–—]?\s*risk\s+factors\b',
    ],
    "part2_item2_equity": [
        r'(?i)\bitem\s*2[\.\s]*[-–—]?\s*unregistered\b',
        r'(?i)\bitem\s*2[\.\s]*[-–—]?\s*repurchases\b',
    ],
    "part2_item5_other": [
        r'(?i)\bitem\s*5[\.\s]*[-–—]?\s*other\s+information\b',
    ],
    "part2_item6_exhibits": [
        r'(?i)\bitem\s*6[\.\s]*[-–—]?\s*exhibits\b',
    ],
}

# Sections we want to extract from
TARGET_SECTIONS_10Q = [
    "part1_item1_financials",
    "part1_item2_mda",
    "part2_item1a_risk_factors",
]


def find_section_boundaries(text: str) -> list[tuple[str, int]]:
    """Find character positions where each section begins."""
    boundaries = []

    for section_name, patterns in ITEM_PATTERNS_10Q.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if not matches:
                continue

            if len(matches) == 1:
                boundaries.append((section_name, matches[0].start()))
            else:
                # Skip TOC matches in the first 10% of document
                threshold = len(text) * 0.10
                body_matches = [m for m in matches if m.start() > threshold]
                if body_matches:
                    boundaries.append((section_name, body_matches[0].start()))
                else:
                    boundaries.append((section_name, matches[-1].start()))
            break

    boundaries.sort(key=lambda x: x[1])
    return boundaries


def split_into_sections(text: str) -> dict[str, str]:
    """Split full 10-Q text into target sections."""
    boundaries = find_section_boundaries(text)

    if not boundaries:
        print("  WARNING: Could not identify any section boundaries.")
        print("  Falling back to full-text extraction (single API call).")
        return {"full_text": text}

    sections = {}
    for i, (name, start) in enumerate(boundaries):
        if name not in TARGET_SECTIONS_10Q:
            continue

        if i + 1 < len(boundaries):
            end = boundaries[i + 1][1]
        else:
            end = len(text)

        section_text = text[start:end].strip()
        sections[name] = section_text

    found = list(sections.keys())
    missing = [s for s in TARGET_SECTIONS_10Q if s not in sections]
    print(f"  Sections found: {', '.join(found)}")
    if missing:
        # Risk factors in 10-Q are optional — only present if updated since last 10-K
        optional = ["part2_item1a_risk_factors"]
        truly_missing = [s for s in missing if s not in optional]
        optional_missing = [s for s in missing if s in optional]
        if truly_missing:
            print(f"  Sections missing: {', '.join(truly_missing)}")
        if optional_missing:
            print(f"  Optional sections not found (normal): {', '.join(optional_missing)}")

    return sections


# ---------------------------------------------------------------------------
# Extraction Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial document extraction engine. Your task is to read a section of a 10-Q filing and produce a structured JSON extraction.

CRITICAL RULES:
1. Extract only. Do not analyze, interpret, or editorialize. Do not assess whether something is good or bad.
2. Preserve the company's own language for key disclosures.
3. If information is not present in the provided text, use "NOT_PROVIDED". Do not fabricate, infer, or substitute.
4. All numerical values must appear verbatim in the filing. Do not compute, estimate, or round.
5. Cite the Item number and section header for every extraction.
6. Output valid JSON only. No markdown fences, no preamble, no commentary before or after the JSON."""


ITEM2_MDA_PROMPT = """Extract the following from this Item 2 — MD&A section of a 10-Q filing.

The 10-Q MD&A is a quarterly update. It provides more recent context than the annual 10-K. Focus on what is NEW or CHANGED relative to what would be in an annual filing — emerging trends, updated guidance, quarterly performance shifts.

Output this exact JSON structure. Populate all fields from the text. Use "NOT_PROVIDED" for anything not found.

{
  "mda": {
    "source": "10-Q, Item 2 — MD&A",
    "quarter_period": "e.g., Q3 2024 or Three months ended September 30, 2024 — as stated in the filing",
    "management_narrative_summary": "3-5 sentence summary of management's narrative about the quarter's performance",
    "key_performance_drivers": [
      {
        "driver": "description of the performance driver",
        "direction": "positive or negative",
        "magnitude_if_stated": "verbatim number/percentage, or NOT_PROVIDED"
      }
    ],
    "emerging_trends": [
      "verbatim or near-verbatim language about trends, developments, or conditions that are new or evolving compared to prior periods — this is the most valuable content from a 10-Q"
    ],
    "updated_guidance_language": [
      "any forward-looking statements, outlook commentary, or guidance updates — verbatim"
    ],
    "known_trends_and_uncertainties": [
      "verbatim language about known trends, events, or uncertainties management expects to affect future results"
    ],
    "segment_discussion": [
      {
        "segment_name": "",
        "revenue": "verbatim figure or NOT_PROVIDED",
        "operating_income": "verbatim figure or NOT_PROVIDED",
        "management_commentary": "key points from management's discussion of this segment for the quarter"
      }
    ],
    "margin_commentary": {
      "gross_margin_discussion": "what management says about gross margin trends this quarter",
      "operating_margin_discussion": "what management says about operating margin trends this quarter",
      "factors_cited_for_margin_changes": [
        "each factor cited for quarterly margin expansion or compression"
      ]
    },
    "pricing_commentary": {
      "pricing_actions_discussed": true or false,
      "details": "any pricing changes, pricing environment, or competitive pricing pressure discussed"
    },
    "management_emphasized_kpis": [
      "metrics or KPIs management highlights in the quarterly MD&A"
    ],
    "liquidity_and_capital_resources": {
      "cash_position_discussed": "management's characterization of quarterly liquidity",
      "capital_allocation_updates": [
        "any new capital allocation actions or changes in priorities"
      ],
      "share_repurchase_discussion": "quarterly buyback activity discussion"
    },
    "share_repurchase_data": {
      "shares_repurchased_quarter": "verbatim figure or NOT_PROVIDED",
      "amount_spent_on_repurchases": "verbatim dollar amount or NOT_PROVIDED",
      "remaining_authorization": "verbatim figure or NOT_PROVIDED",
      "source_section": ""
    },
    "capital_expenditure_discussion": {
      "total_capex_discussed": "verbatim figure or NOT_PROVIDED",
      "management_capex_outlook": "any updated capex guidance or commentary"
    },
    "critical_accounting_estimates_updates": [
      "any changes to critical accounting estimates from the most recent 10-K — often a brief section. If 'no material changes', note that."
    ],
    "material_interim_events": [
      {
        "event": "description of any material event discussed in the quarterly MD&A — acquisitions, divestitures, restructuring, debt transactions, significant contracts",
        "details": "verbatim key language",
        "source_subsection": ""
      }
    ]
  }
}

IMPORTANT:
- The highest-value content from a 10-Q MD&A is what's NEW or CHANGED. Capture emerging_trends and updated_guidance_language thoroughly.
- Many 10-Q MD&As follow the same structure as the 10-K MD&A but with quarterly figures. Don't just extract a repeat of annual themes — focus on what's different this quarter.
- Share repurchase data is often disclosed in the Liquidity and Capital Resources subsection with specific quarterly amounts.
- If the filing covers both quarterly (three-month) and year-to-date (six/nine-month) periods, prioritize the most recent quarter's discussion for performance drivers and margin commentary.

FILING TEXT:
"""


ITEM1_FINANCIALS_PROMPT = """Extract the following from the Item 1 — Financial Statements section of a 10-Q filing.

Focus on the NOTES to the condensed financial statements, not the face statements. The 10-Q notes provide interim updates to the annual 10-K notes. Extract only items that provide NEW information beyond what the last 10-K would contain.

Output this exact JSON structure. Populate all fields from the text. Use "NOT_PROVIDED" for anything not found.

{
  "quarterly_financial_notes": {
    "source": "10-Q, Item 1 — Financial Statements and Notes",
    "quarter_period": "e.g., Q3 2024 — as stated in the filing",

    "debt_updates": {
      "any_changes_from_annual": true or false,
      "new_issuances": [
        {
          "description": "instrument description",
          "principal_amount": "verbatim",
          "interest_rate": "verbatim",
          "maturity_date": "verbatim",
          "date_issued": ""
        }
      ],
      "repayments_or_redemptions": [
        {
          "description": "what was repaid",
          "amount": "verbatim",
          "date": ""
        }
      ],
      "credit_facility_updates": {
        "drawn_amount": "verbatim or NOT_PROVIDED",
        "available": "verbatim or NOT_PROVIDED",
        "covenant_compliance_noted": "any covenant compliance language"
      },
      "note_reference": ""
    },

    "acquisitions_in_period": [
      {
        "target": "name of acquired entity",
        "date": "closing date",
        "purchase_price": "verbatim total consideration",
        "goodwill_recognized": "verbatim or NOT_PROVIDED",
        "description": "brief description",
        "preliminary_purchase_price_allocation": true or false
      }
    ],

    "divestitures_in_period": [
      {
        "description": "",
        "date": "",
        "proceeds": "verbatim or NOT_PROVIDED",
        "gain_loss": "verbatim or NOT_PROVIDED"
      }
    ],

    "restructuring_updates": {
      "any_activity": true or false,
      "charges_in_period": "verbatim amount or NOT_PROVIDED",
      "description": "nature of restructuring activity",
      "note_reference": ""
    },

    "goodwill_and_intangibles_updates": {
      "any_changes_from_annual": true or false,
      "goodwill_balance": "verbatim or NOT_PROVIDED",
      "impairment_charges_in_period": "verbatim amount or NOT_PROVIDED",
      "changes_description": "description of what changed — e.g., acquisition-related goodwill additions",
      "note_reference": ""
    },

    "contingencies_and_litigation_updates": {
      "any_new_matters": true or false,
      "any_status_changes": true or false,
      "matters": [
        {
          "description": "brief description",
          "status": "new / updated / resolved / unchanged",
          "potential_exposure_if_stated": "verbatim or NOT_PROVIDED",
          "accrual_recorded": "verbatim or NOT_PROVIDED"
        }
      ],
      "note_reference": ""
    },

    "revenue_recognition_updates": {
      "any_changes_from_annual": true or false,
      "changes_description": "any changes to revenue recognition policies, or 'No material changes from 10-K'",
      "remaining_performance_obligations": "verbatim figure or NOT_PROVIDED — ASC 606 disclosure often updated quarterly",
      "note_reference": ""
    },

    "segment_updates": {
      "any_segment_changes": true or false,
      "changes_description": "any segment realignment, additions, or removals",
      "quarterly_segment_data": [
        {
          "name": "",
          "revenue": "verbatim",
          "operating_income": "verbatim or NOT_PROVIDED"
        }
      ],
      "note_reference": ""
    },

    "subsequent_events": "any post-quarter-end events disclosed, verbatim, or NOT_PROVIDED",

    "other_notable_items": [
      {
        "topic": "any other notable disclosure in the quarterly notes not covered above",
        "details": "verbatim key language",
        "note_reference": ""
      }
    ]
  }
}

IMPORTANT:
- The 10-Q notes are CONDENSED. Many notes simply state "no material changes from the 10-K." When you see this, set the any_changes_from_annual flag to false and move on. Don't extract unchanged data — that's already in the 10-K extraction.
- Focus on what's INTERIM — new debt, acquisitions closed since the 10-K, litigation updates, restructuring charges, goodwill changes, updated RPO figures.
- Remaining performance obligations (RPO) under ASC 606 is often updated quarterly and is valuable for the revenue quality assessment.
- If preliminary purchase price allocation is noted for an acquisition, flag it — the numbers may change in subsequent filings.

FILING TEXT:
"""


ITEM1A_RISK_PROMPT = """Extract the following from the Item 1A — Risk Factors section of a 10-Q filing.

IMPORTANT CONTEXT: 10-Q risk factor disclosures are UPDATES to the last 10-K. Many 10-Qs say "no material changes" or only list risks that are new or modified. Your job is to identify what is NEW or CHANGED.

Output this exact JSON structure:

{
  "risk_factor_updates": {
    "source": "10-Q, Item 1A — Risk Factors",
    "quarter_period": "e.g., Q3 2024",
    "update_type": "one of: no_material_changes / new_risks_added / risks_modified / full_restatement",
    "no_material_changes_statement": "verbatim language if the filing states no material changes, or NOT_PROVIDED",
    "new_risks": [
      {
        "heading": "exact heading text",
        "summary": "1-2 sentence summary",
        "verbatim_key_sentence": "the sentence that most directly states the potential impact",
        "category": "regulatory / competitive / operational / financial / macroeconomic / technology / legal / customer_concentration / key_personnel / cybersecurity / supply_chain / geopolitical / esg / other",
        "why_new": "any context on why this risk is being added now — e.g., new regulation, emerging threat"
      }
    ],
    "modified_risks": [
      {
        "heading": "exact heading text",
        "nature_of_modification": "what changed — expanded scope, increased severity language, new specifics added",
        "verbatim_key_change": "the key new or modified language"
      }
    ],
    "removed_risks_if_identifiable": [
      {
        "heading": "heading of risk that was removed (if identifiable from the 10-Q text)",
        "possible_reason": "if the filing provides context for removal"
      }
    ]
  }
}

RULES:
- If the section simply states "no material changes from our 10-K," capture that statement verbatim and set update_type to "no_material_changes". Leave new_risks and modified_risks empty.
- If the section restates the full risk factor list (some companies do this even in 10-Qs), set update_type to "full_restatement" and extract all risks using the same format as new_risks.
- If specific new or modified risks are called out, extract each one with its heading and key language.
- Removed risks are rarely explicitly noted in 10-Qs but if identifiable, capture them.

FILING TEXT:
"""


FULL_TEXT_PROMPT = """Extract the following from this 10-Q filing. The filing could not be automatically split into sections, so you are receiving the full text.

Identify and extract from these sections:
- Item 1 (Financial Statements & Notes) — focus on the notes, not the face statements
- Item 2 (MD&A) — quarterly narrative, emerging trends, updated guidance
- Item 1A Part II (Risk Factors) — if present, new or modified risks

Output valid JSON with this structure:

{
  "metadata": {
    "company_name": "",
    "ticker": "",
    "quarter_period": "",
    "filing_date": "",
    "filing_type": "10-Q",
    "extraction_mode": "full_text"
  },
  "mda": {
    "source": "10-Q, Item 2 — MD&A",
    "quarter_period": "",
    "management_narrative_summary": "",
    "key_performance_drivers": [],
    "emerging_trends": [],
    "updated_guidance_language": [],
    "known_trends_and_uncertainties": [],
    "segment_discussion": [],
    "margin_commentary": {"gross_margin_discussion": "", "operating_margin_discussion": "", "factors_cited_for_margin_changes": []},
    "pricing_commentary": {"pricing_actions_discussed": false, "details": ""},
    "management_emphasized_kpis": [],
    "liquidity_and_capital_resources": {"cash_position_discussed": "", "capital_allocation_updates": [], "share_repurchase_discussion": ""},
    "share_repurchase_data": {"shares_repurchased_quarter": "NOT_PROVIDED", "amount_spent_on_repurchases": "NOT_PROVIDED", "remaining_authorization": "NOT_PROVIDED", "source_section": ""},
    "capital_expenditure_discussion": {"total_capex_discussed": "NOT_PROVIDED", "management_capex_outlook": ""},
    "critical_accounting_estimates_updates": [],
    "material_interim_events": []
  },
  "quarterly_financial_notes": {
    "source": "10-Q, Item 1 — Financial Statements and Notes",
    "quarter_period": "",
    "debt_updates": {"any_changes_from_annual": false, "new_issuances": [], "repayments_or_redemptions": [], "credit_facility_updates": {"drawn_amount": "NOT_PROVIDED", "available": "NOT_PROVIDED", "covenant_compliance_noted": ""}, "note_reference": ""},
    "acquisitions_in_period": [],
    "divestitures_in_period": [],
    "restructuring_updates": {"any_activity": false, "charges_in_period": "NOT_PROVIDED", "description": "", "note_reference": ""},
    "goodwill_and_intangibles_updates": {"any_changes_from_annual": false, "goodwill_balance": "NOT_PROVIDED", "impairment_charges_in_period": "NOT_PROVIDED", "changes_description": "", "note_reference": ""},
    "contingencies_and_litigation_updates": {"any_new_matters": false, "any_status_changes": false, "matters": [], "note_reference": ""},
    "revenue_recognition_updates": {"any_changes_from_annual": false, "changes_description": "", "remaining_performance_obligations": "NOT_PROVIDED", "note_reference": ""},
    "segment_updates": {"any_segment_changes": false, "changes_description": "", "quarterly_segment_data": [], "note_reference": ""},
    "subsequent_events": "NOT_PROVIDED",
    "other_notable_items": []
  },
  "risk_factor_updates": {
    "source": "10-Q, Item 1A — Risk Factors",
    "quarter_period": "",
    "update_type": "no_material_changes",
    "no_material_changes_statement": "NOT_PROVIDED",
    "new_risks": [],
    "modified_risks": [],
    "removed_risks_if_identifiable": []
  }
}

Populate every field you can find. Use "NOT_PROVIDED" for anything not located.

FILING TEXT:
"""


SECTION_PROMPTS = {
    "part1_item1_financials": ITEM1_FINANCIALS_PROMPT,
    "part1_item2_mda": ITEM2_MDA_PROMPT,
    "part2_item1a_risk_factors": ITEM1A_RISK_PROMPT,
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

            # Clean up
            raw_text = re.sub(r'^```json\s*', '', raw_text)
            raw_text = re.sub(r'\s*```$', '', raw_text)

            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group()

            result = json.loads(raw_text)

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            print(f"  ✓ {section_name} complete ({input_tokens:,} in / {output_tokens:,} out)")

            return section_name, result

        except json.JSONDecodeError as e:
            print(f"  ✗ {section_name}: Failed to parse JSON response")
            print(f"    Error: {e}")
            debug_path = OUTPUT_DIR / f"debug_10q_{section_name}.txt"
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
    """Merge per-section extractions into a single 10-Q extraction."""

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
            "filing_type": "10-Q",
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_mode": "per_section",
            "sections_extracted": list(section_results.keys()),
            "sections_not_found": [
                s for s in TARGET_SECTIONS_10Q if s not in section_results
            ],
        }
    }

    section_key_map = {
        "part1_item1_financials": "quarterly_financial_notes",
        "part1_item2_mda": "mda",
        "part2_item1a_risk_factors": "risk_factor_updates",
    }

    for section_name, json_key in section_key_map.items():
        if section_name in section_results:
            result = section_results[section_name]
            # Extract from wrapper key if present
            if json_key in result:
                extraction[json_key] = result[json_key]
            else:
                extraction[json_key] = result
        else:
            if section_name == "part2_item1a_risk_factors":
                # Risk factors are optional in 10-Qs
                extraction[json_key] = {
                    "source": "10-Q, Item 1A — Risk Factors",
                    "quarter_period": "NOT_PROVIDED",
                    "update_type": "section_not_present",
                    "no_material_changes_statement": "Item 1A not included in this 10-Q filing — risk factors unchanged from most recent 10-K",
                    "new_risks": [],
                    "modified_risks": [],
                    "removed_risks_if_identifiable": [],
                }
            else:
                extraction[json_key] = "NOT_PROVIDED"

    return extraction


def validate_extraction(extraction: dict) -> list[str]:
    """Run quality checks on the assembled 10-Q extraction."""
    warnings = []

    meta = extraction.get("metadata", {})
    sections_not_found = meta.get("sections_not_found", [])
    # Filter out optional sections for warning purposes
    required_missing = [s for s in sections_not_found if s != "part2_item1a_risk_factors"]
    if required_missing:
        warnings.append(f"Required sections not found: {', '.join(required_missing)}")

    # --- MD&A checks ---
    mda = extraction.get("mda", {})
    if mda == "NOT_PROVIDED":
        warnings.append("MD&A section entirely missing — this is the primary 10-Q extraction target")
    elif isinstance(mda, dict):
        if not mda.get("management_narrative_summary") or mda["management_narrative_summary"] == "NOT_PROVIDED":
            warnings.append("No management narrative summary extracted from MD&A")
        if not mda.get("key_performance_drivers"):
            warnings.append("No key performance drivers extracted from quarterly MD&A")
        if not mda.get("emerging_trends"):
            warnings.append("No emerging trends extracted — this is the highest-value 10-Q content")
        if not mda.get("management_emphasized_kpis"):
            warnings.append("No management-emphasized KPIs extracted from quarterly MD&A")

        # Check quarter period
        qp = mda.get("quarter_period", "")
        if not qp or qp == "NOT_PROVIDED":
            warnings.append("Quarter period not identified in MD&A extraction")

    # --- Financial Notes checks ---
    notes = extraction.get("quarterly_financial_notes", {})
    if notes == "NOT_PROVIDED":
        warnings.append("Quarterly financial notes section entirely missing")
    elif isinstance(notes, dict):
        # Check if RPO is captured (valuable for revenue quality)
        rev_updates = notes.get("revenue_recognition_updates", {})
        if isinstance(rev_updates, dict):
            rpo = rev_updates.get("remaining_performance_obligations", "NOT_PROVIDED")
            if rpo == "NOT_PROVIDED":
                warnings.append("Remaining performance obligations (RPO) not extracted — check if disclosed in quarterly notes")

    # --- Risk Factor checks ---
    risk = extraction.get("risk_factor_updates", {})
    if isinstance(risk, dict):
        update_type = risk.get("update_type", "")
        if update_type == "full_restatement" and not risk.get("new_risks"):
            warnings.append("Risk factors flagged as full_restatement but no risks extracted")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_single_file(filepath: Path, api_key: str) -> Path | None:
    """Process a single 10-Q filing."""
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

    if len(full_text) < 500:
        print("  ERROR: Extracted text is too short. File may be corrupted or not a 10-Q.")
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
    output_path = OUTPUT_DIR / output_name
    output_path.write_text(
        json.dumps(extraction, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  ✓ Saved to {output_path}")

    # Print summary
    if isinstance(extraction.get("mda"), dict):
        et = len(extraction["mda"].get("emerging_trends", []))
        print(f"  Emerging trends captured: {et}")
        ug = len(extraction["mda"].get("updated_guidance_language", []))
        print(f"  Updated guidance items: {ug}")
    if isinstance(extraction.get("risk_factor_updates"), dict):
        ut = extraction["risk_factor_updates"].get("update_type", "unknown")
        nr = len(extraction["risk_factor_updates"].get("new_risks", []))
        print(f"  Risk factor update type: {ut}")
        if nr > 0:
            print(f"  New risks added: {nr}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from 10-Q filings using Claude API"
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
    print(f"  Output:    {OUTPUT_DIR.resolve()}")
    for name, path in results:
        status = f"✓ {path.name}" if path else "✗ FAILED"
        print(f"    {name} → {status}")


if __name__ == "__main__":
    main()
