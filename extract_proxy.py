"""
DEF 14A Proxy Statement Extraction Pipeline
=============================================
Extracts structured data from SEC DEF 14A proxy filings using Claude API.
Schema validated against equity research prompt v4 requirements.

The v4 prompt uses proxy data exclusively for Section 4.2 (Incentive Alignment)
and supporting governance analysis. Key extraction targets:
  - CD&A: Pay philosophy, performance metrics, peer group
  - Summary Compensation Table: Total comp, equity/cash split per NEO
  - Grants of Plan-Based Awards: RSU/PSU/option grants, vesting, performance conditions
  - Outstanding Equity Awards: Unvested holdings, exercise prices
  - Stock Ownership Guidelines: Required levels, holding periods
  - Clawback/Recoupment Policies: Existence, scope, triggers
  - Director & Officer Ownership: Shares and % owned per insider
  - Board of Directors: Independence, tenure, committee assignments

Proxy statements are long (~60-100 pages) but highly structured. This extractor
splits into two sections — the narrative/governance portion (CD&A, governance)
and the tabular/quantitative portion (comp tables, ownership tables) — and
processes them in parallel.

Usage:
    python extract_proxy.py path/to/proxy.htm
    python extract_proxy.py path/to/proxy.pdf
    python extract_proxy.py path/to/proxies_folder/

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
    import fitz
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
# Text Extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(filepath: Path) -> str:
    if fitz is None:
        print("ERROR: pymupdf not installed. Run: pip install pymupdf")
        sys.exit(1)
    doc = fitz.open(str(filepath))
    pages = [page.get_text() for page in doc]
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

# Proxy statements have two broad halves relevant to us:
# 1. CD&A + governance narrative (qualitative)
# 2. Compensation tables + ownership tables (quantitative)
#
# The boundary is typically at the Summary Compensation Table.
# We also want to capture the Board/Governance section which can appear
# before or after the comp sections.

PROXY_SECTION_PATTERNS = {
    "cda_start": [
        r'(?i)compensation\s+discussion\s+and\s+analysis',
        r'(?i)CD\s*&\s*A',
    ],
    "summary_comp_table": [
        r'(?i)summary\s+compensation\s+table',
    ],
    "grants_plan_based": [
        r'(?i)grants\s+of\s+plan[\s-]*based\s+awards',
    ],
    "outstanding_equity": [
        r'(?i)outstanding\s+equity\s+awards',
    ],
    "stock_ownership_guidelines": [
        r'(?i)stock\s+ownership\s+(?:guidelines|requirements|policy)',
        r'(?i)equity\s+ownership\s+(?:guidelines|requirements|policy)',
    ],
    "clawback": [
        r'(?i)clawback',
        r'(?i)recoupment\s+polic',
    ],
    "director_officer_ownership": [
        r'(?i)(?:security|stock|share)\s+ownership\s+of\s+(?:certain|principal)',
        r'(?i)beneficial\s+ownership',
        r'(?i)(?:director|officer)\s+(?:and\s+(?:officer|director)\s+)?(?:stock\s+)?ownership',
    ],
    "board_directors": [
        r'(?i)(?:election|proposal.*election)\s+of\s+directors',
        r'(?i)board\s+of\s+directors',
        r'(?i)director\s+nominees',
        r'(?i)information\s+(?:about|regarding|concerning)\s+(?:the\s+)?(?:board|directors)',
    ],
    "audit_committee": [
        r'(?i)audit\s+committee\s+report',
    ],
}

# We split into two chunks for API calls:
# 1. "narrative" = CD&A through just before Summary Comp Table + Board/Governance
# 2. "tables" = Summary Comp Table through ownership tables
# If we can't split, send full text


def find_proxy_boundaries(text: str) -> dict[str, int | None]:
    """Find key section boundaries in the proxy statement."""
    boundaries = {}

    for section_name, patterns in PROXY_SECTION_PATTERNS.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if not matches:
                continue

            # Skip TOC matches (first 8% of document)
            threshold = len(text) * 0.08
            body_matches = [m for m in matches if m.start() > threshold]
            if body_matches:
                boundaries[section_name] = body_matches[0].start()
            elif matches:
                boundaries[section_name] = matches[-1].start()
            break

    return boundaries


def split_proxy(text: str) -> dict[str, str]:
    """Split proxy into narrative and tables sections."""
    boundaries = find_proxy_boundaries(text)

    if not boundaries:
        print("  WARNING: Could not identify proxy section boundaries.")
        print("  Processing entire proxy as a single section.")
        return {"full_proxy": text}

    # Report what we found
    found = sorted(boundaries.keys())
    print(f"  Sections identified: {', '.join(found)}")

    # Strategy: split at Summary Compensation Table
    sct_start = boundaries.get("summary_comp_table")
    cda_start = boundaries.get("cda_start")

    if cda_start is not None and sct_start is not None and sct_start > cda_start:
        # Clean split: CD&A narrative vs comp tables
        narrative = text[cda_start:sct_start].strip()
        tables = text[sct_start:].strip()

        # Also grab board/governance section if it appears before CD&A
        board_start = boundaries.get("board_directors")
        if board_start is not None and board_start < cda_start:
            # Board section is before CD&A — prepend it to narrative
            narrative = text[board_start:sct_start].strip()

        sections = {
            "narrative": narrative,
            "tables": tables,
        }

        print(f"  Narrative section: {len(narrative):,} chars (CD&A + governance)")
        print(f"  Tables section: {len(tables):,} chars (comp tables + ownership)")
        return sections

    elif cda_start is not None:
        # Found CD&A but no clear table boundary — send everything from CD&A onward
        print("  Could not identify Summary Comp Table boundary — sending CD&A onward as single section")
        return {"full_proxy": text[cda_start:]}

    else:
        print("  Could not identify CD&A section — processing full text")
        return {"full_proxy": text}


# ---------------------------------------------------------------------------
# Extraction Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial document extraction engine. Your task is to read a DEF 14A proxy statement section and produce a structured JSON extraction.

CRITICAL RULES:
1. Extract only. Do not analyze, interpret, or assess whether compensation is aligned or governance is good. That is the downstream analyst's job.
2. Preserve exact figures from compensation tables — do not round or estimate.
3. If information is not present in the provided text, use "NOT_PROVIDED". Do not fabricate or infer.
4. All dollar amounts, share counts, and percentages must be verbatim from the filing.
5. Output valid JSON only. No markdown fences, no preamble, no commentary."""


NARRATIVE_PROMPT = """Extract the following from the CD&A (Compensation Discussion & Analysis) and governance narrative sections of a DEF 14A proxy statement.

Output this exact JSON structure. Populate all fields from the text. Use "NOT_PROVIDED" for anything not found.

{
  "cda_and_governance": {
    "source": "DEF 14A",

    "compensation_philosophy": {
      "stated_philosophy": "verbatim or near-verbatim description of the company's stated compensation philosophy",
      "key_principles": [
        "each principle the company articulates — e.g., 'pay for performance', 'attract and retain', 'align with shareholder interests'"
      ],
      "target_positioning": "where the company says it targets comp relative to peers — e.g., 'median of peer group', '50th-75th percentile'"
    },

    "peer_group": {
      "peer_companies_listed": [
        "each company named in the peer/benchmarking group"
      ],
      "peer_selection_criteria": "how the company says it selected the peer group",
      "changes_from_prior_year": "any companies added or removed from the peer group, or NOT_PROVIDED"
    },

    "compensation_structure_narrative": {
      "base_salary_philosophy": "how the company describes its approach to base salary",
      "annual_incentive_plan": {
        "description": "how the short-term/annual bonus works",
        "performance_metrics": [
          {
            "metric": "name of the metric (e.g., Revenue, Adjusted EBITDA, EPS, Operating Income)",
            "weight": "verbatim percentage weight or NOT_PROVIDED",
            "threshold_target_maximum": "payout levels if disclosed — e.g., 'threshold 50%, target 100%, maximum 200%'"
          }
        ],
        "actual_payout_percentage": "verbatim — what % of target was actually paid out this year, or NOT_PROVIDED"
      },
      "long_term_incentive_plan": {
        "description": "how the LTI program works overall",
        "equity_vehicles": [
          {
            "type": "stock_options / RSUs / PSUs / other",
            "proportion_of_lti": "verbatim percentage of total LTI, or NOT_PROVIDED",
            "vesting_schedule": "verbatim vesting terms — e.g., '3-year ratable', '3-year cliff', 'performance-based over 3 years'",
            "performance_conditions_if_psu": "what performance metrics PSUs are tied to and over what period",
            "option_term_if_applicable": "e.g., '10-year term' for options, or NOT_PROVIDED"
          }
        ]
      },
      "pay_mix_discussion": "any narrative about the targeted mix between fixed and variable, cash and equity"
    },

    "performance_metrics_summary": {
      "all_metrics_tied_to_comp": [
        "comprehensive list of every metric used in any compensation program — annual and long-term"
      ],
      "metrics_with_specific_targets": [
        {
          "metric": "",
          "target": "verbatim target value",
          "actual": "verbatim actual result if disclosed",
          "payout_result": "above/below/at target, or specific % if disclosed"
        }
      ]
    },

    "say_on_pay": {
      "most_recent_vote_result": "verbatim approval percentage or NOT_PROVIDED",
      "company_response_to_vote": "any discussion of how the company responded to the vote result"
    },

    "governance_safeguards": {
      "clawback_policy": {
        "exists": true or false,
        "scope": "who is covered — NEOs only, all executives, broader",
        "triggers": "what triggers clawback — financial restatement, misconduct, other",
        "verbatim_description": "key language describing the policy"
      },
      "bonus_caps": {
        "exist": true or false,
        "details": "maximum payout levels if disclosed"
      },
      "option_repricing": {
        "has_occurred": true or false,
        "details": "any discussion of option repricing, or explicit policy prohibiting it"
      },
      "hedging_and_pledging_policy": {
        "hedging_prohibited": true or false,
        "pledging_prohibited": true or false,
        "details": ""
      },
      "stock_ownership_guidelines": {
        "exist": true or false,
        "requirements": [
          {
            "role": "e.g., CEO, other NEOs, Directors",
            "required_level": "verbatim — e.g., '6x base salary', '5x annual retainer'",
            "holding_period": "how long they have to meet the requirement, or NOT_PROVIDED",
            "compliance_status_if_disclosed": ""
          }
        ]
      }
    },

    "board_of_directors": {
      "total_board_size": null,
      "number_independent": null,
      "directors": [
        {
          "name": "",
          "age": null,
          "independent": true or false,
          "tenure_years_or_since": "year joined board or years of service",
          "committees": ["list committee memberships"],
          "is_chair_or_lead_independent": true or false,
          "other_public_boards": ["other boards served on, if disclosed"],
          "primary_background": "brief — current or most recent professional role"
        }
      ],
      "board_leadership_structure": "combined chair/CEO or separate, lead independent director if applicable",
      "notable_governance_features": [
        "any other governance features highlighted — e.g., majority voting, proxy access, annual elections, board refreshment"
      ]
    }
  }
}

IMPORTANT:
- For compensation_structure_narrative, the annual incentive metrics and their weights are the most critical fields. The synthesis stage needs to know exactly what metrics comp is tied to in order to assess alignment.
- For equity vehicles, distinguish clearly between time-vesting RSUs and performance-based PSUs. PSUs with performance conditions are more aligned — capture the specific conditions.
- For governance_safeguards, clawback and stock ownership guidelines are the most important items. Capture the specific triggers and requirements.
- For board_of_directors, independence status and tenure are the key fields. Committee assignments help identify who controls audit, compensation, and nominating.
- Peer group composition is important for the time-series assessment — if peers change year over year, the synthesis stage needs to know.

FILING TEXT:
"""


TABLES_PROMPT = """Extract the following from the compensation tables and ownership tables sections of a DEF 14A proxy statement.

These are the formal SEC-mandated tables. Extract exact figures — do not round, estimate, or compute.

Output this exact JSON structure. Populate all fields from the text. Use "NOT_PROVIDED" for anything not found.

{
  "compensation_and_ownership_tables": {
    "source": "DEF 14A — Compensation Tables",

    "summary_compensation_table": {
      "fiscal_year_covered": "",
      "named_executive_officers": [
        {
          "name": "",
          "title": "",
          "year": "",
          "salary": "verbatim",
          "bonus": "verbatim or NOT_PROVIDED",
          "stock_awards": "verbatim",
          "option_awards": "verbatim or NOT_PROVIDED",
          "non_equity_incentive": "verbatim or NOT_PROVIDED",
          "change_in_pension_value": "verbatim or NOT_PROVIDED",
          "all_other_compensation": "verbatim or NOT_PROVIDED",
          "total": "verbatim"
        }
      ],
      "notes": "any footnotes that clarify the table figures"
    },

    "equity_cash_split_per_neo": [
      {
        "name": "",
        "title": "",
        "total_compensation": "verbatim from SCT",
        "cash_components": "salary + bonus + non-equity incentive — verbatim sum if stated, or list the components",
        "equity_components": "stock awards + option awards — verbatim sum if stated, or list the components",
        "equity_percentage_of_total": "verbatim if disclosed, or REQUIRES_COMPUTATION with the components"
      }
    ],

    "grants_of_plan_based_awards": {
      "fiscal_year": "",
      "grants": [
        {
          "name": "",
          "grant_date": "",
          "award_type": "PSU / RSU / stock_option / annual_incentive / other",
          "threshold_payout": "verbatim or NOT_PROVIDED",
          "target_payout": "verbatim or NOT_PROVIDED",
          "maximum_payout": "verbatim or NOT_PROVIDED",
          "shares_granted": "verbatim or NOT_PROVIDED",
          "exercise_price": "verbatim or NOT_PROVIDED",
          "grant_date_fair_value": "verbatim or NOT_PROVIDED"
        }
      ]
    },

    "outstanding_equity_awards": {
      "as_of_date": "",
      "awards": [
        {
          "name": "",
          "award_type": "option / RSU / PSU",
          "shares_unvested": "verbatim or NOT_PROVIDED",
          "shares_unexercised_exercisable": "verbatim or NOT_PROVIDED",
          "shares_unexercised_unexercisable": "verbatim or NOT_PROVIDED",
          "exercise_price": "verbatim or NOT_PROVIDED",
          "expiration_date": "verbatim or NOT_PROVIDED",
          "market_value_of_unvested": "verbatim or NOT_PROVIDED"
        }
      ]
    },

    "option_exercises_and_stock_vested": {
      "data": [
        {
          "name": "",
          "options_exercised_shares": "verbatim or NOT_PROVIDED",
          "options_exercised_value_realized": "verbatim or NOT_PROVIDED",
          "stock_vested_shares": "verbatim or NOT_PROVIDED",
          "stock_vested_value_realized": "verbatim or NOT_PROVIDED"
        }
      ]
    },

    "director_compensation_table": {
      "fiscal_year": "",
      "directors": [
        {
          "name": "",
          "fees_earned_or_paid_in_cash": "verbatim or NOT_PROVIDED",
          "stock_awards": "verbatim or NOT_PROVIDED",
          "option_awards": "verbatim or NOT_PROVIDED",
          "all_other_compensation": "verbatim or NOT_PROVIDED",
          "total": "verbatim"
        }
      ]
    },

    "beneficial_ownership_table": {
      "as_of_date": "",
      "shares_outstanding": "verbatim total shares outstanding used as denominator, or NOT_PROVIDED",
      "insiders": [
        {
          "name": "",
          "title_or_relationship": "",
          "shares_beneficially_owned": "verbatim",
          "percent_of_class": "verbatim",
          "includes_options_exercisable_within_60_days": "verbatim or NOT_PROVIDED"
        }
      ],
      "total_insider_ownership": {
        "shares": "verbatim or NOT_PROVIDED",
        "percent": "verbatim or NOT_PROVIDED"
      },
      "significant_external_holders": [
        {
          "name": "institutional holder name",
          "shares": "verbatim",
          "percent": "verbatim"
        }
      ]
    },

    "ceo_pay_ratio": {
      "disclosed": true or false,
      "ratio": "verbatim — e.g., '195:1'",
      "ceo_total": "verbatim",
      "median_employee_total": "verbatim",
      "methodology_notes": "brief description of how the ratio was calculated"
    },

    "pay_versus_performance": {
      "table_provided": true or false,
      "data": [
        {
          "year": "",
          "sct_total_ceo": "verbatim",
          "compensation_actually_paid_ceo": "verbatim",
          "average_sct_total_other_neos": "verbatim or NOT_PROVIDED",
          "average_cap_other_neos": "verbatim or NOT_PROVIDED",
          "tsr": "verbatim or NOT_PROVIDED",
          "peer_tsr": "verbatim or NOT_PROVIDED",
          "net_income": "verbatim or NOT_PROVIDED",
          "company_selected_measure": "verbatim or NOT_PROVIDED"
        }
      ],
      "company_selected_performance_measure": "which measure the company chose and why"
    }
  }
}

IMPORTANT:
- The Summary Compensation Table is the single most important table. Extract EVERY NEO, EVERY column, EVERY year available in the table. Companies typically show 3 years.
- For equity_cash_split_per_neo, if the proxy doesn't explicitly state the equity percentage, capture the components and mark as REQUIRES_COMPUTATION. The synthesis stage will compute it.
- The beneficial ownership table is critical for insider ownership assessment. Capture both individual insiders AND significant external holders (typically >5% holders).
- For outstanding equity awards, the unvested shares and expiration dates tell the synthesis stage how much skin in the game executives have.
- The Pay versus Performance table (SEC requirement since 2022) provides "compensation actually paid" vs TSR — capture the full multi-year table if present.
- CEO pay ratio is a single data point but useful for context.
- Footnotes on the Summary Compensation Table often contain important clarifications (e.g., special one-time grants, sign-on bonuses). Capture them.

FILING TEXT:
"""


FULL_PROXY_PROMPT = """Extract the following from this DEF 14A proxy statement. The filing could not be automatically split into sections, so you are receiving the full text.

Identify and extract from these sections:
- CD&A (Compensation Discussion & Analysis)
- Summary Compensation Table
- Grants of Plan-Based Awards
- Outstanding Equity Awards
- Stock Ownership Guidelines
- Clawback/Recoupment Policies
- Beneficial Ownership Table
- Board of Directors

Output valid JSON with this structure:

{
  "metadata": {
    "company_name": "",
    "fiscal_year": "",
    "filing_date": "",
    "filing_type": "DEF 14A",
    "extraction_mode": "full_proxy"
  },
  "cda_and_governance": {
    "source": "DEF 14A",
    "compensation_philosophy": {"stated_philosophy": "", "key_principles": [], "target_positioning": "NOT_PROVIDED"},
    "peer_group": {"peer_companies_listed": [], "peer_selection_criteria": "", "changes_from_prior_year": "NOT_PROVIDED"},
    "compensation_structure_narrative": {
      "base_salary_philosophy": "",
      "annual_incentive_plan": {"description": "", "performance_metrics": [], "actual_payout_percentage": "NOT_PROVIDED"},
      "long_term_incentive_plan": {"description": "", "equity_vehicles": []},
      "pay_mix_discussion": ""
    },
    "performance_metrics_summary": {"all_metrics_tied_to_comp": [], "metrics_with_specific_targets": []},
    "say_on_pay": {"most_recent_vote_result": "NOT_PROVIDED", "company_response_to_vote": ""},
    "governance_safeguards": {
      "clawback_policy": {"exists": false, "scope": "NOT_PROVIDED", "triggers": "NOT_PROVIDED", "verbatim_description": "NOT_PROVIDED"},
      "bonus_caps": {"exist": false, "details": "NOT_PROVIDED"},
      "option_repricing": {"has_occurred": false, "details": "NOT_PROVIDED"},
      "hedging_and_pledging_policy": {"hedging_prohibited": false, "pledging_prohibited": false, "details": "NOT_PROVIDED"},
      "stock_ownership_guidelines": {"exist": false, "requirements": []}
    },
    "board_of_directors": {"total_board_size": null, "number_independent": null, "directors": [], "board_leadership_structure": "", "notable_governance_features": []}
  },
  "compensation_and_ownership_tables": {
    "source": "DEF 14A — Compensation Tables",
    "summary_compensation_table": {"fiscal_year_covered": "", "named_executive_officers": [], "notes": ""},
    "equity_cash_split_per_neo": [],
    "grants_of_plan_based_awards": {"fiscal_year": "", "grants": []},
    "outstanding_equity_awards": {"as_of_date": "", "awards": []},
    "option_exercises_and_stock_vested": {"data": []},
    "director_compensation_table": {"fiscal_year": "", "directors": []},
    "beneficial_ownership_table": {"as_of_date": "", "shares_outstanding": "NOT_PROVIDED", "insiders": [], "total_insider_ownership": {"shares": "NOT_PROVIDED", "percent": "NOT_PROVIDED"}, "significant_external_holders": []},
    "ceo_pay_ratio": {"disclosed": false, "ratio": "NOT_PROVIDED", "ceo_total": "NOT_PROVIDED", "median_employee_total": "NOT_PROVIDED", "methodology_notes": ""},
    "pay_versus_performance": {"table_provided": false, "data": [], "company_selected_performance_measure": ""}
  }
}

Populate every field you can find. Use "NOT_PROVIDED" for anything not located.

FILING TEXT:
"""


SECTION_PROMPTS = {
    "narrative": NARRATIVE_PROMPT,
    "tables": TABLES_PROMPT,
    "full_proxy": FULL_PROXY_PROMPT,
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
            debug_path = RUN_OUTPUT_DIR / f"debug_proxy_{section_name}.txt"
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
    """Merge section extractions into a single proxy extraction."""

    if "full_proxy" in section_results:
        extraction = section_results["full_proxy"]
        extraction.setdefault("metadata", {})
        extraction["metadata"]["source_file"] = filepath.name
        extraction["metadata"]["extraction_timestamp"] = datetime.now().isoformat()
        extraction["metadata"]["extraction_mode"] = "full_proxy"
        return extraction

    extraction = {
        "metadata": {
            "source_file": filepath.name,
            "filing_type": "DEF 14A",
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_mode": "split_sections",
            "sections_extracted": list(section_results.keys()),
        }
    }

    # Merge narrative (CD&A + governance)
    if "narrative" in section_results:
        result = section_results["narrative"]
        if "cda_and_governance" in result:
            extraction["cda_and_governance"] = result["cda_and_governance"]
        else:
            extraction["cda_and_governance"] = result
    else:
        extraction["cda_and_governance"] = "NOT_PROVIDED"

    # Merge tables
    if "tables" in section_results:
        result = section_results["tables"]
        if "compensation_and_ownership_tables" in result:
            extraction["compensation_and_ownership_tables"] = result["compensation_and_ownership_tables"]
        else:
            extraction["compensation_and_ownership_tables"] = result
    else:
        extraction["compensation_and_ownership_tables"] = "NOT_PROVIDED"

    return extraction


def validate_extraction(extraction: dict) -> list[str]:
    """Run quality checks on the assembled proxy extraction."""
    warnings = []

    # --- CD&A / Governance checks ---
    cda = extraction.get("cda_and_governance", {})
    if cda == "NOT_PROVIDED":
        warnings.append("CD&A / governance section entirely missing")
    elif isinstance(cda, dict):
        # Compensation metrics — most critical field
        metrics = cda.get("performance_metrics_summary", {})
        if isinstance(metrics, dict):
            all_metrics = metrics.get("all_metrics_tied_to_comp", [])
            if not all_metrics:
                warnings.append("No compensation performance metrics extracted — this is the most important proxy data point")

        # Annual incentive plan metrics
        comp_narrative = cda.get("compensation_structure_narrative", {})
        if isinstance(comp_narrative, dict):
            aip = comp_narrative.get("annual_incentive_plan", {})
            if isinstance(aip, dict):
                if not aip.get("performance_metrics"):
                    warnings.append("No annual incentive plan metrics extracted")

            lti = comp_narrative.get("long_term_incentive_plan", {})
            if isinstance(lti, dict):
                vehicles = lti.get("equity_vehicles", [])
                if not vehicles:
                    warnings.append("No equity vehicles extracted from LTI plan")
                else:
                    has_psu = any(v.get("type") == "PSU" or "PSU" in str(v.get("type", "")) for v in vehicles)
                    has_conditions = any(v.get("performance_conditions_if_psu") and v["performance_conditions_if_psu"] != "NOT_PROVIDED" for v in vehicles)
                    if has_psu and not has_conditions:
                        warnings.append("PSUs identified but no performance conditions extracted — check filing")

        # Governance safeguards
        gov = cda.get("governance_safeguards", {})
        if isinstance(gov, dict):
            clawback = gov.get("clawback_policy", {})
            if isinstance(clawback, dict) and not clawback.get("exists"):
                warnings.append("No clawback policy found — verify against filing (required by SEC since 2023)")

            ownership = gov.get("stock_ownership_guidelines", {})
            if isinstance(ownership, dict) and not ownership.get("exist"):
                warnings.append("No stock ownership guidelines found — verify against filing")

        # Board
        board = cda.get("board_of_directors", {})
        if isinstance(board, dict):
            directors = board.get("directors", [])
            if not directors:
                warnings.append("No directors extracted")
            else:
                unnamed = [d for d in directors if not d.get("name")]
                if unnamed:
                    warnings.append(f"{len(unnamed)} director(s) without names")
                no_independence = [d for d in directors if d.get("independent") is None]
                if no_independence:
                    warnings.append(f"{len(no_independence)} director(s) without independence classification")

    # --- Tables checks ---
    tables = extraction.get("compensation_and_ownership_tables", {})
    if tables == "NOT_PROVIDED":
        warnings.append("Compensation tables section entirely missing")
    elif isinstance(tables, dict):
        # Summary Compensation Table
        sct = tables.get("summary_compensation_table", {})
        if isinstance(sct, dict):
            neos = sct.get("named_executive_officers", [])
            if not neos:
                warnings.append("No NEOs extracted from Summary Compensation Table")
            elif len(neos) < 3:
                warnings.append(f"Only {len(neos)} NEOs extracted — most companies have 5+")
            else:
                # Check for CEO
                has_ceo = any("ceo" in (n.get("title", "") or "").lower() or
                              "chief executive" in (n.get("title", "") or "").lower()
                              for n in neos)
                if not has_ceo:
                    warnings.append("No CEO identified in Summary Compensation Table")

        # Beneficial ownership
        ownership = tables.get("beneficial_ownership_table", {})
        if isinstance(ownership, dict):
            insiders = ownership.get("insiders", [])
            if not insiders:
                warnings.append("No insiders extracted from beneficial ownership table")

        # Equity/cash split
        split = tables.get("equity_cash_split_per_neo", [])
        if not split:
            warnings.append("Equity/cash split per NEO not extracted — key for incentive alignment assessment")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_single_file(filepath: Path, api_key: str) -> Path | None:
    """Process a single DEF 14A proxy statement."""
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
        print("  ERROR: Extracted text is too short. File may be corrupted or not a proxy statement.")
        return None

    # Step 2: Split into sections
    print("\n[Step 2] Identifying section boundaries...")
    sections = split_proxy(full_text)

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

    # Print summary
    if isinstance(extraction.get("cda_and_governance"), dict):
        cda = extraction["cda_and_governance"]
        metrics = cda.get("performance_metrics_summary", {}).get("all_metrics_tied_to_comp", [])
        print(f"  Compensation metrics tied to pay: {len(metrics)}")

        board = cda.get("board_of_directors", {})
        directors = board.get("directors", [])
        independent = sum(1 for d in directors if d.get("independent"))
        print(f"  Board: {len(directors)} directors ({independent} independent)")

        gov = cda.get("governance_safeguards", {})
        clawback = gov.get("clawback_policy", {}).get("exists", False)
        ownership_guidelines = gov.get("stock_ownership_guidelines", {}).get("exist", False)
        print(f"  Clawback policy: {'Yes' if clawback else 'No'}")
        print(f"  Stock ownership guidelines: {'Yes' if ownership_guidelines else 'No'}")

    if isinstance(extraction.get("compensation_and_ownership_tables"), dict):
        tables = extraction["compensation_and_ownership_tables"]
        neos = tables.get("summary_compensation_table", {}).get("named_executive_officers", [])
        print(f"  NEOs in compensation table: {len(neos)}")
        insiders = tables.get("beneficial_ownership_table", {}).get("insiders", [])
        print(f"  Insiders in ownership table: {len(insiders)}")

    return output_path


def main():
    global RUN_OUTPUT_DIR
    RUN_OUTPUT_DIR = create_run_output_dir()

    parser = argparse.ArgumentParser(
        description="Extract structured data from DEF 14A proxy statements using Claude API"
    )
    parser.add_argument(
        "path",
        help="Path to a single proxy filing (PDF/HTML) or a folder of filings",
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
        print(f"Found {len(files)} proxy filing(s) to process")
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
    print(f"  Processed: {len(results)} filing(s)")
    print(f"  Succeeded: {succeeded}")
    if failed > 0:
        print(f"  Failed:    {failed}")
    print(f"  Output:    {RUN_OUTPUT_DIR.resolve()}")
    for name, path in results:
        status = f"✓ {path.name}" if path else "✗ FAILED"
        print(f"    {name} → {status}")


if __name__ == "__main__":
    main()
