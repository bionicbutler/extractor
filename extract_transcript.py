"""
Earnings Transcript Extraction Pipeline
=========================================
Extracts structured data from earnings call transcripts using Claude API.
Schema validated against equity research prompt v4 requirements.

Transcripts are the most attention-intensive document type. The v4 prompt
uses transcript data for:
  - Section 3: Pricing signals, competitive dynamics, revenue quality evidence
  - Section 4.3: Communication clarity — management transparency assessment
  - Section 5.2: Risk inventory — transcript-derived risks
  - Section 6.3: Messaging vs metrics consistency
  - Section 8: Full earnings call insights — theme evolution, follow-through,
    forward-looking language

This extractor splits each transcript into Prepared Remarks and Q&A and
processes them as separate API calls to ensure thorough reading of both.

Usage:
    python extract_transcript.py path/to/transcript.txt
    python extract_transcript.py path/to/transcript.pdf
    python extract_transcript.py path/to/transcripts_folder/

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
MAX_TOKENS = 16000
CONCURRENT_LIMIT = 5

OUTPUT_DIR = Path("./extractions")
OUTPUT_DIR.mkdir(exist_ok=True)


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
# Transcript Splitting
# ---------------------------------------------------------------------------

# Patterns to identify the boundary between Prepared Remarks and Q&A
QA_BOUNDARY_PATTERNS = [
    r'(?i)question[\s-]*and[\s-]*answer\s*(?:session|section|portion)?',
    r'(?i)Q\s*&\s*A\s*(?:session|section|portion)?',
    r'(?i)(?:we|I|the company)\s+(?:will|would)\s+now\s+(?:like\s+to\s+)?(?:open|take)\s+(?:the\s+)?(?:call|line|floor)\s+(?:for|to)\s+questions',
    r'(?i)(?:operator|moderator)[,:]?\s+(?:we are|we\'re)\s+ready\s+for\s+questions',
    r'(?i)let\s*(?:me|us)\s+(?:now\s+)?(?:open|turn)\s+(?:it\s+)?(?:up\s+)?(?:for|to)\s+questions',
    r'(?i)\[operator\s+instructions\]',
]

# Patterns to identify transcript metadata (call type, date)
CALL_TYPE_PATTERNS = [
    r'(?i)(Q[1-4]\s+\d{4})\s+(?:earnings|results)\s+(?:call|conference)',
    r'(?i)((?:first|second|third|fourth)\s+quarter\s+\d{4})\s+(?:earnings|results)',
    r'(?i)(FY\s*\d{4})\s+(?:earnings|results|annual)',
    r'(?i)(?:earnings|results)\s+(?:call|conference)\s*[-–—]?\s*(Q[1-4]\s+\d{4})',
]

CALL_DATE_PATTERNS = [
    r'(?i)(?:date|held\s+on|dated?)\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
    r'(?i)(\w+\s+\d{1,2},?\s+\d{4})',  # Fallback: any date-like pattern
]


def extract_call_metadata(text: str) -> dict:
    """Extract call type, date, and period from transcript header."""
    # Usually in the first ~2000 characters
    header = text[:3000]

    metadata = {
        "call_type": "NOT_PROVIDED",
        "call_date": "NOT_PROVIDED",
        "period": "NOT_PROVIDED",
        "company_name": "NOT_PROVIDED",
    }

    # Call type / period
    for pattern in CALL_TYPE_PATTERNS:
        match = re.search(pattern, header)
        if match:
            raw = match.group(0)
            period = match.group(1)
            metadata["call_type"] = raw.strip()
            metadata["period"] = period.strip()
            break

    # Call date
    for pattern in CALL_DATE_PATTERNS:
        match = re.search(pattern, header)
        if match:
            metadata["call_date"] = match.group(1).strip()
            break

    return metadata


def split_transcript(text: str) -> dict[str, str]:
    """
    Split transcript into header, prepared remarks, and Q&A sections.
    Returns dict with keys: 'full_text' (always), plus 'prepared_remarks'
    and 'qanda' if the boundary is found.
    """
    # Try to find the Q&A boundary
    qa_start = None
    for pattern in QA_BOUNDARY_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Take the match that's roughly in the middle-to-late part of the doc
            # (skip any early mention in the agenda/intro)
            for m in matches:
                if m.start() > len(text) * 0.15:
                    qa_start = m.start()
                    break
            if qa_start:
                break

    if qa_start is None:
        print("  WARNING: Could not identify Q&A boundary.")
        print("  Processing entire transcript as a single section.")
        return {"full_transcript": text}

    prepared_remarks = text[:qa_start].strip()
    qanda = text[qa_start:].strip()

    print(f"  Prepared Remarks: {len(prepared_remarks):,} chars")
    print(f"  Q&A Section: {len(qanda):,} chars")

    sections = {
        "prepared_remarks": prepared_remarks,
        "qanda": qanda,
    }

    return sections


# ---------------------------------------------------------------------------
# Extraction Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial transcript extraction engine. Your task is to read an earnings call transcript section and produce a structured JSON extraction.

CRITICAL RULES:
1. Extract only. Do not analyze, interpret, or assess management quality. That is the downstream analyst's job.
2. Every quote must be EXACT — copied verbatim from the transcript. Do not paraphrase, summarize, or clean up quotes.
3. Every quote must have independent citation metadata. Do not carry forward metadata from earlier quotes.
4. If a speaker's name or role is not explicitly stated, mark it as "NOT_PROVIDED" — do not guess.
5. All numerical values mentioned by speakers must be captured verbatim.
6. Output valid JSON only. No markdown fences, no preamble, no commentary."""


PREPARED_REMARKS_PROMPT = """Extract the following from the PREPARED REMARKS section of an earnings call transcript.

This is the scripted portion where management presents results and outlook. Extract thoroughly — every substantive statement matters for downstream analysis.

Output this exact JSON structure:

{
  "prepared_remarks": {
    "source": "Prepared Remarks",

    "speakers": [
      {
        "full_name": "exact name as it appears in the transcript",
        "role": "title/role as stated (CEO, CFO, VP of Operations, etc.) or NOT_PROVIDED",
        "is_company_executive": true or false,
        "is_operator_or_moderator": true or false
      }
    ],

    "key_themes": [
      {
        "theme": "short label for the theme (e.g., 'margin expansion', 'cloud migration', 'pricing actions', 'M&A integration')",
        "supporting_quotes": [
          {
            "quote": "exact verbatim quote from the transcript — copy precisely, including any verbal tics or filler",
            "speaker": "full name",
            "role": "role or NOT_PROVIDED"
          }
        ],
        "category": "one of: growth_strategy / margin_outlook / capital_allocation / competitive_dynamics / pricing / product_development / regulatory / risk_acknowledgment / guidance / operational_efficiency / M&A / talent / other"
      }
    ],

    "financial_metrics_cited": [
      {
        "metric": "what metric is being discussed (e.g., revenue, EPS, EBITDA, free cash flow)",
        "value": "verbatim number as spoken",
        "context": "what the speaker said about this metric — verbatim",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "guidance_and_outlook": [
      {
        "topic": "what the guidance pertains to (e.g., FY revenue, Q4 margins, capex)",
        "verbatim_language": "exact quote of the guidance statement",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "commitments_and_priorities": [
      {
        "commitment": "what management commits to, prioritizes, or promises — verbatim",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED",
        "specificity": "specific (has measurable targets/timelines) or vague (general aspiration)"
      }
    ],

    "risk_acknowledgments": [
      {
        "risk_topic": "what risk or challenge is being acknowledged",
        "verbatim_language": "exact quote",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "pricing_and_demand_commentary": [
      {
        "topic": "pricing actions / pricing environment / demand trends / volume trends / competitive pricing",
        "verbatim_language": "exact quote",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "competitive_commentary": [
      {
        "topic": "what competitive dynamic is being discussed",
        "verbatim_language": "exact quote",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "management_kpis_emphasized": [
      "list every metric or KPI that management highlights, frames as important, or uses to describe the business performance — include non-GAAP metrics"
    ],

    "tone_indicators": {
      "confidence_language": [
        "verbatim phrases that indicate confidence or optimism (e.g., 'we are very pleased', 'strong momentum', 'well-positioned')"
      ],
      "caution_language": [
        "verbatim phrases that indicate caution, hedging, or uncertainty (e.g., 'we remain cautious', 'headwinds', 'uncertain environment')"
      ],
      "deflection_or_reframing": [
        "verbatim instances where management reframes a negative as positive or deflects from a difficult topic"
      ]
    }
  }
}

CRITICAL INSTRUCTIONS:
- Extract EVERY substantive statement from management. Do not skip sections or summarize.
- Quotes must be EXACT. If a speaker says "we delivered strong double-digit growth of approximately 14%", capture that entire phrase verbatim — do not clean it up to "approximately 14% growth".
- For speakers, capture EVERY unique speaker. The operator/moderator should be listed too.
- For key_themes, a single theme may have multiple supporting quotes from different speakers. Group them together.
- For financial_metrics_cited, capture every specific number management mentions — revenue figures, growth rates, margin percentages, EPS, cash flow amounts, etc. These are audit points.
- For commitments_and_priorities, this is critical for the downstream follow-through analysis. Capture anything that sounds like a promise, target, or stated priority.
- For tone_indicators, you are NOT assessing tone. You are extracting the verbatim language that a downstream analyst would use for tone assessment. Just capture the phrases.

TRANSCRIPT TEXT:
"""


QA_PROMPT = """Extract the following from the Q&A section of an earnings call transcript.

The Q&A is the unscripted portion where analysts question management. This section is critical for assessing management's communication quality — how directly they answer, whether they evade, and what their unscripted responses reveal.

Output this exact JSON structure:

{
  "qanda": {
    "source": "Q&A",

    "exchanges": [
      {
        "analyst_name": "full name or NOT_PROVIDED",
        "analyst_firm": "firm name or NOT_PROVIDED",
        "questions": [
          {
            "question_text": "verbatim question as asked",
            "topic": "short label for what the question is about"
          }
        ],
        "responses": [
          {
            "speaker": "full name of the executive responding",
            "role": "role or NOT_PROVIDED",
            "response_text": "verbatim response — capture the FULL response, do not truncate",
            "directness": "direct (clearly answers the question) / partial (addresses part of it) / evasive (redirects or avoids the question) / deferred (promises to follow up later)"
          }
        ],
        "follow_ups": [
          {
            "question_text": "verbatim follow-up question",
            "response_speaker": "full name",
            "response_text": "verbatim response",
            "directness": "direct / partial / evasive / deferred"
          }
        ]
      }
    ],

    "financial_metrics_cited_in_qa": [
      {
        "metric": "what metric",
        "value": "verbatim number",
        "context": "verbatim surrounding language",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "new_information_revealed": [
      {
        "topic": "information that was NOT in the prepared remarks but emerged in Q&A",
        "verbatim_language": "exact quote",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED",
        "significance": "brief description of why this is notable"
      }
    ],

    "guidance_clarifications": [
      {
        "topic": "what guidance point is being clarified or updated in Q&A",
        "verbatim_language": "exact quote",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "pricing_and_demand_commentary": [
      {
        "topic": "pricing / demand / competitive dynamics discussed in Q&A",
        "verbatim_language": "exact quote",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED"
      }
    ],

    "evasion_and_deflection_instances": [
      {
        "analyst_question_summary": "what was asked",
        "question_text": "verbatim question",
        "response_text": "verbatim response that evades or deflects",
        "speaker": "full name",
        "role": "role or NOT_PROVIDED",
        "evasion_type": "topic_redirect (answered a different question) / vagueness (gave a non-specific answer) / forward_reference (said 'we'll share more later') / blame_external (attributed to external factors without addressing substance)"
      }
    ],

    "tone_indicators": {
      "defensive_responses": [
        "verbatim responses where management sounds defensive when challenged"
      ],
      "candid_acknowledgments": [
        "verbatim responses where management candidly acknowledges a problem or miss"
      ],
      "enthusiasm_or_confidence": [
        "verbatim language showing strong confidence in Q&A responses"
      ]
    },

    "qa_summary_stats": {
      "total_analyst_questioners": null,
      "total_exchanges": null,
      "questions_answered_directly": null,
      "questions_answered_partially": null,
      "questions_evaded": null,
      "questions_deferred": null
    }
  }
}

CRITICAL INSTRUCTIONS:
- Extract EVERY exchange. Do not skip any analyst's questions. Each analyst interaction is a separate exchange.
- Capture FULL responses — do not truncate long answers. The downstream analyst needs the complete text to assess communication quality.
- The directness assessment is the most important field per response. Be rigorous:
    - "direct" = the response clearly and specifically addresses what was asked
    - "partial" = the response addresses part of the question but skips part
    - "evasive" = the response redirects to a different topic or gives a non-answer
    - "deferred" = the response explicitly says they'll provide more detail later
- For evasion_and_deflection_instances, only include clear cases. If you're unsure whether a response is evasive, it's probably "partial" — err on the side of not overcounting evasion.
- For new_information_revealed, capture anything substantive that an analyst question draws out that wasn't in the prepared remarks. This often includes competitive detail, customer-specific information, or operational specifics management chose not to front in prepared remarks.
- The qa_summary_stats should be computed from the exchanges you've extracted.
- Analyst names and firms are usually stated by the operator when introducing each questioner. Capture both when available.

TRANSCRIPT TEXT:
"""


FULL_TRANSCRIPT_PROMPT = """Extract the following from this complete earnings call transcript. The transcript could not be automatically split into Prepared Remarks and Q&A sections.

Identify the boundary between Prepared Remarks and Q&A yourself, then extract thoroughly from both sections.

Output valid JSON with this structure:

{
  "metadata": {
    "call_type": "e.g., Q4 2024 Earnings Call",
    "call_date": "as stated in transcript, or NOT_PROVIDED",
    "period": "e.g., Q4 2024",
    "company_name": "",
    "extraction_mode": "full_transcript",
    "estimated_qa_start": "approximate description of where Q&A begins"
  },
  "prepared_remarks": {
    "source": "Prepared Remarks",
    "speakers": [],
    "key_themes": [],
    "financial_metrics_cited": [],
    "guidance_and_outlook": [],
    "commitments_and_priorities": [],
    "risk_acknowledgments": [],
    "pricing_and_demand_commentary": [],
    "competitive_commentary": [],
    "management_kpis_emphasized": [],
    "tone_indicators": {
      "confidence_language": [],
      "caution_language": [],
      "deflection_or_reframing": []
    }
  },
  "qanda": {
    "source": "Q&A",
    "exchanges": [],
    "financial_metrics_cited_in_qa": [],
    "new_information_revealed": [],
    "guidance_clarifications": [],
    "pricing_and_demand_commentary": [],
    "evasion_and_deflection_instances": [],
    "tone_indicators": {
      "defensive_responses": [],
      "candid_acknowledgments": [],
      "enthusiasm_or_confidence": []
    },
    "qa_summary_stats": {
      "total_analyst_questioners": null,
      "total_exchanges": null,
      "questions_answered_directly": null,
      "questions_answered_partially": null,
      "questions_evaded": null,
      "questions_deferred": null
    }
  }
}

Use the same extraction rules as the individual section prompts. Extract EVERY exchange, EVERY substantive quote, and EVERY metric cited. Quotes must be verbatim.

TRANSCRIPT TEXT:
"""


SECTION_PROMPTS = {
    "prepared_remarks": PREPARED_REMARKS_PROMPT,
    "qanda": QA_PROMPT,
    "full_transcript": FULL_TRANSCRIPT_PROMPT,
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
            debug_path = OUTPUT_DIR / f"debug_transcript_{section_name}.txt"
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
# Citation Tier Classification
# ---------------------------------------------------------------------------
def classify_citation_tier(metadata: dict, speakers: list[dict]) -> dict:
    """
    Determine the citation tier for this transcript based on available metadata.

    Tier 1: All 5 fields confirmed (call_type, call_date, speaker names, roles, section)
    Tier 2: 3-4 fields confirmed
    Tier 3: <3 fields confirmed
    """
    confirmed_fields = []
    missing_fields = []

    # Field 1: Call Type
    if metadata.get("call_type") and metadata["call_type"] != "NOT_PROVIDED":
        confirmed_fields.append("call_type")
    else:
        missing_fields.append("call_type")

    # Field 2: Call Date
    if metadata.get("call_date") and metadata["call_date"] != "NOT_PROVIDED":
        confirmed_fields.append("call_date")
    else:
        missing_fields.append("call_date")

    # Field 3 & 4: Speaker names and roles
    exec_speakers = [s for s in speakers if s.get("is_company_executive")]
    if exec_speakers:
        has_names = any(s.get("full_name") and s["full_name"] != "NOT_PROVIDED" for s in exec_speakers)
        has_roles = any(s.get("role") and s["role"] != "NOT_PROVIDED" for s in exec_speakers)
        if has_names:
            confirmed_fields.append("speaker_names")
        else:
            missing_fields.append("speaker_names")
        if has_roles:
            confirmed_fields.append("speaker_roles")
        else:
            missing_fields.append("speaker_roles")
    else:
        missing_fields.extend(["speaker_names", "speaker_roles"])

    # Field 5: Section (Prepared Remarks / Q&A) — confirmed if we split successfully
    confirmed_fields.append("section_identification")

    count = len(confirmed_fields)
    if count >= 5:
        tier = 1
    elif count >= 3:
        tier = 2
    else:
        tier = 3

    return {
        "tier": tier,
        "tier_label": f"Tier {tier}",
        "confirmed_fields": confirmed_fields,
        "missing_fields": missing_fields,
        "confirmed_count": count,
        "admissible": tier <= 2,
        "note": "Tier 3 transcripts are excluded from evidentiary use" if tier == 3 else "",
    }


# ---------------------------------------------------------------------------
# Assembly & Validation
# ---------------------------------------------------------------------------
def assemble_extraction(section_results: dict, filepath: Path, call_metadata: dict) -> dict:
    """Merge section extractions into a single transcript extraction."""

    if "full_transcript" in section_results:
        extraction = section_results["full_transcript"]
        extraction.setdefault("metadata", {})
        extraction["metadata"].update(call_metadata)
        extraction["metadata"]["source_file"] = filepath.name
        extraction["metadata"]["extraction_timestamp"] = datetime.now().isoformat()
        extraction["metadata"]["extraction_mode"] = "full_transcript"

        # Classify tier
        speakers = extraction.get("prepared_remarks", {}).get("speakers", [])
        extraction["metadata"]["citation_tier"] = classify_citation_tier(
            extraction["metadata"], speakers
        )
        return extraction

    extraction = {
        "metadata": {
            **call_metadata,
            "source_file": filepath.name,
            "filing_type": "earnings_transcript",
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_mode": "split_sections",
            "sections_extracted": list(section_results.keys()),
        }
    }

    # Merge prepared remarks
    if "prepared_remarks" in section_results:
        result = section_results["prepared_remarks"]
        if "prepared_remarks" in result:
            extraction["prepared_remarks"] = result["prepared_remarks"]
        else:
            extraction["prepared_remarks"] = result
    else:
        extraction["prepared_remarks"] = "NOT_PROVIDED"

    # Merge Q&A
    if "qanda" in section_results:
        result = section_results["qanda"]
        if "qanda" in result:
            extraction["qanda"] = result["qanda"]
        else:
            extraction["qanda"] = result
    else:
        extraction["qanda"] = "NOT_PROVIDED"

    # Classify citation tier
    speakers = []
    if isinstance(extraction.get("prepared_remarks"), dict):
        speakers = extraction["prepared_remarks"].get("speakers", [])
    extraction["metadata"]["citation_tier"] = classify_citation_tier(
        extraction["metadata"], speakers
    )

    return extraction


def validate_extraction(extraction: dict) -> list[str]:
    """Run quality checks on the assembled transcript extraction."""
    warnings = []

    meta = extraction.get("metadata", {})

    # --- Metadata checks ---
    if meta.get("call_type") == "NOT_PROVIDED":
        warnings.append("Call type not identified — transcript may be classified as Tier 3")
    if meta.get("call_date") == "NOT_PROVIDED":
        warnings.append("Call date not identified — transcript will be classified as Tier 3 (excluded from evidence)")

    tier_info = meta.get("citation_tier", {})
    if tier_info.get("tier") == 3:
        warnings.append("TIER 3 TRANSCRIPT — will be excluded from evidentiary use in the report")
    elif tier_info.get("tier") == 2:
        missing = tier_info.get("missing_fields", [])
        warnings.append(f"Tier 2 transcript — missing fields: {', '.join(missing)}")

    # --- Prepared Remarks checks ---
    pr = extraction.get("prepared_remarks", {})
    if pr == "NOT_PROVIDED":
        warnings.append("Prepared remarks section entirely missing")
    elif isinstance(pr, dict):
        speakers = pr.get("speakers", [])
        if not speakers:
            warnings.append("No speakers identified in prepared remarks")
        else:
            exec_speakers = [s for s in speakers if s.get("is_company_executive")]
            if not exec_speakers:
                warnings.append("No company executives identified among speakers")
            unnamed = [s for s in exec_speakers if not s.get("full_name") or s["full_name"] == "NOT_PROVIDED"]
            if unnamed:
                warnings.append(f"{len(unnamed)} executive speaker(s) without names identified")

        themes = pr.get("key_themes", [])
        if not themes:
            warnings.append("No key themes extracted from prepared remarks")
        elif len(themes) < 3:
            warnings.append(f"Only {len(themes)} themes extracted from prepared remarks — unusually low")

        metrics = pr.get("financial_metrics_cited", [])
        if not metrics:
            warnings.append("No financial metrics extracted from prepared remarks — management almost always cites numbers")

        guidance = pr.get("guidance_and_outlook", [])
        if not guidance:
            warnings.append("No guidance/outlook statements extracted from prepared remarks")

        commitments = pr.get("commitments_and_priorities", [])
        if not commitments:
            warnings.append("No commitments or priorities extracted — needed for follow-through analysis")

        kpis = pr.get("management_kpis_emphasized", [])
        if not kpis:
            warnings.append("No management-emphasized KPIs extracted from prepared remarks")

        pricing = pr.get("pricing_and_demand_commentary", [])
        # Pricing commentary is not always present — only warn if it seems like it should be
        # Not warning here, just noting it's empty

    # --- Q&A checks ---
    qa = extraction.get("qanda", {})
    if qa == "NOT_PROVIDED":
        warnings.append("Q&A section entirely missing")
    elif isinstance(qa, dict):
        exchanges = qa.get("exchanges", [])
        if not exchanges:
            warnings.append("No Q&A exchanges extracted — this section is critical for communication clarity assessment")
        elif len(exchanges) < 3:
            warnings.append(f"Only {len(exchanges)} Q&A exchanges extracted — most calls have 8-15+ questioners")

        stats = qa.get("qa_summary_stats", {})
        total_q = stats.get("total_analyst_questioners")
        if total_q is not None and len(exchanges) != total_q:
            warnings.append(f"Inconsistency: {total_q} questioners reported but {len(exchanges)} exchanges extracted")

        # Check directness coverage
        if exchanges:
            total_responses = sum(len(e.get("responses", [])) for e in exchanges)
            direct = stats.get("questions_answered_directly", 0) or 0
            partial = stats.get("questions_answered_partially", 0) or 0
            evasive = stats.get("questions_evaded", 0) or 0
            deferred = stats.get("questions_deferred", 0) or 0
            classified = direct + partial + evasive + deferred
            if classified > 0 and total_responses > 0 and abs(classified - total_responses) > 2:
                warnings.append(f"Directness classification gap: {total_responses} responses but {classified} classified")

        new_info = qa.get("new_information_revealed", [])
        # New info is valuable but not always present — don't warn

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_single_file(filepath: Path, api_key: str) -> Path | None:
    """Process a single earnings transcript."""
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
        print("  ERROR: Extracted text is too short. File may be corrupted or not a transcript.")
        return None

    # Step 2: Extract metadata
    print("\n[Step 2] Extracting call metadata...")
    call_metadata = extract_call_metadata(full_text)
    print(f"  Call type: {call_metadata['call_type']}")
    print(f"  Call date: {call_metadata['call_date']}")
    print(f"  Period: {call_metadata['period']}")

    if call_metadata["call_date"] == "NOT_PROVIDED":
        print("  ⚠ WARNING: No call date found — this transcript will be Tier 3 (excluded from evidence)")

    # Step 3: Split transcript
    print("\n[Step 3] Splitting into Prepared Remarks and Q&A...")
    sections = split_transcript(full_text)

    # Step 4: Run extractions
    print("\n[Step 4] Running Claude API extractions...")
    section_results = asyncio.run(run_extractions(sections, api_key))

    if not section_results:
        print("  ERROR: All extractions failed.")
        return None

    # Step 5: Assemble
    print("\n[Step 5] Assembling extraction...")
    extraction = assemble_extraction(section_results, filepath, call_metadata)

    # Step 6: Validate
    print("\n[Step 6] Validating extraction...")
    warnings = validate_extraction(extraction)
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    ⚠ {w}")
        extraction["metadata"]["validation_warnings"] = warnings
    else:
        print("  ✓ All checks passed")

    # Step 7: Save
    output_name = filepath.stem + "_extraction.json"
    output_path = OUTPUT_DIR / output_name
    output_path.write_text(
        json.dumps(extraction, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  ✓ Saved to {output_path}")

    # Print summary
    tier = extraction["metadata"].get("citation_tier", {})
    print(f"  Citation tier: {tier.get('tier_label', 'Unknown')} ({'admissible' if tier.get('admissible') else 'EXCLUDED'})")

    if isinstance(extraction.get("prepared_remarks"), dict):
        themes = len(extraction["prepared_remarks"].get("key_themes", []))
        metrics = len(extraction["prepared_remarks"].get("financial_metrics_cited", []))
        guidance = len(extraction["prepared_remarks"].get("guidance_and_outlook", []))
        commitments = len(extraction["prepared_remarks"].get("commitments_and_priorities", []))
        print(f"  Prepared Remarks: {themes} themes, {metrics} metrics, {guidance} guidance items, {commitments} commitments")

    if isinstance(extraction.get("qanda"), dict):
        exchanges = len(extraction["qanda"].get("exchanges", []))
        stats = extraction["qanda"].get("qa_summary_stats", {})
        evasions = len(extraction["qanda"].get("evasion_and_deflection_instances", []))
        new_info = len(extraction["qanda"].get("new_information_revealed", []))
        print(f"  Q&A: {exchanges} exchanges, {evasions} evasion instances, {new_info} new information items")
        if stats.get("questions_answered_directly") is not None:
            print(f"  Directness: {stats.get('questions_answered_directly', 0)} direct, "
                  f"{stats.get('questions_answered_partially', 0)} partial, "
                  f"{stats.get('questions_evaded', 0)} evasive, "
                  f"{stats.get('questions_deferred', 0)} deferred")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from earnings call transcripts using Claude API"
    )
    parser.add_argument(
        "path",
        help="Path to a single transcript (PDF/HTML/TXT) or a folder of transcripts",
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
        print(f"Found {len(files)} transcript(s) to process")
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
    print(f"  Processed: {len(results)} transcript(s)")
    print(f"  Succeeded: {succeeded}")
    if failed > 0:
        print(f"  Failed:    {failed}")
    print(f"  Output:    {OUTPUT_DIR.resolve()}")
    for name, path in results:
        status = f"✓ {path.name}" if path else "✗ FAILED"
        print(f"    {name} → {status}")


if __name__ == "__main__":
    main()
