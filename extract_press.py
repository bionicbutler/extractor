"""
Press Release Extraction Pipeline
===================================
Extracts structured data from company press releases using Claude API.
Schema validated against equity research prompt v4 requirements.

The v4 prompt uses press releases to bridge the gap between the most recent
SEC filing and the report date. They feed into:
  - Section 2.1: Material changes to business description (acquisitions, divestitures)
  - Section 5.3: Post-filing developments affecting balance sheet or risk profile
  - Section 8.5: Press release developments table
  - Section 9: Investment conclusion — post-filing developments that alter assessments

Press releases are short (typically 1-3 pages each). This extractor batches
multiple releases into a single API call when possible, or processes them
individually if the batch exceeds size limits.

Usage:
    python extract_press.py path/to/press_release.txt
    python extract_press.py path/to/press_releases_folder/

Output:
    Creates a single JSON extraction file in the ./extractions/ directory
    containing all press releases for the company.

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
MAX_TOKENS = 12000
CONCURRENT_LIMIT = 5

# Max characters per batch — press releases are short, so we can batch several.
# If total text exceeds this, we split into multiple API calls.
BATCH_CHAR_LIMIT = 150_000

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
# Batching
# ---------------------------------------------------------------------------
def create_batches(files_and_texts: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """
    Group press releases into batches that fit within the character limit.
    Each batch will be sent as a single API call.
    """
    batches = []
    current_batch = []
    current_size = 0

    for filename, text in files_and_texts:
        if current_size + len(text) > BATCH_CHAR_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append((filename, text))
        current_size += len(text)

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial document extraction engine. Your task is to read company press releases and produce a structured JSON extraction for each one.

CRITICAL RULES:
1. Extract only. Do not analyze materiality or assess impact — that is the downstream analyst's job.
2. Press releases are company communications with promotional framing. Extract the FACTS, not the company's characterization. Separate factual claims from promotional language.
3. All numerical values (deal values, revenue figures, guidance numbers) must be verbatim.
4. If information is not present, use "NOT_PROVIDED".
5. Output valid JSON only. No markdown fences, no preamble, no commentary."""


EXTRACTION_PROMPT = """Extract structured data from each press release below. There may be one or multiple press releases separated by file markers.

For EACH press release, extract the following. Output a JSON object with a "press_releases" array:

{
  "press_releases": [
    {
      "source_file": "filename if provided",
      "title": "exact title/headline of the press release",
      "date": "publication date as stated in the release",
      "category": "one of: M&A / guidance / leadership / financing / product / earnings_preliminary / restructuring / partnership / legal / dividend / share_repurchase / other",

      "factual_summary": "2-4 sentence summary of the factual content only — strip promotional language, focus on what happened",

      "key_facts": [
        {
          "fact": "a specific factual claim from the release",
          "verbatim_language": "the exact sentence or phrase from the release",
          "is_numerical": true or false,
          "numerical_value_if_applicable": "verbatim number or NOT_PROVIDED"
        }
      ],

      "financial_figures_cited": [
        {
          "metric": "what is being measured (e.g., deal value, revenue, EPS, guidance range)",
          "value": "verbatim figure",
          "context": "verbatim surrounding language",
          "is_preliminary_or_estimated": true or false,
          "is_audited": false
        }
      ],

      "guidance_updates": [
        {
          "metric": "what metric guidance is for",
          "prior_guidance_if_stated": "verbatim or NOT_PROVIDED",
          "new_guidance": "verbatim",
          "direction": "raised / lowered / maintained / initiated / withdrawn"
        }
      ],

      "material_events": {
        "acquisition_announced": {
          "applicable": true or false,
          "target": "",
          "deal_value": "verbatim or NOT_PROVIDED",
          "expected_close_date": "",
          "strategic_rationale_stated": ""
        },
        "divestiture_announced": {
          "applicable": true or false,
          "asset_or_business": "",
          "proceeds": "verbatim or NOT_PROVIDED",
          "expected_close_date": ""
        },
        "debt_transaction": {
          "applicable": true or false,
          "type": "issuance / redemption / refinancing / credit_facility_amendment / other",
          "amount": "verbatim or NOT_PROVIDED",
          "terms": "rate, maturity, other key terms"
        },
        "leadership_change": {
          "applicable": true or false,
          "who": "",
          "old_role": "",
          "new_role": "",
          "effective_date": ""
        },
        "restructuring": {
          "applicable": true or false,
          "description": "",
          "expected_charges": "verbatim or NOT_PROVIDED",
          "expected_savings": "verbatim or NOT_PROVIDED"
        },
        "dividend_action": {
          "applicable": true or false,
          "type": "initiation / increase / decrease / suspension / special",
          "amount_per_share": "verbatim or NOT_PROVIDED",
          "ex_date": "",
          "payment_date": ""
        },
        "share_repurchase_action": {
          "applicable": true or false,
          "type": "new_authorization / increase / accelerated_buyback / completion",
          "amount": "verbatim or NOT_PROVIDED"
        }
      },

      "checklist_relevance": {
        "affects_business_quality": true or false,
        "affects_capital_allocation": true or false,
        "affects_balance_sheet": true or false,
        "affects_risk_profile": true or false,
        "brief_explanation": "one sentence on which section(s) of the equity research report this development is relevant to"
      },

      "promotional_language_flagged": [
        "verbatim phrases that are clearly promotional rather than factual — e.g., 'transformative acquisition', 'industry-leading', 'best-in-class'. These should be noted but NOT used as factual evidence."
      ]
    }
  ]
}

IMPORTANT:
- Separate FACTS from FRAMING. A press release saying "completed the transformative acquisition of XYZ for $2.1 billion" has one fact (acquired XYZ for $2.1B) and one promotional frame ("transformative"). Extract the fact, flag the framing.
- For guidance_updates, capture both the new guidance AND the prior guidance if the release states it. The direction field helps the synthesis stage quickly assess whether expectations are being raised or lowered.
- For material_events, set applicable to true ONLY for events that actually occurred in this release. Most releases will only trigger one or two event types.
- For checklist_relevance, you are NOT assessing impact. You are flagging which areas of the equity research framework this release is relevant to, so the synthesis stage knows where to incorporate it.
- Financial figures from press releases are NEVER audited. Always set is_audited to false. Preliminary results and estimates should be flagged as is_preliminary_or_estimated: true.
- If multiple press releases are provided, extract each one independently. Do not cross-reference between releases.

PRESS RELEASE TEXT:
"""


# ---------------------------------------------------------------------------
# API Calls
# ---------------------------------------------------------------------------
async def extract_batch(
    client: anthropic.AsyncAnthropic,
    batch: list[tuple[str, str]],
    batch_num: int,
    semaphore: asyncio.Semaphore,
) -> list[dict] | None:
    """Send a batch of press releases to Claude for extraction."""

    # Combine press releases with file markers
    combined_text = ""
    for filename, text in batch:
        combined_text += f"\n{'='*40}\n"
        combined_text += f"FILE: {filename}\n"
        combined_text += f"{'='*40}\n\n"
        combined_text += text
        combined_text += "\n\n"

    user_message = EXTRACTION_PROMPT + combined_text

    async with semaphore:
        filenames = [f for f, _ in batch]
        print(f"  Extracting batch {batch_num} ({len(batch)} releases, {len(combined_text):,} chars)...")
        print(f"    Files: {', '.join(filenames)}")
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
            print(f"  ✓ Batch {batch_num} complete ({input_tokens:,} in / {output_tokens:,} out)")

            # Extract the press_releases array
            if "press_releases" in result:
                return result["press_releases"]
            elif isinstance(result, list):
                return result
            else:
                return [result]

        except json.JSONDecodeError as e:
            print(f"  ✗ Batch {batch_num}: Failed to parse JSON response")
            print(f"    Error: {e}")
            debug_path = RUN_OUTPUT_DIR / f"debug_press_batch_{batch_num}.txt"
            debug_path.write_text(raw_text, encoding="utf-8")
            print(f"    Raw response saved to {debug_path}")
            return None

        except anthropic.APIError as e:
            print(f"  ✗ Batch {batch_num}: API error — {e}")
            return None


async def run_extractions(batches: list[list[tuple[str, str]]], api_key: str) -> list[dict]:
    """Run all batch extractions concurrently."""
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    tasks = [
        extract_batch(client, batch, i + 1, semaphore)
        for i, batch in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks)

    # Flatten results
    all_releases = []
    for batch_result in results:
        if batch_result is not None:
            all_releases.extend(batch_result)

    return all_releases


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_extraction(releases: list[dict]) -> list[str]:
    """Run quality checks on extracted press releases."""
    warnings = []

    if not releases:
        warnings.append("No press releases extracted")
        return warnings

    for i, pr in enumerate(releases):
        prefix = f"Release {i+1}"
        title = pr.get("title", "untitled")

        if not pr.get("title") or pr["title"] == "NOT_PROVIDED":
            warnings.append(f"{prefix}: No title extracted")

        if not pr.get("date") or pr["date"] == "NOT_PROVIDED":
            warnings.append(f"{prefix} ({title}): No date extracted — cannot confirm post-filing timing")

        if not pr.get("category") or pr["category"] == "NOT_PROVIDED":
            warnings.append(f"{prefix} ({title}): No category assigned")

        if not pr.get("key_facts"):
            warnings.append(f"{prefix} ({title}): No key facts extracted")

        # Check that material events are flagged
        events = pr.get("material_events", {})
        has_any_event = any(
            isinstance(v, dict) and v.get("applicable") is True
            for v in events.values()
        )

        relevance = pr.get("checklist_relevance", {})
        has_any_relevance = any(
            v is True
            for k, v in relevance.items()
            if k != "brief_explanation"
        )

        if has_any_event and not has_any_relevance:
            warnings.append(f"{prefix} ({title}): Has material events but no checklist relevance flagged")

        # Check financial figures flagged correctly
        figures = pr.get("financial_figures_cited", [])
        for fig in figures:
            if fig.get("is_audited") is True:
                warnings.append(f"{prefix} ({title}): Financial figure marked as audited — press release figures are never audited")

    # Check for duplicate titles
    titles = [pr.get("title", "") for pr in releases if pr.get("title")]
    if len(titles) != len(set(titles)):
        warnings.append("Duplicate press release titles detected — may indicate extraction error")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global RUN_OUTPUT_DIR
    RUN_OUTPUT_DIR = create_run_output_dir()

    parser = argparse.ArgumentParser(
        description="Extract structured data from press releases using Claude API"
    )
    parser.add_argument(
        "path",
        help="Path to a single press release (PDF/HTML/TXT) or a folder of press releases",
    )
    args = parser.parse_args()

    api_key = load_api_key()
    target = Path(args.path)

    # Collect files
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
    else:
        print(f"Path not found: {target}")
        sys.exit(1)

    print(f"Found {len(files)} press release(s) to process")

    # Step 1: Extract text from all files
    print("\n[Step 1] Extracting text from files...")
    files_and_texts = []
    for filepath in files:
        try:
            text = extract_text(filepath)
            if len(text) < 100:
                print(f"  ⚠ {filepath.name}: Too short ({len(text)} chars), skipping")
                continue
            files_and_texts.append((filepath.name, text))
            print(f"  ✓ {filepath.name} ({len(text):,} chars)")
        except Exception as e:
            print(f"  ✗ {filepath.name}: Failed — {e}")

    if not files_and_texts:
        print("ERROR: No valid press release text extracted.")
        sys.exit(1)

    # Step 2: Create batches
    print(f"\n[Step 2] Creating batches...")
    batches = create_batches(files_and_texts)
    print(f"  {len(files_and_texts)} releases → {len(batches)} batch(es)")

    # Step 3: Run extractions
    print(f"\n[Step 3] Running Claude API extractions...")
    all_releases = asyncio.run(run_extractions(batches, api_key))

    if not all_releases:
        print("  ERROR: All extractions failed.")
        sys.exit(1)

    print(f"\n  Extracted {len(all_releases)} press release(s)")

    # Step 4: Validate
    print("\n[Step 4] Validating extractions...")
    warnings = validate_extraction(all_releases)
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    ⚠ {w}")
    else:
        print("  ✓ All checks passed")

    # Step 5: Assemble and save
    extraction = {
        "metadata": {
            "extraction_type": "press_releases",
            "extraction_timestamp": datetime.now().isoformat(),
            "total_releases_processed": len(all_releases),
            "source_files": [f for f, _ in files_and_texts],
            "validation_warnings": warnings if warnings else [],
        },
        "press_releases": all_releases,
    }

    # Sort by date if possible
    def sort_key(pr):
        d = pr.get("date", "")
        if d and d != "NOT_PROVIDED":
            return d
        return "9999"

    extraction["press_releases"].sort(key=sort_key)

    output_name = "press_releases_extraction.json"
    output_path = RUN_OUTPUT_DIR / output_name
    output_path.write_text(
        json.dumps(extraction, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n  ✓ Saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total releases: {len(all_releases)}")

    # Category breakdown
    categories = {}
    for pr in all_releases:
        cat = pr.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"  Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Material events
    event_types = {}
    for pr in all_releases:
        events = pr.get("material_events", {})
        for event_type, details in events.items():
            if isinstance(details, dict) and details.get("applicable"):
                event_types[event_type] = event_types.get(event_type, 0) + 1
    if event_types:
        print(f"  Material events:")
        for et, count in sorted(event_types.items(), key=lambda x: -x[1]):
            print(f"    {et}: {count}")

    # Checklist relevance
    relevance_areas = {
        "affects_business_quality": 0,
        "affects_capital_allocation": 0,
        "affects_balance_sheet": 0,
        "affects_risk_profile": 0,
    }
    for pr in all_releases:
        rel = pr.get("checklist_relevance", {})
        for area in relevance_areas:
            if rel.get(area):
                relevance_areas[area] += 1
    print(f"  Checklist relevance:")
    for area, count in relevance_areas.items():
        if count > 0:
            print(f"    {area.replace('_', ' ').title()}: {count} release(s)")


if __name__ == "__main__":
    main()
