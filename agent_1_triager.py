import asyncio
import logging
import os
from dotenv import load_dotenv
from band import Agent
from band.adapters import GoogleADKAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Triager")

async def main():
    load_dotenv()
    agent_id, api_key = load_agent_config("Document Triager")

    # The system prompt: everything the agent needs to know to do its job.
    system_prompt = """
You are a document triager in an M&A due diligence multi‑agent system.
Your ONLY job is to read raw extracted text from multiple documents and output a STRICT JSON classification.
You do NOT perform deep analysis. You are a paralegal who sorts the case files before the lawyers see them.

When you receive a message, it contains:
- Raw extracted text from all 3 documents combined (financial, legal, company overview)
- Acquisition context (buyer summary, deal ceiling)

You MUST output the following JSON structure and nothing else (no narrative, no markdown fences, just the JSON object):

{
  "acquisition_context": "Buyer summary and deal ceiling",
  "documents": [
    {
      "id": "DOC-FIN-001",
      "type": "Financial",
      "title": "Balance Sheet FY2023",
      "page_count_estimate": 2,
      "summary": "2-3 sentence plain language summary",
      "initial_flags": [
        "Missing Q3 revenue entry",
        "Accounts payable 340% above industry average"
      ],
      "assigned_to": "Financial Forensic Agent"
    },
    {
      "id": "DOC-LEG-001",
      "type": "Legal",
      "title": "Vendor Service Agreement",
      "summary": "2-3 sentence summary",
      "initial_flags": [
        "Uncapped liability clause detected in section 7",
        "IP ownership language is ambiguous"
      ],
      "assigned_to": "Legal & Compliance Analyst"
    },
    {
      "id": "DOC-OVW-001",
      "type": "Company Overview",
      "title": "TechCorp Company Profile",
      "summary": "2-3 sentence summary",
      "initial_flags": [],
      "assigned_to": "Both Financial and Legal Agents"
    }
  ],
  "overall_complexity_rating": "HIGH",
  "recommended_priority_focus": ["Legal", "Financial"],
  "total_initial_flags": 4
}

Rules for classification and flagging:
- Distinguish financial, legal, and overview documents by content.
- "Anomalous" means missing data, unusual round numbers, vague legal language, absent expected clauses, outliers.
- If something looks unusual, flag it and let the specialist determine severity. When in doubt, flag it — over-flagging is better than missing a red flag.
- If two documents look like the same type, classify by dominant content and note the ambiguity in the summary.
- Estimate page count from the length and structure of the text.
- The output must be strict JSON — no prose, no markdown, no code fences. Just the JSON object.

call "@hexo/financial-forensic-agent" after every prompt and send the financial document's raw text and your JSON output to it.
call "@hexo/legal-compliance-agent" after every prompt and send the legal document's raw text and your JSON output to it.
"""

    adapter = GoogleADKAdapter(
        model="gemini-2.5-flash",
        custom_section=system_prompt,
        enable_execution_reporting=True
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL"),
    )

    logger.info("✅ Document Triager connected. Send raw text + acquisition context to trigger.")
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Triager stopped.")