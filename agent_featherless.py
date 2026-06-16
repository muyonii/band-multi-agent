import asyncio
import logging
import os
from dotenv import load_dotenv
from band import Agent
from band.adapters import GoogleADKAdapter
from band.config import load_agent_config
from google.adk.models.lite_llm import LiteLlm   # <-- Import for custom LLMs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Triager")

async def main():
    load_dotenv()
    agent_id, api_key = load_agent_config("Document Triager")

    # The system prompt: everything the agent needs to know to do its job.
    system_prompt = """
You are the Document Triager for an M&A Due Diligence "War Room".
Your role is intake coordinator and paralegal. You organize, classify, and flag documents for specialized downstream agents.

INPUT CONTEXT:
You will receive a single large text block containing multiple extracted documents and a buyer's acquisition context.

CORE DIRECTIVES & CONSTRAINTS:
1. CONSTRAINT-AWARE BOUNDING: You are NOT performing deep analysis. You are organizing and flagging for specialists. Let the specialist agents determine severity.
2. CLASSIFICATION: Distinguish Financial documents, Legal documents, and Company Overviews based solely on text content.
3. EDGE CASE HANDLING: If two documents look like the same type, classify them by their dominant content and explicitly note the ambiguity in the "summary" field.
4. ANOMALY FLAGGING: Look for what is "anomalous" in each document type (e.g., missing data, unusual round numbers, vague language, absent expected clauses). 
5. THE GOLDEN RULE: When in doubt, flag it — it is better to over-flag than to miss something.
6. ZERO-RISK HANDLING: If a document is completely clean and contains no anomalies, you must output an empty array [] for "initial_flags". Do not invent or estimate issues.

OUTPUT FORMAT:
You must output ONLY valid, strict JSON matching the exact schema below. include markdown formatting (such as ```json), narrative prose, conversational filler, or routing tags (like @mentions).

{
  "acquisition_context": "Extract and summarize the buyer's context and deal ceiling",
  "documents": [
    {
      "id": "Assign a unique ID (e.g., DOC-FIN-001, DOC-LEG-001, DOC-OVW-001)",
      "type": "Financial | Legal | Company Overview",
      "title": "Extracted or inferred document title",
      "page_count_estimate": 0,
      "summary": "2-3 sentence plain language summary. Note any classification ambiguity here.",
      "initial_flags": [
        "Specific anomaly detected 1",
        "Specific anomaly detected 2"
      ],
      "assigned_to": "Financial Forensic Agent | Legal & Compliance Analyst | Both Financial and Legal Agents"
    }
  ],
  "overall_complexity_rating": "HIGH | MEDIUM | LOW",
  "recommended_priority_focus": [
    "List of recommended domains, e.g., Legal, Financial"
  ],
  "total_initial_flags": 0
}
"""

    # ---------- Featherless API configuration ----------
    featherless_model = LiteLlm(
        model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",   # Change to any model Featherless supports
        api_key=os.getenv("FEATHERLESS_API_KEY"),               # Must be set in .env
        api_base="https://api.featherless.ai/v1"                # Featherless OpenAI‑compatible endpoint
    )

    adapter = GoogleADKAdapter(
        model=featherless_model,          # <-- Use the Featherless model instead of gemini
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

    logger.info("✅ Document Triager connected with Featherless API. Send raw text + acquisition context to trigger.")
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Triager stopped.")