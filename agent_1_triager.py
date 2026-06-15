import asyncio
import logging
import os
from dotenv import load_dotenv
from band import Agent
from band.adapters import GoogleADKAdapter
from band.config import load_agent_config
from google.adk.models.lite_llm import LiteLlm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Triager")

# ------------------------------------------------------------------
# Custom adapter that prints LLM output to terminal before returning it
# ------------------------------------------------------------------
# class PrintingGoogleADKAdapter(GoogleADKAdapter):
#     """Adapter that prints every LLM response to terminal before passing it on."""
#
#     async def generate(self, *args, **kwargs):
#         """Override the generate method to capture and print the response."""
#         # Call the original generate method to get the LLM output
#         response = await super().generate(*args, **kwargs)
#
#         # Extract the text content (adjust this based on actual response structure)
#         if hasattr(response, 'text'):
#             output_text = response.text
#         elif isinstance(response, dict):
#             output_text = response.get('text', response.get('content', str(response)))
#         else:
#             output_text = str(response)
#
#         # Print to terminal first
#         print("\n" + "="*60)
#         print("🔵 LLM OUTPUT (captured before posting):")
#         print("="*60)
#         print(output_text)
#         print("="*60 + "\n")
#
#         # Also log it at INFO level for consistency
#         logger.info("LLM output preview:\n%s", output_text[:500] + ("..." if len(output_text) > 500 else ""))
#
#         # Return the response so it can be posted
#         return response

# ------------------------------------------------------------------
# Placeholder: replace this with your actual POST logic
# ------------------------------------------------------------------
async def post_output(llm_response):
    """Simulate posting the LLM output to an external endpoint."""
    # Example: await httpx.post("https://your-endpoint.com", json={"output": llm_response})
    logger.info("📡 Would now POST the LLM output to downstream system (replace with actual HTTP POST).")
    # For demonstration, just print a message
    print("📤 [POST] Output would be sent to external system now.")

# ------------------------------------------------------------------
# Main async workflow
# ------------------------------------------------------------------
async def main():
    load_dotenv()
    agent_id, api_key = load_agent_config("Document Triager")

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
You must output ONLY valid, strict JSON matching the exact schema below. Do not include markdown formatting (such as ```json), narrative prose, conversational filler, or routing tags (like @mentions).

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
    featherless_model = LiteLlm(
        model="openai/Qwen/Qwen3-8B",  # Change to any model Featherless supports
        api_key=os.getenv("FEATHERLESS_API_KEY"),  # Must be set in .env
        api_base="https://api.featherless.ai/v1"  # Featherless OpenAI‑compatible endpoint
    )

    # Use the custom adapter that prints output first
    adapter = GoogleADKAdapter(
        model=featherless_model,
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
    
    # Run the agent – it will now print every LLM response before returning it.
    # If agent.run() returns a final response, capture it.
    # If it's a long-running listener, the printing happens inside generate() above.
    final_output = await agent.run()
    
    # If the agent run returns something, post it after printing.
    if final_output is not None:
        await post_output(final_output)
    else:
        # If agent.run() is non‑blocking and doesn't return, the printing still happens
        # because every generate() call prints. To also post, you'd need a separate
        # mechanism (e.g., event hook). Below is an example of how to extend:
        logger.warning("Agent.run() returned None – if the agent is listening passively, "
                       "you may need to attach a callback to the adapter instead.")
        # Example callback approach (if the framework supports it):
        # adapter.on_response = lambda resp: asyncio.create_task(post_output(resp))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Triager stopped.")