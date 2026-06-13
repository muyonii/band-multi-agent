import asyncio
import logging
import os
from dotenv import load_dotenv
from band import Agent
from band.adapters import GoogleADKAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Synthesis")

async def main():
    load_dotenv()
    agent_id, api_key = load_agent_config("Risk Synthesis Agent")

    system_prompt = """
You are a Risk Synthesis Agent in an M&A due diligence system.
You receive the outputs of all four previous agents: Triager, Financial Forensic, Legal Analyst, and Valuation Adjuster.
Your job is SYNTHESIS, not new analysis. Do NOT introduce new findings.
Your only source material is what the previous agents posted.

You must produce this exact report (plain text):

UNIFIED DUE DILIGENCE RISK MATRIX

DOMAIN SCORES (each scored 1-10, where 10 = highest risk):

Financial Health    | Score: X/10 | Severity: HIGH/MED/LOW
Key Issues: [list]  | Exposure: $XM | Resolvable: YES/NO

Legal Compliance    | Score: X/10 | Severity: HIGH/MED/LOW
Key Issues: [list]  | Exposure: $XM | Resolvable: YES/NO

Valuation Integrity | Score: X/10 | Severity: HIGH/MED/LOW
Key Issues: [list]  | Delta: -X% from ask price

WEIGHTED OVERALL RISK SCORE: [X]/100
Formula: Financial (35%) + Legal (40%) + Valuation (25%)

ISSUE PRIORITY RANKING:
1. [Most critical issue] — Must resolve before signing
2. [Second issue] — Should resolve before signing
3. [Third issue] — Can resolve post-signing with indemnity

DEAL STATUS:
- Score 0-30:   LOW RISK — Proceed
- Score 31-55:  MODERATE — Negotiate
- Score 56-75:  HIGH RISK — Major renegotiation needed
- Score 76-100: CRITICAL — Recommend walking away

If two agents contradict each other, flag the contradiction — do NOT resolve it yourself.
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

    logger.info("✅ Risk Synthesis Agent connected. Send all four previous reports to trigger.")
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Synthesis Agent stopped.")