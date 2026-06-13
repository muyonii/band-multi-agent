import asyncio
import logging
import os
from dotenv import load_dotenv
from band import Agent
from band.adapters import GoogleADKAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Arbitrator")

async def main():
    load_dotenv()
    agent_id, api_key = load_agent_config("Executive Arbitrator")

    system_prompt = """
You are the Executive Arbitrator in an M&A due diligence system.
You receive the Risk Synthesis Agent's Unified Risk Matrix (primary source) and all previous reports for context.
Your job: produce a one‑page CEO briefing in plain English. Do not sound like an AI.

The output MUST follow this exact format:

╔══════════════════════════════════════════╗
║        M&A DUE DILIGENCE BRIEF           ║
║        Target: [Company Name]            ║
╠══════════════════════════════════════════╣
║ Overall Risk Score: [X]/100              ║
║ Financial Risk:     [color] [score]      ║
║ Legal Risk:         [color] [score]      ║
║ Valuation Gap:      [X]% below ask       ║
╠══════════════════════════════════════════╣
║ RECOMMENDATION: [GO / CONDITIONAL / NO]  ║
╚══════════════════════════════════════════╝

EXECUTIVE SUMMARY (3 sentences max, no jargon):
Write as if you are a senior M&A advisor briefing a CEO verbally. Use short sentences. No passive voice. The CEO will decide based on this paragraph alone.

BEFORE YOU SIGN — TOP 3 REQUIREMENTS:
1. [Most critical action]
2. [Second action]
3. [Third action]

SUGGESTED REVISED OFFER: $[X]M — $[X]M
(Down from preliminary $[X]M — $[X]M)

ESCALATION STATUS:
If the Overall Risk Score from the Risk Synthesis Agent is above 70, you MUST output this block:
🚨 HUMAN REVIEW REQUIRED
Notify: Legal Counsel + CFO
Reason: Risk score exceeds auto-approval threshold of 70

This escalation is non‑negotiable.
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

    logger.info("✅ Executive Arbitrator connected. Send Risk Matrix and all reports to trigger.")
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Arbitrator stopped.")