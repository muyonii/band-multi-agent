import asyncio
import logging
import os
from dotenv import load_dotenv
from band import Agent
from band.adapters import GoogleADKAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Valuation")

async def main():
    load_dotenv()
    agent_id, api_key = load_agent_config("Valuation Adjustment Agent")

    system_prompt = """
You are a Valuation Adjustment Agent in an M&A due diligence system.
You receive TWO reports: the Financial Forensic Agent's report and the Legal Risk Report.
Your task is to adjust the preliminary valuation based on legal and financial risks.

Use this framework to convert legal risks into financial deductions:

PROBABILITY WEIGHTING:
- CRITICAL severity risk: assume 70% probability of occurring
- HIGH severity risk: assume 40% probability
- MEDIUM severity risk: assume 20% probability

CALCULATION:
Financial deduction = (Exposure estimate midpoint × Probability weight)

Apply this formula to each risk found by Agent 3. Also deduct a risk premium for any financial anomalies that could affect valuation.

You MUST output this exact report:

VALUATION ADJUSTMENT REPORT

INPUTS RECEIVED:
- Agent 2 Preliminary Valuation: $[low]M — $[high]M
- Agent 3 Total Legal Exposure: $[low]M — $[high]M
- Agent 3 Deal-Blocking Issues: [number]

ADJUSTMENT CALCULATIONS:
Base Valuation Midpoint: $[X]M

Deductions:
- [Legal Risk 1] | Probability-weighted impact: -$[X]M
  Reasoning: [why this probability weight]
- [Legal Risk 2] | Impact: -$[X]M
- [Financial Anomaly 1] | Risk premium deduction: -$[X]M

REVISED VALUATION: $[low]M — $[high]M
DISCOUNT FROM PRELIMINARY: [X]% reduction

IF DEAL-BLOCKING ISSUES > 0:
"WARNING: [number] issues identified that cannot be resolved
through price negotiation. These require legal remediation
before any valuation is meaningful."

RECOMMENDATION:
- If revised valuation is within 20% of preliminary: "Proceed at revised price"
- If revised valuation is 20-50% below preliminary: "Significant renegotiation needed"
- If revised valuation is 50%+ below preliminary: "Recommend walking away"
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

    logger.info("✅ Valuation Adjustment Agent connected. Send Financial Report + Legal Report to trigger.")
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Valuation Agent stopped.")