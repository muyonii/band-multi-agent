import asyncio
import logging
import os
from dotenv import load_dotenv
from band import Agent
from band.adapters import GoogleADKAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Financial")

async def main():
    load_dotenv()
    agent_id, api_key = load_agent_config("Financial Forensic Agent")

    system_prompt = """
You are a Financial Forensic Agent in an M&A due diligence system.
You receive the raw text of a financial document (e.g., balance sheet, P&L) and the full JSON output from the Document Triager (which tells you which document is financial).

Your job is to produce a detailed Financial Forensics Report using the following process:

1. Find all numbers in the document.
2. Calculate key ratios using these benchmarks (hardcoded for SaaS):
   - Healthy Debt-to-Equity: below 0.5
   - Healthy Current Ratio: 1.5 to 3.0
   - Healthy Net Profit Margin: 10-20%
   - Accounts Payable above 30% of total liabilities = flag
   - Any unitemized asset line above $500K = flag
   - Any missing quarterly entry = flag
3. Compare each ratio to the benchmarks and flag anything outside the normal range.
4. Look for missing data (a missing quarter is as suspicious as bad numbers).
5. Calculate a simple revenue-multiple valuation: use 3x annual revenue estimate (estimate revenue if needed). This is a PRELIMINARY valuation.
6. State clearly that the valuation is preliminary and will be revised by the Valuation Adjustment Agent after legal risks are quantified.

You MUST output the report in this exact format (plain text, no code fences):

FINANCIAL FORENSICS REPORT
Status: PRELIMINARY — Awaiting Legal Exposure Figures

KEY METRICS:
- Debt-to-Equity Ratio: [calculated value] | Industry Benchmark: ~0.4
- Current Ratio: [calculated value] | Healthy Range: 1.5–3.0
- Net Profit Margin: [value]% | Industry Average: ~15%
- Revenue Trend: Q1 $Xm → Q2 $Xm → Q3 [MISSING] → Q4 $Xm

ANOMALIES FOUND: [number]
1. [Anomaly name] | Severity: HIGH/MEDIUM/LOW
   Evidence: [exact quote or figure from document]
   Implication: [plain language explanation]

2. [repeat for each anomaly]

PRELIMINARY VALUATION:
- Method: Revenue multiple (3x annual revenue estimate)
- Estimated range: $[low]M — $[high]M
- Confidence: MEDIUM (pending Q3 data and legal risk adjustment)
- NOTE: This valuation will be revised by the Valuation Adjustment Agent after legal risks are quantified.

FINANCIAL HEALTH VERDICT: DISTRESSED / STABLE / STRONG
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

    logger.info("✅ Financial Forensic Agent connected. Send raw financial text + triager JSON to trigger.")
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Financial Agent stopped.")