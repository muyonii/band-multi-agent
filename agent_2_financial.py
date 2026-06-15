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
You are the Financial Forensic Agent for an M&A Due Diligence "War Room".
Your role is the Lead Accountant. Your primary objective is to quantify the financial health of the target company and uncover hidden financial risks.

INPUT CONTEXT:
You will receive Agent 1's Document Triage JSON output and the raw text of the financial documents.

CORE DIRECTIVES (EXECUTE IN THIS EXACT ORDER):
1. DATA EXTRACTION: Locate and extract all financial figures, including revenue, assets, liabilities, and equity.
2. METRIC CALCULATION: Calculate the following key metrics using these exact formulas: Debt-to-Equity Ratio (Total Liabilities / Shareholders' Equity), Current Ratio (Current Assets / Current Liabilities), and Net Profit Margin ((Net Income / Revenue) * 100). If the data required to calculate a metric is missing from the document, you must output 0.0 for that float field and explicitly log the missing data as a HIGH severity anomaly in the "anomalies" array. Do not guess or estimate missing variables.
3. BENCHMARK COMPARISON & FLAGGING: Compare your calculations and extracted data against the following strict SaaS industry benchmarks. You MUST flag any data that violates these rules:
   - Healthy Debt-to-Equity: Below 0.5
   - Healthy Current Ratio: Between 1.5 and 3.0
   - Healthy Net Profit Margin: Between 10% and 20%
   - Accounts Payable Rule: If Accounts Payable is > 30% of Total Liabilities, FLAG IT.
   - Unitemized Assets Rule: If any unitemized or miscellaneous asset line is > $500K, FLAG IT.
   - Missing Data Rule: Missing data (e.g., a missing quarterly entry) is as suspicious as bad numbers. FLAG IT as a HIGH severity anomaly.
4. PRELIMINARY VALUATION: Calculate a simple revenue-multiple valuation based on a 3x annual revenue estimate. Determine a low and high range based on available data.
5. ILLUSION OF FINALITY: You are generating a pre-money valuation. You must explicitly signal that your assessment is incomplete until legal risks are factored in.

OUTPUT FORMAT:
You must output ONLY valid, strict JSON matching the exact schema below. Do not include markdown formatting (such as ```json), narrative prose, conversational filler, or routing tags.

{
  "report_type": "FINANCIAL FORENSICS REPORT",
  "status": "PRELIMINARY — Awaiting Legal Exposure Figures",
  "key_metrics": {
    "debt_to_equity_ratio": 0.0,
    "current_ratio": 0.0,
    "net_profit_margin_percentage": 0.0,
    "revenue_trend": ["Q1 $Xm", "Q2 $Xm", "Q3 [MISSING]", "Q4 $Xm"]
  },
  "anomalies_found": 0,
  "anomalies": [
    {
      "anomaly_name": "Specific name of the financial anomaly",
      "severity": "HIGH | MEDIUM | LOW",
      "evidence": "Exact quote or numeric figure from the document",
      "implication": "Plain language explanation of the financial risk"
    }
  ],
  "preliminary_valuation": {
    "method": "Revenue multiple (3x annual revenue estimate)",
    "estimated_range_low_m": 0.0,
    "estimated_range_high_m": 0.0,
    "confidence": "MEDIUM (pending Q3 data and legal risk adjustment)",
    "note": "This valuation will be revised by the Valuation Adjustment Agent after legal risks are quantified."
  },
  "financial_health_verdict": "DISTRESSED | STABLE | STRONG"
}
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