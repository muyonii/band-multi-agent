import asyncio
import logging

# Import the main functions from each agent module
from agent_1_triager import main as triager_main
from agent_2_financial import main as financial_main
from agent_3_legal import main as legal_main
from agent_4_valuation import main as valuation_main
from agent_5_synthesis import main as synthesis_main
from agent_6_arbitrator import main as arbitrator_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Orchestrator")

async def run_all_agents():
    """Run all 6 agents concurrently. Each agent's main() runs forever."""
    await asyncio.gather(
        triager_main(),
        financial_main(),
        legal_main(),
        valuation_main(),
        synthesis_main(),
        arbitrator_main()
    )

if __name__ == "__main__":
    try:
        asyncio.run(run_all_agents())
    except KeyboardInterrupt:
        logger.info("\n🛑 All agents stopped.")