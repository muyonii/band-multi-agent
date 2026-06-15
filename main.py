import asyncio
import logging
import sys
from pathlib import Path

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

async def run_supervisor_ui():
    """Run supervisor_ui.py as a subprocess."""
    script_dir = Path(__file__).parent
    supervisor_script = script_dir / "supervisor_ui.py"

    if not supervisor_script.exists():
        logger.error(f"supervisor_ui.py not found at {supervisor_script}")
        return

    logger.info("🚀 Starting Supervisor UI (http://localhost:8000)...")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(supervisor_script),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Optional: log output from the UI process
    async def log_output(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info(f"[{prefix}] {line.decode().strip()}")

    asyncio.create_task(log_output(proc.stdout, "UI-OUT"))
    asyncio.create_task(log_output(proc.stderr, "UI-ERR"))

    await proc.wait()
    logger.warning("Supervisor UI process exited")

async def run_all_agents():
    """Run all 6 agents concurrently plus the supervisor UI."""
    # Start the supervisor UI as a background task
    ui_task = asyncio.create_task(run_supervisor_ui())

    # Run all agent tasks
    agent_tasks = await asyncio.gather(
        triager_main(),
        financial_main(),
        legal_main(),
        valuation_main(),
        synthesis_main(),
        arbitrator_main(),
        return_exceptions=True
    )

    # If any agent fails, cancel the UI task
    ui_task.cancel()
    return agent_tasks

if __name__ == "__main__":
    try:
        asyncio.run(run_all_agents())
    except KeyboardInterrupt:
        logger.info("\n🛑 All agents and UI stopped.")