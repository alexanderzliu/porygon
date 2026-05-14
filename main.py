import argparse
import logging
import os
from logging.handlers import RotatingFileHandler

from agent.simple_agent import SimpleAgent

logger = logging.getLogger(__name__)


def setup_logging(tui: bool) -> None:
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if tui:
        # Keep stdout clean for the TUI; route logs to a rotating file so
        # long runs (hours, thousands of steps) don't grow unbounded.
        handlers = [RotatingFileHandler("porygon.log", mode="w", maxBytes=10_000_000, backupCount=3)]
    else:
        handlers = [logging.StreamHandler()]
    # force=True replaces handlers installed at import time (simple_agent.py
    # calls basicConfig at module scope), which would otherwise make this a no-op.
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)

def main():
    parser = argparse.ArgumentParser(description="Claude Plays Pokemon - Starter Version")
    parser.add_argument(
        "--rom", 
        type=str, 
        default="pokemon.gb",
        help="Path to the Pokemon ROM file"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=10, 
        help="Number of agent steps to run"
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Run with display (not headless)"
    )
    parser.add_argument(
        "--sound", 
        action="store_true", 
        help="Enable sound (only applicable with display)"
    )
    parser.add_argument(
        "--max-history", 
        type=int, 
        default=30, 
        help="Maximum number of messages in history before summarization"
    )
    parser.add_argument(
        "--load-state",
        type=str,
        default=None,
        help="Path to a saved state to load"
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Render a live terminal UI showing Claude's reasoning, actions, and cost"
    )

    args = parser.parse_args()
    setup_logging(args.tui)
    
    # Get absolute path to ROM
    if not os.path.isabs(args.rom):
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rom)
    else:
        rom_path = args.rom
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return
    
    tui = None
    if args.tui:
        from agent.tui import TUI
        tui = TUI()

    # Create and run agent
    agent = SimpleAgent(
        rom_path=rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
        max_history=args.max_history,
        load_state=args.load_state,
        tui=tui,
    )

    if tui:
        tui.start()
    try:
        logger.info(f"Starting agent for {args.steps} steps")
        steps_completed = agent.run(num_steps=args.steps)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
    finally:
        agent.stop()
        if tui:
            tui.stop()

if __name__ == "__main__":
    main()