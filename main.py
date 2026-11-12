"""Entry point: build indexes, build the agent, run a sample query."""

import asyncio
import sys

from agent import build_agent
from config import configure_settings
from ingestion import build_indexes
from tools import build_all_tools

DEFAULT_QUESTION = (
    "According to the McKinsey report, what are the main organizational changes "
    "companies are making for AI adoption? Also search ArXiv for recent papers on "
    "AI governance to add context."
)


async def main(question: str) -> None:
    configure_settings()
    summary_index, vector_index = build_indexes()
    agent = build_agent(build_all_tools(summary_index, vector_index))

    print(f"\nQuestion: {question}\n{'=' * 60}")
    response = await agent.run(question)
    print(response)


if __name__ == "__main__":
    # Optional: python main.py "your question"
    asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_QUESTION))
