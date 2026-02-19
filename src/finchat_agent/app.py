"""CLI entrypoint for running the financial Q&A agent."""

from __future__ import annotations

import sys

from .graph import run_query


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m finchat_agent.app "<user query>"')
        return 1
    query = sys.argv[1]
    state = run_query(query)
    print(state["answer_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
