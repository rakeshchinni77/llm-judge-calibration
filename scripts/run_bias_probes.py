"""
run_bias_probes.py — CLI script to execute Phase 5 position and verbosity probes.

Usage:
    python scripts/run_bias_probes.py [--backend mock|hosted|local] [--input path] [--output path]

Outputs:
    results/bias.csv
    results/bias_audit_position.csv
    results/bias_audit_verbosity.csv
    results/bias_audit.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bias_probes import (
    POSITION_SAMPLE_SIZE,
    VERBOSITY_SAMPLE_SIZE,
    main as run_main,
)

logger = logging.getLogger("run_bias_probes")


def main() -> None:
    """Execute bias probing suite."""
    run_main()


if __name__ == "__main__":
    main()
