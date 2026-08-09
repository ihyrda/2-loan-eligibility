"""Console logging configuration for the loan-eligibility project.

Logs appear in the VS Code terminal, the local Streamlit terminal, and the
Streamlit Community Cloud log view, so no logs folder is required.
"""

import logging


def configure_logging() -> None:
    """Configure project console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
