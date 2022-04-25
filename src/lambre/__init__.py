from pathlib import Path

RELATION_MAP = f"{Path(__file__).parent.resolve()}/relation_map"
RULE_LINKS = f"{Path(__file__).parent.resolve()}/rule_links"

from .download import download_lambre_files as download
from .metric import compute_metric as score
