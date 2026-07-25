from pathlib import Path

import yaml


def load_config(path):
    """Read a YAML config file into a dict."""
    with open(Path(path)) as f:
        return yaml.safe_load(f)


def format_results(results):
    """Render an evaluate() results dict as aligned lines for printing."""
    width = max(len(k) for k in results)
    return "\n".join(f"  {k.ljust(width)} : {v}" for k, v in results.items())