"""Environment, config.json and prompt loading."""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RECORDINGS_DIR = ROOT / "recordings"
LOGS_DIR = ROOT / "logs"
RECORDINGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def fill(template: str, values: dict) -> str:
    """Replace {{key}} placeholders. Unknown keys are left visible so mistakes show up."""
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: str(values.get(m.group(1), m.group(0))), template)


class BotConfig:
    def __init__(self, name: str, overrides: dict | None = None):
        cfg = load_config()
        if name not in cfg["bots"]:
            raise ValueError(f"Unknown bot '{name}'. Available: {', '.join(cfg['bots'])}")
        bot = cfg["bots"][name]
        self.name = name
        self.values = {**bot["placeholders"], **(overrides or {})}
        common = (ROOT / "prompts" / "common.txt").read_text(encoding="utf-8")
        prompt = (ROOT / bot["prompt_file"]).read_text(encoding="utf-8")
        self.system_prompt = fill(prompt, {**self.values, "common": common})
        self.greeting = fill(bot["greeting"], self.values)
        self.checkin = fill(bot["checkin"], self.values)
        self.fallbacks = cfg.get("fallback_lines", {})
