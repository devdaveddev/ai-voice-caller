"""Per-call artifacts in ./recordings: <ts>_<bot>_<callsid>.{wav,txt,json}."""
import json
import logging
from datetime import datetime

import httpx

from .config import RECORDINGS_DIR, env

log = logging.getLogger("storage")

_bases: dict[str, str] = {}


def register_call(call_sid: str, bot: str) -> str:
    base = _bases.get(call_sid)
    if not base:
        base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{bot}_{call_sid}"
        _bases[call_sid] = base
    return base


def base_for(call_sid: str) -> str:
    return _bases.get(call_sid) or register_call(call_sid, "unknown")


def write_call_files(call_sid: str, transcript_text: str, record: dict) -> None:
    base = base_for(call_sid)
    (RECORDINGS_DIR / f"{base}.txt").write_text(transcript_text, encoding="utf-8")
    json_path = RECORDINGS_DIR / f"{base}.json"
    if json_path.exists():  # keep the recording_file if the recording arrived first
        try:
            record = {**json.loads(json_path.read_text(encoding="utf-8")), **record}
        except ValueError:
            pass
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _patch_json(base: str, **fields) -> None:
    p = RECORDINGS_DIR / f"{base}.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data.update(fields)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def download_recording(call_sid: str, recording_url: str) -> None:
    base = base_for(call_sid)
    dest = RECORDINGS_DIR / f"{base}.wav"
    auth = (env("TWILIO_ACCOUNT_SID"), env("TWILIO_AUTH_TOKEN"))
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
        r = await c.get(recording_url + ".wav", auth=auth)
    if r.status_code != 200:
        log.error("recording download failed %s %s", r.status_code, r.text[:200])
        return
    dest.write_bytes(r.content)
    _patch_json(base, recording_file=dest.name)
    log.info("saved recording %s (%d bytes)", dest.name, len(r.content))
