"""Twilio REST helpers (httpx, async) and the IST calling-hours compliance guard."""
import json
import logging
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

from .config import LOGS_DIR, env

log = logging.getLogger("dialer")
IST = ZoneInfo("Asia/Kolkata")
API = "https://api.twilio.com/2010-04-01/Accounts"


def _auth() -> tuple[str, str]:
    return env("TWILIO_ACCOUNT_SID"), env("TWILIO_AUTH_TOKEN")


def compliance_check(force: bool, to: str, bot: str, now: datetime | None = None) -> tuple[bool, str]:
    """Allowed 08:00 <= t < 19:00 IST. Every decision is appended to logs/compliance.log."""
    now = (now or datetime.now(IST)).astimezone(IST)
    in_window = 8 <= now.hour < 19
    if in_window:
        decision, ok = "allowed", True
    elif force:
        decision, ok = "FORCED_OUTSIDE_WINDOW", True
    else:
        decision, ok = "blocked", False
    entry = {"ts_ist": now.isoformat(timespec="seconds"), "to": to, "bot": bot, "decision": decision}
    with open(LOGS_DIR / "compliance.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    if not ok:
        log.warning("Blocked: %s is outside 08:00-19:00 IST. Use --force to override.", now.strftime("%H:%M"))
    return ok, decision


async def place_call(to: str, bot: str, extra: dict | None = None) -> dict:
    base = env("PUBLIC_BASE_URL").rstrip("/")
    params = {"bot": bot, **(extra or {})}
    data = {
        "To": to,
        "From": env("TWILIO_FROM_NUMBER"),
        "Url": f"{base}/twiml?{urlencode(params)}",
        "Method": "POST",
        "StatusCallback": f"{base}/status",
        "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
        "Record": "true",
        "RecordingChannels": "dual",
        "RecordingStatusCallback": f"{base}/recording-status",
        "RecordingStatusCallbackEvent": ["completed"],
    }
    sid = env("TWILIO_ACCOUNT_SID")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{API}/{sid}/Calls.json", data=data, auth=_auth())
    if r.status_code >= 300:
        raise RuntimeError(f"Twilio error {r.status_code}: {r.text}")
    return r.json()


async def hangup(call_sid: str) -> None:
    sid = env("TWILIO_ACCOUNT_SID")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{API}/{sid}/Calls/{call_sid}.json", data={"Status": "completed"}, auth=_auth())
    if r.status_code >= 300:
        raise RuntimeError(f"hangup {r.status_code}: {r.text[:200]}")
