"""FastAPI app: /twiml, /media-stream (websocket), /status, /recording-status, /call."""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
from xml.sax.saxutils import quoteattr

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import JSONResponse

from . import dialer, storage
from .config import BotConfig, env
from .llm import make_llm
from .session import CallSession
from .stt import SarvamSTT
from .tts import SarvamTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")
app = FastAPI(title="Outbound AI caller")

stt, tts, llm = SarvamSTT(), SarvamTTS(), make_llm()
_overrides: dict[str, dict] = {}  # call_sid -> placeholder overrides


def _valid_signature(request: Request, form: dict) -> bool:
    """Twilio X-Twilio-Signature check: HMAC-SHA1(token, url + sorted key+value pairs)."""
    if env("VALIDATE_TWILIO_SIGNATURE", "true").lower() != "true":
        return True
    url = env("PUBLIC_BASE_URL").rstrip("/") + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    data = url + "".join(k + form[k] for k in sorted(form))
    mac = hmac.new(env("TWILIO_AUTH_TOKEN").encode(), data.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(base64.b64encode(mac).decode(), request.headers.get("X-Twilio-Signature", ""))


async def _form(request: Request) -> dict:
    form = {k: str(v) for k, v in (await request.form()).items()}
    if not _valid_signature(request, form):
        raise HTTPException(403, "bad Twilio signature")
    return form


@app.post("/twiml")
async def twiml(request: Request):
    form = await _form(request)
    bot = request.query_params.get("bot", "loan")
    ws_url = env("PUBLIC_BASE_URL").rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    params = {"bot": bot, "to": form.get("To", ""), "token": _stream_token(form.get("CallSid", "")), **{k[4:]: v for k, v in request.query_params.items() if k.startswith("var_")}}
    plist = "".join(f"<Parameter name={quoteattr(k)} value={quoteattr(v)}/>" for k, v in params.items())
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{ws_url}/media-stream">{plist}</Stream></Connect></Response>'
    return Response(xml, media_type="application/xml")


def _stream_token(call_sid: str) -> str:
    return hmac.new(env("TWILIO_AUTH_TOKEN").encode(), b"stream:" + call_sid.encode(), hashlib.sha256).hexdigest()[:32]


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    # Twilio sends `connected` then `start` (with customParameters). Read until start to build the session.
    first = None
    while first is None:
        msg = json.loads(await ws.receive_text())
        if msg.get("event") == "start":
            first = msg
    p = first["start"].get("customParameters", {})
    call_sid = first["start"].get("callSid", "")
    if not hmac.compare_digest(p.get("token", ""), _stream_token(call_sid)):
        log.warning("rejecting media stream with bad token (call=%s)", call_sid)
        await ws.close(code=1008)
        return
    overrides = {k: v for k, v in p.items() if k not in ("bot", "to", "token")}
    bot = BotConfig(p.get("bot", "loan"), overrides)
    session = CallSession(ws, bot, stt, tts, llm, dialer.hangup, call_sid=call_sid, to=p.get("to", ""))
    await session.run(initial=[first])


@app.post("/status")
async def status(request: Request):
    form = await _form(request)
    log.info("call %s status=%s", form.get("CallSid"), form.get("CallStatus"))
    sid, st = form.get("CallSid", ""), form.get("CallStatus", "")
    if st in ("busy", "failed", "no-answer", "canceled"):
        storage.write_call_files(sid, "", {"call_sid": sid, "to": form.get("To"), "end_reason": st,
                                           "outcome": {"committed": False, "summary": f"Call not connected ({st})"}})
    return Response(status_code=204)


@app.post("/recording-status")
async def recording_status(request: Request):
    form = await _form(request)
    if form.get("RecordingStatus") == "completed":
        asyncio.create_task(storage.download_recording(form["CallSid"], form["RecordingUrl"]))
    return Response(status_code=204)


@app.post("/call")
async def call_endpoint(request: Request, payload: dict):
    """JSON: {"to": "+91...", "bot": "loan", "force": false}. Local use only; do not expose publicly."""
    if request.headers.get("x-forwarded-for"):  # came through ngrok, not from this machine
        raise HTTPException(403, "/call is local-only")
    to, bot = payload.get("to", ""), payload.get("bot", "loan")
    ok, decision = dialer.compliance_check(bool(payload.get("force")), to, bot)
    if not ok:
        return JSONResponse({"error": "outside 08:00-19:00 IST", "hint": "set force=true to override"}, status_code=403)
    res = await dialer.place_call(to, bot)
    return {"call_sid": res["sid"], "status": res["status"], "compliance": decision}


@app.get("/health")
async def health():
    return {"ok": True}
