# Outbound AI voice caller (Twilio + Sarvam + Anthropic)

Realtime loop: caller audio (mulaw 8 kHz) → energy VAD/endpointing → Sarvam STT → Claude (streaming) → sentence-level Sarvam TTS → mulaw back to Twilio. Barge-in sends Twilio `clear`.

## Verified against current docs (and what differs from the brief)
- **STT:** the docs no longer list "Saarika". Current models are `saaras:v4` (default) / `saaras:v3`. REST (`POST /speech-to-text`) is meant for clips under ~30 s, PCM only at 16 kHz, so we upsample each utterance to 16 kHz WAV. A streaming WebSocket exists (`wss://api.sarvam.ai/speech-to-text/ws`, supports 8 kHz and server VAD signals), but we use our own endpointing + one REST call per utterance. `app/stt.py` is the only file to change to use the WebSocket.
- **TTS:** `POST /text-to-speech` (Bulbul `bulbul:v3`, optionally `bulbul:v4-flash`), max 2500 chars. Response is base64 WAV in `audios[]`. We request WAV at 8 kHz and convert to mulaw ourselves, because the docs don't say how a `mulaw` response is containered. TTS streaming (WebSocket/HTTP) exists but isn't used: sentence-level REST calls are short enough, and the docs don't confirm `mulaw`@8 kHz for streaming.
- **Twilio:** bidirectional Media Streams need `<Connect><Stream>`, outbound audio must be base64 mulaw 8 kHz with no header; `clear` flushes buffered audio. Dual-channel recording is done via `Record=true&RecordingChannels=dual` on the Calls API.
- **Not verifiable here:** no real keys were available, so Sarvam/Anthropic/Twilio calls have not been exercised live. Tests use fakes. Expect to tune VAD thresholds on a real phone line.
- Calls per second default to 1 on Twilio. Trial accounts can only call verified numbers. Calling Indian numbers requires meeting Twilio/TRAI regulatory requirements (registered caller ID/DLT) for your account. Check this before production use.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in keys
```
1. Start ngrok: `ngrok http 8000` and put the https URL in `.env` as `PUBLIC_BASE_URL` (no trailing slash).
2. No Twilio console webhook is needed: `call.py` passes the TwiML, status and recording URLs on each call. (Optional: set nothing on the phone number.)
3. Run the server: `uvicorn app.main:app --port 8000`
4. Place a call (second terminal): `python call.py --to +91XXXXXXXXXX --bot loan`
   - `--force` dials outside 08:00–19:00 IST; every decision is logged to `logs/compliance.log`.
   - `--var customer_name=Rahul --var emi_amount=5,000` overrides `config.json` placeholders per call.
   - Alternative: `curl -X POST localhost:8000/call -H 'content-type: application/json' -d '{"to":"+91...","bot":"loan"}'` (local only).

## Test without calling
```bash
python chat.py --bot loan            # real Claude, needs ANTHROPIC_API_KEY
python chat.py --bot loan --mock     # offline plumbing check
pytest -q                            # unit + fake end-to-end websocket test
```
Edit `prompts/*.txt`, `prompts/common.txt` (shared voice rules) and `config.json` (placeholders, greeting, fallback lines).

## Output per call (`./recordings`)
`<timestamp>_<bot>_<CallSid>.wav` (dual channel; arrives after the call), `.txt` transcript, `.json` (outcome: committed/date/amount/refused, end reason, average latency, transcript). Per-turn latency (STT, LLM first token, TTS first audio) is logged.

## Behaviour notes
- Silence: 5 s (`SILENCE_TIMEOUT_S`) → check-in line; after 3 check-ins the bot says goodbye and hangs up.
- END_CALL: the model appends `END_CALL`; it is stripped before TTS, playback finishes, then the call is hung up via the REST API.
- Webhooks verify `X-Twilio-Signature`; the media websocket requires a per-call token issued in the TwiML.
- Tests write sample files into `./recordings` (named `*CAtest123*`); delete them if you like.
