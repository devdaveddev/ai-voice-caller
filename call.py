"""Place an outbound call:  python call.py --to +91XXXXXXXXXX --bot loan [--force] [--var key=value ...]"""
import argparse
import asyncio
import re
import sys

from app import dialer
from app.config import BotConfig, env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--to", required=True, help="E.164 number, e.g. +919876543210")
    ap.add_argument("--bot", required=True, choices=["loan", "insurance"])
    ap.add_argument("--force", action="store_true", help="dial outside 08:00-19:00 IST (logged)")
    ap.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config.json placeholder for this call, e.g. --var customer_name=Rahul")
    args = ap.parse_args()

    if not re.fullmatch(r"\+\d{8,15}", args.to):
        print("--to must be E.164, e.g. +919876543210", file=sys.stderr)
        return 2
    missing = [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "PUBLIC_BASE_URL") if not env(k)]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}", file=sys.stderr)
        return 2
    overrides = dict(v.split("=", 1) for v in args.var if "=" in v)
    BotConfig(args.bot, overrides)  # fail fast on bad bot/config

    ok, decision = dialer.compliance_check(args.force, args.to, args.bot)
    if not ok:
        print("Refusing to dial: outside 08:00-19:00 IST. Re-run with --force to override (will be logged).")
        return 1
    if decision == "FORCED_OUTSIDE_WINDOW":
        print("WARNING: dialing outside permitted hours because of --force (logged).")

    res = asyncio.run(dialer.place_call(args.to, args.bot, {f"var_{k}": v for k, v in overrides.items()}))
    print(f"Call placed: sid={res['sid']} status={res['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
