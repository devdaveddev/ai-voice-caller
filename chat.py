"""Terminal test mode: tune prompts without placing calls.

  python chat.py --bot loan            # real Anthropic model (needs ANTHROPIC_API_KEY)
  python chat.py --bot loan --mock     # offline canned replies, plumbing only
Type as the customer; blank line = silence; /quit exits.
"""
import argparse
import asyncio
import os
import time

from app.config import BotConfig


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True, choices=["loan", "insurance"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"
    from app.llm import make_llm
    from app.session import Conversation

    bot = BotConfig(args.bot, dict(v.split("=", 1) for v in args.var if "=" in v))
    convo = Conversation(bot, make_llm())
    print(f"--- {args.bot} bot | each line below is one TTS sentence ---")
    print(f"BOT: {bot.greeting}")
    convo.add_assistant(bot.greeting)
    loop = asyncio.get_running_loop()
    while not convo.end_call:
        user = (await loop.run_in_executor(None, input, "YOU: ")).strip()
        if user == "/quit":
            break
        user = user or "(customer is silent)"
        timing, t0 = {}, time.monotonic()
        async for sentence in convo.reply(user, timing):
            print(f"BOT [{(time.monotonic() - t0) * 1000:.0f}ms]: {sentence}")
        print(f"     (first token {timing.get('llm_first_token', 0) * 1000:.0f}ms, "
              f"first sentence {timing.get('llm_first_sentence', 0) * 1000:.0f}ms)")
    if convo.end_call:
        print("--- END_CALL emitted: call would hang up here ---")


if __name__ == "__main__":
    asyncio.run(main())
