"""Minimal terminal client for the running API.

Usage: python scripts/chat_cli.py [--url http://localhost:8080] [--trace]
"""

import argparse
import json

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--trace", action="store_true", help="print route, moderation and agent steps")
    args = ap.parse_args()

    session_id = None
    print("CBT anxiety support (Ctrl+C to quit). Not a substitute for professional care.\n")
    with httpx.Client(timeout=120) as client:
        while True:
            try:
                msg = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not msg:
                continue
            r = client.post(f"{args.url}/v1/chat", json={"message": msg, "session_id": session_id})
            r.raise_for_status()
            body = r.json()
            session_id = body["session_id"]
            print(f"\ncounselor> {body['reply']}\n")
            if args.trace:
                trace = {k: body[k] for k in ("route", "latency_ms", "input_moderation", "output_moderation", "agent")}
                print(json.dumps(trace, indent=2), "\n")


if __name__ == "__main__":
    main()
