"""Read a transient API key without echo and verify access to a pinned judge model."""
from __future__ import annotations

import getpass
import json
import urllib.error
import urllib.request
import warnings

MODEL = "gpt-4o-mini-2024-07-18"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> None:
    print("READY_FOR_OPENAI_KEY", flush=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("OpenAI key: ")
    except getpass.GetPassWarning:
        print(json.dumps({"error": "SecureTerminalRequired"}), flush=True)
        raise SystemExit(2) from None
    request = urllib.request.Request(
        "https://api.openai.com/v1/models/" + MODEL,
        headers={"Authorization": "Bearer " + api_key},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            payload = json.load(response)
            if payload.get("id") != MODEL:
                raise ValueError("Unexpected model response")
            print(json.dumps({"status": response.status, "model": MODEL}), flush=True)
    except urllib.error.HTTPError as exc:
        # Error bodies can echo part of the credential; do not print them.
        print(json.dumps({"status": exc.code, "error": "HTTPError"}), flush=True)
        raise SystemExit(1) from None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    finally:
        api_key = ""


if __name__ == "__main__":
    main()
