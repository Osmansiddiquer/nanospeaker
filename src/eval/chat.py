"""
Talk to nanoSpeaker interactively, or serve it over HTTP.

A REPL around the model's own KV-cache decode loop, streaming text as tokens are
sampled. Each prompt is independent -- 1024 trained context, no multi-turn state.

    python -m src.eval.chat                        # REPL against model.pt
    python -m src.eval.chat --instruct             # wrap prompts in the SFT template
    python -m src.eval.chat --serve 8765           # POST /generate {"prompt": ...}
"""

import argparse
import contextlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tokenizers import Tokenizer

from .decode import INSTRUCT, load, pick_device, stream

STOP = ("### Instruction",)   # the SFT format's next-example boundary
CHAT_STOP_IDS = frozenset({0, 3})   # <|endoftext|>, <|im_end|>


def render_chat(messages) -> str:
    """ChatML template; ends open for the assistant to complete.

    Leads with <|endoftext|>: SFT rows are packed, so in training a conversation
    start is (nearly) always preceded by the document separator -- a bare
    position-0 <|im_start|> is out-of-distribution and measurably collapses the
    model (P(<|think|>) 0.55 packed vs 0.0000 cold at sft step 950)."""
    return "<|endoftext|>" + "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                                     for m in messages) + "<|im_start|>assistant\n"


@contextlib.contextmanager
def _quiet_keyboard():
    """While a response streams, typed characters would echo into the output
    line. Silence the echo and drop type-ahead; restore on exit. No-op when
    stdin is not a tty (pipes)."""
    try:
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, new)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSANOW, old)


def chat_stream(model, tok, device, messages, n=200, **kw):
    """Stream a chat completion with the template applied correctly.

    `messages` is a list of {'role','content'} dicts, or a plain string (treated
    as a single user turn). Use THIS in notebooks -- calling decode.stream() on
    raw text skips the eos prefix and turn structure, and a chat model answers
    from the wrong distribution entirely.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    kw.setdefault("stop_ids", CHAT_STOP_IDS)
    kw.setdefault("min_tokens", 8)
    yield from stream(model, tok, device, render_chat(messages), n, **kw)


def repl(model, tok, device, args) -> None:
    mode = "chat" if args.chat else "instruct" if args.instruct else "completion"
    print(f"{mode} mode | temperature {args.temperature}, top_k {args.top_k}, "
          f"top_p {args.top_p}, rep_penalty {args.rep_penalty}, {args.tokens} tokens max "
          f"| '\\n' in a prompt becomes a newline | ctrl+d quits"
          + (" | /clear resets history" if args.chat else ""))
    history = [{"role": "system", "content": args.system}] if args.system else []
    while True:
        try:
            prompt = input("\n> ")
        except EOFError:
            return
        if not prompt.strip():
            continue
        if args.chat and prompt.strip() == "/clear":
            history = history[:1] if args.system else []
            print("[history cleared]")
            continue
        prompt = prompt.replace("\\n", "\n")
        if args.chat:
            history.append({"role": "user", "content": prompt})
            rendered, stop, stop_ids = render_chat(history), (), CHAT_STOP_IDS
        elif args.instruct:
            rendered, stop, stop_ids = INSTRUCT.format(prompt), STOP, frozenset()
        else:
            rendered, stop, stop_ids = prompt, (), frozenset()
        t0, count, reply = time.perf_counter(), 0, []
        try:
            with _quiet_keyboard():
                for delta in stream(model, tok, device, rendered, args.tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    top_p=args.top_p, rep_penalty=args.rep_penalty,
                                    stop=stop, stop_ids=stop_ids,
                                    min_tokens=8 if args.chat else 0):
                    print(delta, end="", flush=True)
                    reply.append(delta)
                    count += 1
        except KeyboardInterrupt:
            print("  [aborted]")
            if args.chat:
                history.pop()
            continue
        if args.chat:
            # History must hold canonical special-token text: the display markers
            # ("〔think: ") would re-encode as ordinary tokens and corrupt every
            # later turn's context (measured: think-probability collapses).
            canonical = ("".join(reply).replace("〔think: ", "<|think|>")
                         .replace("〕 ", "<|/think|>").strip())
            history.append({"role": "assistant", "content": canonical})
        print(f"\n[{count / max(time.perf_counter() - t0, 1e-9):.1f} tok/s]")


def serve(model, tok, device, args) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            stop, stop_ids = (), frozenset()
            if "messages" in req:
                prompt, stop_ids = render_chat(req["messages"]), CHAT_STOP_IDS
            elif req.get("instruct", args.instruct):
                prompt, stop = INSTRUCT.format(req["prompt"]), STOP
            else:
                prompt = req["prompt"]
            t0 = time.perf_counter()
            parts = list(stream(model, tok, device, prompt,
                                int(req.get("tokens", args.tokens)),
                                temperature=float(req.get("temperature", args.temperature)),
                                top_k=int(req.get("top_k", args.top_k)),
                                top_p=float(req.get("top_p", args.top_p)),
                                rep_penalty=float(req.get("rep_penalty", args.rep_penalty)),
                                stop=stop, stop_ids=stop_ids,
                                min_tokens=8 if stop_ids else 0))
            dt = time.perf_counter() - t0
            body = json.dumps({"text": "".join(parts),
                               "tok_per_s": round(len(parts) / max(dt, 1e-9), 1)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):
            pass

    print(f"listening on 127.0.0.1:{args.serve} -- POST /generate "
          '{"prompt": ..., "tokens"?, "temperature"?, "top_k"?, "top_p"?, '
          '"rep_penalty"?, "instruct"?}')
    HTTPServer(("127.0.0.1", args.serve), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/nanospeaker/model.pt")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0, help="1.0 disables")
    ap.add_argument("--rep-penalty", type=float, default=1.05,
                    help="CTRL-style; 1.0 disables")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--instruct", action="store_true")
    ap.add_argument("--chat", action="store_true",
                    help="ChatML multi-turn mode (SFT-era models)")
    ap.add_argument("--system", default=None, help="system prompt (chat mode)")
    ap.add_argument("--serve", type=int, default=None, metavar="PORT")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ck = load(Path(args.weights), device)
    # tokenizer_chat.json differs only in the four renamed specials; ordinary text
    # encodes identically, so prefer it whenever it exists.
    tok_file = Path("tokenizer/tokenizer_chat.json")
    tok = Tokenizer.from_file(str(tok_file if tok_file.exists()
                                  else "tokenizer/tokenizer.json"))
    print(f"nanoSpeaker step {ck['step']:,}  "
          f"{ck.get('tokens_seen', 0) / 1e9:.3f}B tokens  on {device}")
    if args.serve:
        serve(model, tok, device, args)
    else:
        repl(model, tok, device, args)


if __name__ == "__main__":
    main()
