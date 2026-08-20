"""Stage 2 -- LLM fill. Content and delivery; never touches a timestamp.

Talks to any OpenAI-compatible /v1/chat/completions endpoint: Ollama locally,
vLLM on a cluster. Chunked, because small local models drift on long
structured generations (schema adherence and word budgets both degrade).
"""
from __future__ import annotations

import json
import re
import urllib.request

STYLES = ["neutral", "engaged", "urgent", "hesitant", "insistent",
          "low_arousal", "affirmative", "mildly_annoyed", "patient"]
LEVELS = ["whisper", "soft", "normal", "raised", "loud"]

SYSTEM = """You write the words for a simulated multi-party meeting.

The TIMING is already fixed and is not yours to change. For each slot you get its
event type, who holds the floor, and a word budget. Write what is said.

`text` contains ONLY the spoken words -- never a speaker name, label, or stage
direction.

Register is spontaneous SPOKEN English, not prose:
  "yeah -- no, I mean, the thing is we never actually tested that bit, did we"
  "sorry, can I just -- was that the same run or a different one"
  "it's, uh... it's more that nobody owns it. that's the actual problem."
  "mm. hm."
Fillers, false starts, self-repairs, trailing off, contractions. People do not
speak in tidy sentences. Do NOT clip words out to hit a number -- write it the
way someone would say it, then check it lands in the range.

Rules:
  - Stay inside the word range given for each slot.
  - Never repeat a line you already wrote.
  - interrupt: the interrupter's FULL cut-in, mid-thought, no wind-up.
  - backchannel: 1-3 words, a listener token ("mm-hm", "right", "yeah no").
  - hold: the SAME speaker continuing after their own pause.
  - It is one coherent conversation.

`style` is the delivery: pick from the vocabulary and VARY it with the event and
what is being said -- a cut-in is rarely `neutral`, a backchannel is often
`affirmative` or `low_arousal`, a long explanation can be `patient`.
`level` is how LOUD, judged separately from style: cut-ins `raised`/`loud`,
backchannels `soft`, asides `soft`/`whisper`. Do not label everything `normal`."""


def _schema() -> dict:
    return {"type": "object", "properties": {"slots": {"type": "array", "items": {
        "type": "object",
        "properties": {"uid": {"type": "integer"}, "text": {"type": "string"},
                       "style": {"type": "string", "enum": STYLES},
                       "level": {"type": "string", "enum": LEVELS}},
        "required": ["uid", "text", "style", "level"],
        "additionalProperties": False}}},
        "required": ["slots"], "additionalProperties": False}


def _call(base_url, model, messages, timeout=900) -> dict:
    body = {"model": model, "messages": messages, "stream": False,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "fill", "schema": _schema(),
                                                "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base_url}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return json.loads(raw["choices"][0]["message"]["content"])


_LABEL = re.compile(r"^\s*(spk_[A-Z]|[A-Z][a-z]+)\s*:\s*")


def _brief(slots):
    out = []
    for s in slots:
        tw = s["target_words"]
        lo, hi = (1, 3) if s["event"] == "backchannel" else \
                 (max(1, round(tw * 0.7)), round(tw * 1.3))
        out.append({"uid": s["uid"], "speaker": s["speaker"],
                    "event": s["event"], "words": f"{lo}-{hi}"})
    return out


def fill(skeleton: dict, personas: dict, base_url="http://127.0.0.1:11434/v1",
         model="qwen3:30b-a3b", chunk=12, max_retries=3, verbose=True) -> dict:
    """Return {uid: {text, style, level}} for every slot in the skeleton.

    The JSON schema cannot express array length (no minItems/maxItems), so a
    short return is caught here and RETRIED for the missing slots only. Stubbing
    is the last resort, not the first response -- a stubbed slot is a silent
    quality hole in the corpus.
    """
    slots = skeleton["slots"]
    out: dict[int, dict] = {}
    said: list[str] = []
    stubbed: list[int] = []

    for i in range(0, len(slots), chunk):
        part = slots[i:i + chunk]
        want = {s["uid"] for s in part}
        got: dict[int, dict] = {}

        for attempt in range(max_retries + 1):
            todo = [s for s in part if s["uid"] not in got]
            if not todo:
                break
            if attempt:
                print(f"  fill: retry {attempt}/{max_retries} for "
                      f"{len(todo)} slot(s): {[s['uid'] for s in todo]}")
            ctx = ("Story so far (do not repeat these lines):\n"
                   + "\n".join(said[-10:]) + "\n\n") if said else ""
            user = (f"Speakers: {json.dumps(personas)}\n\n{ctx}"
                    f"Slots to fill (return EXACTLY {len(todo)} entries, one per "
                    f"uid listed):\n{json.dumps(_brief(todo), indent=1)}")
            try:
                res = _call(base_url, model,
                            [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}])
            except Exception as exc:                 # transient endpoint failure
                print(f"  fill: call failed ({exc}); retrying")
                continue
            for g in res.get("slots", []):
                uid = g.get("uid")
                if uid not in want or uid in got:    # drop hallucinated/dup uids
                    continue
                text = _LABEL.sub("", (g.get("text") or "")).strip()
                if not text:                         # empty is as bad as missing
                    continue
                got[uid] = {"text": text, "style": g["style"], "level": g["level"]}

        for s in part:                               # last resort
            if s["uid"] not in got:
                stubbed.append(s["uid"])
                got[s["uid"]] = {"text": "mm-hm", "style": "affirmative",
                                 "level": "soft"}
        for s in part:
            out[s["uid"]] = got[s["uid"]]
            said.append(f'{s["speaker"]}: {got[s["uid"]]["text"]}')
        if verbose:
            print(f"  fill: {min(i + chunk, len(slots))}/{len(slots)} slots")

    if stubbed:
        print(f"  fill: WARNING {len(stubbed)} slot(s) stubbed after "
              f"{max_retries} retries: {stubbed}")
    return out
