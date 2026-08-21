"""Stage 2 -- LLM fill. Content and delivery; never touches a timestamp.

Talks to any OpenAI-compatible /v1/chat/completions endpoint: Ollama locally,
vLLM on a cluster. Chunked, because small local models drift on long
structured generations (schema adherence and word budgets both degrade).
"""
from __future__ import annotations

import json
import re
import urllib.request

# No style enum. CosyVoice 3's style channel is natural language
# (inference_instruct2 takes a free-text instruct_text), so the LLM writes the
# delivery per utterance -- it already has the discourse context and the event.
# `level` stays discrete because it is the GAIN channel: free text cannot set a
# dB, and the mixer needs a number for the Lombard sum. It is rendered twice --
# as a vocal-effort phrase in the instruction, and as a sampled gain.
LEVELS = ["whisper", "soft", "normal", "raised", "loud"]
MAX_INSTRUCT_WORDS = 12

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

`instruct` is how THIS line is delivered, written as a short imperative to the
voice actor -- at most 12 words. It is free text, so be specific to the moment
rather than picking from a menu:
  "Cut in fast, you have heard this argument all evening."
  "Trail off, you are not sure you believe it yourself."
  "Say it flatly, you have already moved on."
  "Warm, agreeing without really listening."
Ground it in the event and in what was just said. Do not restate the words, and
do not describe loudness here -- that is `level`.

`level` is how LOUD, judged separately from delivery: cut-ins `raised`/`loud`,
backchannels `soft`, asides `soft`/`whisper`. Do not label everything `normal`."""


def _schema() -> dict:
    return {"type": "object", "properties": {"slots": {"type": "array", "items": {
        "type": "object",
        "properties": {"uid": {"type": "integer"}, "text": {"type": "string"},
                       "instruct": {"type": "string"},
                       "level": {"type": "string", "enum": LEVELS}},
        "required": ["uid", "text", "instruct", "level"],
        "additionalProperties": False}}},
        "required": ["slots"], "additionalProperties": False}


def _call(base_url, model, messages, schema=None, timeout=900) -> dict:
    body = {"model": model, "messages": messages, "stream": False,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "fill",
                                                "schema": schema or _schema(),
                                                "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base_url}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return json.loads(raw["choices"][0]["message"]["content"])


_LABEL = re.compile(r"^\s*(spk_[A-Z]|[A-Z][a-z]+)\s*:\s*")


CAST_SYSTEM = """You set up a conversation that will then be voiced.

You are GIVEN the situation -- do not invent a different one, and do not
sanitise it into something more ordinary. Your job is to make it concrete and
to populate it with specific people.

Return:
  `premise`  -- one sentence, present tense, what is happening between these
                people right now. Concrete, not a topic label. Not a summary of
                what they will say.
  `personas` -- one entry per speaker id given to you, using the SEX given for
                that id (their voice is already cast; names and pronouns must
                match it or the transcript contradicts the audio):
       name    -- a first name people would actually use for each other. Draw
                  from varied backgrounds, not one narrow set.
       manner  -- how this person talks. Half a line. Specific:
                  "long pauses, then says the blunt thing"
                  "over-explains when nervous, which is now"
                  "agrees out loud while disagreeing"
       about   -- one line: who they are HERE and what they want out of this.
                  Give people incompatible wants -- that is what makes them
                  interrupt each other. Free text: it may be a job title, a
                  relationship, a grievance, whatever the situation calls for.

Different people, not variations on one voice."""


def _cast_schema(speakers):
    return {"type": "object", "properties": {
        "premise": {"type": "string"},
        "personas": {"type": "object", "properties": {
            s: {"type": "object",
                "properties": {"name": {"type": "string"},
                               "manner": {"type": "string"},
                               "about": {"type": "string"}},
                "required": ["name", "manner", "about"],
                "additionalProperties": False} for s in speakers},
            "required": list(speakers), "additionalProperties": False}},
        "required": ["premise", "personas"], "additionalProperties": False}


def cast(convo_type, scenario, sexes, base_url="http://127.0.0.1:11434/v1",
         model="qwen3:30b-a3b", max_retries=2):
    """Turn a sampled scenario into a premise and one persona per speaker.

    The scenario is drawn in code (see scenarios.py) precisely so the model does
    not choose it; here it only makes it concrete.
    """
    spec = "\n".join(f"  {k}: {v}" for k, v in scenario.items())
    who = ", ".join(f"{s} is {'male' if x == 'M' else 'female'}"
                    for s, x in sexes.items())
    user = (f"Conversation type: {convo_type}\n\nSituation:\n{spec}\n\n"
            f"Speakers: {who}")
    for attempt in range(max_retries + 1):
        try:
            res = _call(base_url, model,
                        [{"role": "system", "content": CAST_SYSTEM},
                         {"role": "user", "content": user}],
                        schema=_cast_schema(list(sexes)))
        except Exception as exc:
            print(f"  cast: call failed ({exc}); retrying")
            continue
        if res.get("premise") and all(k in res.get("personas", {}) for k in sexes):
            return res
        print(f"  cast: incomplete, retry {attempt + 1}/{max_retries}")
    raise RuntimeError("cast failed: model did not return premise + all personas")


def _brief(slots):
    out = []
    for s in slots:
        tw = s["target_words"]
        lo, hi = (1, 3) if s["event"] == "backchannel" else \
                 (max(1, round(tw * 0.7)), round(tw * 1.3))
        out.append({"uid": s["uid"], "speaker": s["speaker"],
                    "event": s["event"], "words": f"{lo}-{hi}"})
    return out


def fill(skeleton: dict, personas: dict, premise: str = "",
         base_url="http://127.0.0.1:11434/v1",
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
            ctx = ("So far (do not repeat these lines):\n"
                   + "\n".join(said[-10:]) + "\n\n") if said else ""
            head = f"What is happening: {premise}\n\n" if premise else ""
            user = (f"{head}People:\n{json.dumps(personas, indent=1)}\n\n{ctx}"
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
                instruct = " ".join((g.get("instruct") or "").split())
                if not instruct:                     # the delivery channel, not optional
                    continue
                instruct = " ".join(instruct.split()[:MAX_INSTRUCT_WORDS])
                got[uid] = {"text": text, "instruct": instruct, "level": g["level"]}

        for s in part:                               # last resort
            if s["uid"] not in got:
                stubbed.append(s["uid"])
                got[s["uid"]] = {"text": "mm-hm", "level": "soft",
                                 "instruct": "Murmur a short quiet agreement."}
        for s in part:
            out[s["uid"]] = got[s["uid"]]
            said.append(f'{s["speaker"]}: {got[s["uid"]]["text"]}')
        if verbose:
            print(f"  fill: {min(i + chunk, len(slots))}/{len(slots)} slots")

    if stubbed:
        print(f"  fill: WARNING {len(stubbed)} slot(s) stubbed after "
              f"{max_retries} retries: {stubbed}")
    return out
