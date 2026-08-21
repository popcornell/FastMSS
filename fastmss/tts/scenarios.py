"""Scenario sampling -- diversity comes from code, never from the LLM.

An LLM asked to invent a premise collapses onto its priors (every meeting
becomes a Q3 roadmap). So the premise is DRAWN here and handed over; the model
only composes it into people and speech.

An axis is one independent question about the conversation with a short list of
answers. Axes are orthogonal, so the space is their PRODUCT: five lists of ~20
is 3.2M combinations from 100 curated strings. Each ConvoType declares its own
axes, because the questions that matter differ by genre -- "presenting
complaint" is meaningless between siblings.

For unstructured talk the axes describe the SITUATION, never the subject. Nobody
sets "topic = cars"; the subject emerges from who is talking and why, which is
also what makes it drift and double back the way real conversation does.
"""
from __future__ import annotations

import hashlib

CONVO_TYPES: dict[str, dict[str, list[str]]] = {
    # ---------------------------------------------------------------- casual
    "random_convo": {
        "relationship": [
            "two siblings", "old friends drifting apart", "new neighbours",
            "ex-colleagues who genuinely liked each other", "a couple a few months in",
            "flatmates", "cousins who meet once a year", "a parent and their adult child",
            "two people who met at the thing they're both leaving",
            "someone and their oldest friend's new partner",
            "a long-married couple", "two people who used to be closer",
            "an aunt and a nephew", "former bandmates", "in-laws who get on",
            "in-laws who don't", "a mentor and the person they mentored",
            "two people stuck waiting for a third", "estranged half-siblings",
            "a landlord and a tenant who have become friendly",
        ],
        "setting": [
            "a kitchen table", "a long car ride", "a hospital waiting room",
            "a pub after closing", "a parked car outside the house",
            "a shared laundry room", "a station platform", "someone's back garden",
            "a half-packed flat", "the queue for something slow",
            "a hotel breakfast room", "a hospital car park",
            "a kitchen at someone else's party", "a bus replacement service",
            "a walk neither of them chose", "a rooftop in the cold",
            "a waiting room before an appointment", "an airport at an unreasonable hour",
            "a garden centre cafe", "the last table in a closing restaurant",
        ],
        "pretext": [
            "one of them has news they haven't said yet",
            "a shared plan is quietly falling apart",
            "one borrowed something and didn't ask",
            "they're both avoiding the thing they came to discuss",
            "one wants a favour and is circling it",
            "one of them has clearly been crying",
            "a third person's name keeps coming up",
            "one of them is leaving and it hasn't been said out loud",
            "they disagree about something that happened years ago",
            "one has made a decision the other will hate",
            "they are killing time and it turns into something",
            "an old promise is being quietly not-kept",
            "one is trying to apologise without saying sorry",
            "money is involved and neither wants to raise it",
            "one of them is much drunker than the other",
            "a misunderstanding neither has noticed yet",
            "they are pretending a thing is fine",
            "one is fishing for information",
            "a plan has to be made and nobody wants to decide",
            "one keeps checking their phone",
        ],
        "temperature": [
            "warm", "prickly", "exhausted", "giddy", "carefully polite",
            "somewhere past caring", "wary", "affectionate and blunt",
            "tense under a joke", "distracted", "raw", "conspiratorial",
            "brittle", "companionable silence, then not", "impatient",
            "unexpectedly tender", "needling", "flat", "restless", "relieved",
        ],
        "history": [
            "strangers an hour ago", "a few months", "five years", "thirty years",
            "a decade with a gap in the middle", "since school",
            "one summer, long ago", "through someone else entirely",
            "professionally, until recently", "two weeks and moving fast",
        ],
    },
    # ------------------------------------------------------------- structured
    "customer_support": {
        "product": [
            "home broadband", "a flight booking", "a bank card", "a storage unit",
            "a prescription delivery", "a car insurance policy", "a food order",
            "a mobile contract", "a returned parcel", "a gym membership",
            "a utility account", "a concert ticket resale", "a rented van",
            "a software subscription", "a repaired appliance",
        ],
        "failure": [
            "charged twice", "never arrived", "works intermittently",
            "cancelled without notice", "the wrong item entirely",
            "a refund that was promised and never came", "renewed after cancelling",
            "an account locked for no stated reason", "a price that changed after booking",
            "delivered to an address they've never lived at",
            "downgraded without being told", "double-booked",
        ],
        "prior_contacts": [
            "first contact", "third call this week", "escalated twice already",
            "chat transcript they were told would be read", "called after emailing for a month",
            "a callback that never came",
        ],
        "obstacle": [
            "policy needs a reference number they don't have",
            "the refund window closed yesterday",
            "the account is in someone else's name",
            "the agent can see the problem but can't fix it",
            "the system says it was resolved",
            "it needs a department that closed an hour ago",
            "the fix requires them to cancel and re-buy at a higher price",
            "two parts of the company disagree about what happened",
        ],
    },
    "doctor_patient": {
        "complaint": [
            "a cough that won't clear", "numbness in one hand",
            "dizziness when standing", "sleeping badly for months",
            "pain that moves around", "losing weight without trying",
            "headaches at the same time each day", "a lump they found",
            "breathlessness on stairs", "ringing in one ear",
            "stomach trouble after eating", "a rash that comes and goes",
        ],
        "duration": [
            "three days", "six weeks", "on and off for a year",
            "since a specific event they mention", "as long as they can remember",
            "it started again after stopping",
        ],
        "fear": [
            "they think it's nothing", "they're afraid it's serious",
            "a relative had something similar", "they've been reading online",
            "they're worried about missing work", "they want it to be physical, not stress",
            "they've had it dismissed before",
        ],
        "context": [
            "first visit", "a follow-up where nothing has improved",
            "they came about something else", "a test result is due",
            "someone made them come", "they're between doctors",
        ],
    },
    "workplace_meeting": {
        "domain": [
            "logistics", "a clinical trial", "a game studio", "civil engineering",
            "a school", "a charity", "hardware manufacturing", "a newsroom",
            "hospital scheduling", "a small architecture practice",
            "municipal planning", "a research lab",
        ],
        "occasion": [
            "an incident post-mortem", "budget triage", "a design review",
            "a hiring debrief", "a go/no-go before launch",
            "a handover nobody prepared for", "a vendor review",
            "planning after someone resigned", "a deadline being renegotiated",
        ],
        "at_stake": [
            "a slipped deadline", "a number nobody trusts",
            "two incompatible proposals", "a decision already made elsewhere",
            "work that has to be thrown away", "who owns a failure",
            "a request that can't be met with current headcount",
            "a commitment made to a customer without asking",
        ],
        "friction": [
            "one person has the data and won't share it yet",
            "the most senior person is wrong and nobody has said so",
            "two of them agreed something beforehand",
            "one of them is new and asking obvious good questions",
            "everyone is tired of this topic",
            "the real decision is not the one on the agenda",
        ],
    },
}


def sample_scenario(convo_type: str, seed: int) -> dict[str, str]:
    """One independent stream per axis.

    A single shared RNG (`rng = Random(seed); rng.choice(...)` per axis) makes
    ADJACENT seeds correlate -- meetings 0,1,2 come out with the same value on
    the later axes. Since meetings are generated 0..N in order that produces
    visible runs of near-identical setups. Hashing the axis name into the seed
    decorrelates them.
    """
    axes = CONVO_TYPES[convo_type]
    out = {}
    for name, values in axes.items():
        h = hashlib.sha256(f"{convo_type}|{name}|{seed}".encode()).digest()
        out[name] = values[int.from_bytes(h[:8], "big") % len(values)]
    return out


# Only random_convo is drawn by default. The structured types below are kept
# because they are written and cheap to keep, but they have not been checked for
# stage directions, ordinary stakes, or speaker-id leaks the way random_convo
# has -- and doctor_patient in particular needs its own judgement about what
# counts as melodrama, since "afraid it's serious" is an ordinary thing for a
# patient to be. Add weight here once a type has been looked at.
DEFAULT_WEIGHTS = {"random_convo": 1.0}


def pick_type(seed: int, weights: dict[str, float] | None = None) -> str:
    """Weighted draw over ConvoTypes, on its own stream."""
    weights = weights or DEFAULT_WEIGHTS
    names = sorted(weights)
    total = sum(weights[n] for n in names)
    h = hashlib.sha256(f"convotype|{seed}".encode()).digest()
    x = (int.from_bytes(h[:8], "big") / 2 ** 64) * total
    acc = 0.0
    for n in names:
        acc += weights[n]
        if x < acc:
            return n
    return names[-1]


def space_size(convo_type: str) -> int:
    n = 1
    for v in CONVO_TYPES[convo_type].values():
        n *= len(v)
    return n
