"""
src/recommendations.py
=======================
Returns structured, class-specific recommendations for the Streamlit app.

Each entry contains:
    emoji     : display icon
    headline  : one-line summary of the problem
    why       : brief explanation of why this factor causes procrastination
    tips      : list of 5 actionable tips (markdown-formatted)
    resources : list of technique / book names for further reading

Public API
----------
get_recommendation(predicted_class: str) -> dict
"""

from __future__ import annotations  # PEP 563 — defers annotation evaluation (Python 3.9 compat)

# ── Recommendation table ───────────────────────────────────────────────────────
# Keys must exactly match the CLASS_LABELS defined in src/preprocessing.py.

_RECOMMENDATIONS: dict[str, dict] = {

    "Digital Distraction": {
        "emoji": "📱",
        "headline": "Your phone and social media are fragmenting your attention.",
        "why": (
            "High phone use and social media time constantly interrupt your "
            "working memory. Every notification forces a costly context-switch, "
            "making it hard to re-enter the focused state needed for real work."
        ),
        "tips": [
            "📵 Use an **app blocker** (Freedom, Cold Turkey) during work sessions.",
            "⏱️ Try the **Pomodoro Technique**: 25 min deep work → 5 min phone break.",
            "📍 Put your phone in a **different room** while working.",
            "🔕 Disable all non-essential notifications at the OS level.",
            "📅 Schedule **intentional social media slots** (e.g. 12:00–12:15 only).",
        ],
        "resources": ["Pomodoro Technique", "Digital Minimalism (Cal Newport)", "Freedom App"],
    },

    "Low Motivation": {
        "emoji": "🪫",
        "headline": "You're struggling to find a reason to start.",
        "why": (
            "Low task interest combined with low energy creates a motivation "
            "deficit. Without intrinsic or extrinsic pull, the brain defaults to "
            "avoidance — procrastination is the path of least resistance."
        ),
        "tips": [
            "🎯 Find your **'why'**: link the task to a personal goal or value.",
            "🏆 Use **temptation bundling**: pair the boring task with something enjoyable.",
            "🔋 **Move first** — a 10-min walk raises dopamine and energy before work.",
            "🌱 Break it into **micro-goals** (e.g. 'write just one sentence').",
            "📣 Use an **accountability partner**: tell someone your specific goal today.",
        ],
        "resources": ["Atomic Habits (James Clear)", "Temptation Bundling", "Self-Determination Theory"],
    },

    "Task Difficulty": {
        "emoji": "🧗",
        "headline": "The task feels too hard to know where to begin.",
        "why": (
            "When a task exceeds your perceived skill level, stress rises and "
            "confidence drops. The brain treats this as a threat and triggers "
            "avoidance as a self-protective response."
        ),
        "tips": [
            "🔍 **Decompose** the task: identify the single smallest concrete first step.",
            "📖 Spend 15 min **researching** before attempting — reduces uncertainty.",
            "🤝 **Ask for help**: reach out to a classmate, TA, or online community.",
            "💡 Use the **'Swiss Cheese' method**: poke small holes in the task until momentum builds.",
            "🧠 Review **worked examples** in the same domain to build mental schemas.",
        ],
        "resources": ["Task Decomposition", "Worked Example Effect", "Khan Academy / YouTube"],
    },

    "Fear of Failure": {
        "emoji": "😰",
        "headline": "Fear of getting it wrong is keeping you frozen.",
        "why": (
            "Very low confidence combined with high stress creates fear of failure. "
            "The brain treats potential failure as a social or ego threat, so "
            "avoidance feels safer than risking a bad outcome."
        ),
        "tips": [
            "✏️ Reframe: **'done is better than perfect'** — a draft beats a blank page.",
            "🧪 Adopt a **growth mindset**: treat the task as an experiment, not a test.",
            "📝 Write down your **worst-case scenario** and the realistic one — usually far less scary.",
            "🤍 Practice **self-compassion**: treat yourself as you would a struggling friend.",
            "🚀 **Start before you feel ready** — confidence follows action, not the reverse.",
        ],
        "resources": ["Growth Mindset (Carol Dweck)", "The Now Habit (Neil Fiore)", "CBT Techniques"],
    },

    "Perfectionism": {
        "emoji": "🎯",
        "headline": "You're waiting for ideal conditions that may never arrive.",
        "why": (
            "Extremely high perfectionism drives you to delay because the output "
            "might not meet impossibly high internal standards. Perfectionism "
            "mistakes inaction for safety."
        ),
        "tips": [
            "📐 Set a **'good enough' threshold** before starting — define what 80% looks like.",
            "⏱️ Use **time-boxing**: give yourself exactly 2 hours, then submit what you have.",
            "🔄 Embrace **iterative output**: version 1 can always be refined later.",
            "🪞 Ask: *'Would I judge a friend this harshly for the same work?'*",
            "📊 Track how often 'good enough' work produces the same outcome as 'perfect' work.",
        ],
        "resources": ["CBT for Perfectionism", "Mindfulness-Based Stress Reduction", "The Perfectionism Trap"],
    },

    "Poor Sleep / Low Energy": {
        "emoji": "😴",
        "headline": "Your brain doesn't have the fuel it needs to focus.",
        "why": (
            "Very low sleep hours and energy directly impair executive function, "
            "working memory, and willpower. Even simple tasks feel overwhelming "
            "when the prefrontal cortex is under-resourced."
        ),
        "tips": [
            "🛏️ **Prioritise sleep**: aim for 7–9 hours — no intervention improves cognition more.",
            "☀️ Get **sunlight within 30 min of waking** to anchor your circadian rhythm.",
            "☕ Use **caffeine strategically**: 90 min after waking; stop by 2 pm.",
            "🧘 A **10-min nap** (before 3 pm) can meaningfully restore alertness.",
            "🚶 **Light exercise** — even a short walk raises cerebral blood flow and focus.",
        ],
        "resources": ["Why We Sleep (Matthew Walker)", "Huberman Lab Sleep Protocols", "Sleep Hygiene Guidelines"],
    },

    "Lack of Clarity": {
        "emoji": "🌫️",
        "headline": "You don't know exactly what to do or where to start.",
        "why": (
            "Very low task clarity creates decision paralysis. If the brain can't "
            "picture a clear next action, avoidance is the easiest response — "
            "you can't act on an instruction you haven't understood."
        ),
        "tips": [
            "📋 Write down the **single next physical action** (e.g. 'open Word and type the title').",
            "❓ Ask clarifying questions: *'What does done look like?'*",
            "🗺️ Create a **mind map or outline** first — 10 min of planning saves hours.",
            "📌 Use **implementation intentions**: 'If I sit at my desk at 9 am, I will open my notes.'",
            "🔢 Number your subtasks and tackle them **strictly one at a time**.",
        ],
        "resources": ["Getting Things Done (David Allen)", "Implementation Intentions", "Mind Mapping"],
    },

    "Poor Time Management": {
        "emoji": "⏰",
        "headline": "Time is slipping away and priorities feel unclear.",
        "why": (
            "High deadline pressure combined with a habitual history of procrastination "
            "suggests a planning deficit. Tasks grow into crises, reinforcing the cycle "
            "of last-minute stress → poor quality → regret → avoidance."
        ),
        "tips": [
            "📅 Use **calendar time-blocking**: schedule specific tasks into named time slots.",
            "🥅 Apply the **Eisenhower Matrix**: separate urgent+important from everything else.",
            "⏳ Practice **time estimation**: log how long tasks really take vs. how long you guessed.",
            "🔁 **Daily priority review** each morning — a 5-min habit that compounds fast.",
            "🚫 **Say no** deliberately: protect high-priority time from low-value requests.",
        ],
        "resources": ["Getting Things Done (GTD)", "Eisenhower Matrix", "Time Blocking Method"],
    },
}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_recommendation(predicted_class: str) -> dict:
    """
    Return the recommendation dict for a given predicted class label.

    Parameters
    ----------
    predicted_class : str
        One of the eight class labels produced by the model.

    Returns
    -------
    dict with keys: ``emoji``, ``headline``, ``why``, ``tips``, ``resources``.
    Returns a safe fallback dict if the class is unrecognised.
    """
    return _RECOMMENDATIONS.get(predicted_class, {
        "emoji":     "🤔",
        "headline":  "No specific recommendation available.",
        "why":       "",
        "tips":      [],
        "resources": [],
    })


def all_classes() -> list[str]:
    """Return the list of all supported class names."""
    return list(_RECOMMENDATIONS.keys())
