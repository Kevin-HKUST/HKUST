"""Canonical intent labels.

Single source of truth shared by IntentRecognizer (which emits these) and
WorkflowEngine (which dispatches on them). Keeping the names in one module
prevents the two sides from drifting apart -- an earlier version emitted
"general_knowledge" while the engine only ever checked for "knowledge", so
that branch was unreachable.

This module must stay dependency-free so it can be imported in tests
without pulling in the retrieval stack.
"""

WEATHER = "weather"
FINANCE = "finance"
TRANSPORT = "transport"
WEB_SEARCH = "web_search"
MATH = "math"
MULTIMODAL = "multimodal"
KNOWLEDGE = "knowledge"

#: Intents that map directly onto a registered plugin.
PLUGIN_INTENTS = (WEATHER, MATH, WEB_SEARCH, FINANCE, TRANSPORT)

#: Every label the recognizer is allowed to emit.
ALL_INTENTS = PLUGIN_INTENTS + (MULTIMODAL, KNOWLEDGE)
