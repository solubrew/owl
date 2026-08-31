# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""
---
<(META)>:
    docid:
    name: Owl Decision Engine (DEPRECATED)
    description: >
        Decision logic lives in MindOT.
        Use: pip install owl[decisions]  (installs mindot as a dependency)
        Then:
            from mindot import MindOT
            from mindot.mindot_owl import OwlMethod

            mt = MindOT()
            mt.add_method(OwlMethod(name="owl"))
            decision = mt.decide({"options": [...]})
    version: 0.0.0.0.0.0
    authority: document|this
    security: sec|lvl2
    <(WT)>: -32
"""
# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings as _warnings

__all__ = []

# Decision logic moved to MindOT.
# See: mindot.mindot_owl.OwlMethod
#
# Installation:
#   pip install owl[decisions]   # installs mindot
#
# Usage:
#   from mindot import MindOT
#   from mindot.mindot_owl import OwlMethod
#
#   mt = MindOT()
#   mt.add_method(OwlMethod(name="owl_predictor"))
#   result = mt.decide({"options": [...]})


def __getattr__(name: str):
    if name == "DecisionEngine":
        _warnings.warn(
            "owl.decision.DecisionEngine is deprecated. "
            "Use mindot.mindot_owl.OwlMethod instead. "
            "Install: pip install owl[decisions]",
            DeprecationWarning,
            stacklevel=2,
        )
        from mindot.mindot_owl import OwlMethod

        return OwlMethod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
