"""
Escalation Policy Engine module for CompanySupport.
Re-exports from src.escalation.policy for backwards compatibility.
"""

from src.escalation.policy import (
    EscalationEngine,
    RE_EMAIL,
    RE_IMEI,
    RE_SERIAL,
    RE_ORDER_NO
)

__all__ = [
    "EscalationEngine",
    "RE_EMAIL",
    "RE_IMEI",
    "RE_SERIAL",
    "RE_ORDER_NO"
]
