"""
Base Agent - Abstract foundation for all GTAA agents.
Every agent produces a typed Signal with reasoning trail.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import logging
import json


@dataclass
class Signal:
    """Standardized output from any agent."""
    agent_name: str
    timestamp: datetime
    signal_type: str              # "momentum", "regime", "risk", "allocation", "decision", "review"
    data: Dict[str, Any]          # the actual signal payload
    confidence: float             # 0.0 to 1.0
    reasoning: str                # human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "type": self.signal_type,
            "data": self.data,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class AuditEntry:
    """Single audit trail entry."""
    timestamp: datetime
    agent: str
    action: str
    details: Dict[str, Any]


class BaseAgent(ABC):
    """Abstract base for all GTAA agents."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.audit_trail: List[AuditEntry] = []
        self._last_signal: Optional[Signal] = None

    def _log_audit(self, action: str, details: Dict[str, Any]):
        entry = AuditEntry(
            timestamp=datetime.now(),
            agent=self.name,
            action=action,
            details=details,
        )
        self.audit_trail.append(entry)

    @abstractmethod
    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """Run analysis and produce a signal. Must be implemented by each agent."""
        pass

    @property
    def last_signal(self) -> Optional[Signal]:
        return self._last_signal

    def get_audit_trail(self, last_n: int = 10) -> List[Dict]:
        return [
            {"timestamp": e.timestamp.isoformat(), "agent": e.agent,
             "action": e.action, "details": e.details}
            for e in self.audit_trail[-last_n:]
        ]
