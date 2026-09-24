"""
Gestandaardiseerd resultaat object voor elke pipeline stap.
Zo kun je in main.py altijd op dezelfde manier controleren
of een stap gelukt is.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StepResult:
    """
    Resultaat van één pipeline stap.

    Gebruik:
        result = StepResult.ok(data=camera)
        result = StepResult.fail("Camera niet gevonden: ...")

        if result:          # True als success
            ...
        print(result)       # Toont status + bericht
    """

    success:  bool
    message:  str
    data:     Any = field(default=None, repr=False)
    details:  Optional[str] = None   # Technische details voor debug

    # ── Constructors ─────────────────────────────────────────

    @classmethod
    def ok(cls, message: str = "OK", data: Any = None) -> "StepResult":
        return cls(success=True, message=message, data=data)

    @classmethod
    def fail(cls, message: str, details: Optional[str] = None) -> "StepResult":
        return cls(success=False, message=message, details=details)

    # ── Helpers ──────────────────────────────────────────────

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        base = f"[{status}] {self.message}"
        if not self.success and self.details:
            base += f"\n    Details: {self.details}"
        return base