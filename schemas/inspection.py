"""Common inspection verdict shared by domain and API models."""

from enum import Enum
from typing import Any


class InspectionVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"

    @classmethod
    def from_value(cls, value: Any) -> "InspectionVerdict | None":
        if isinstance(value, cls):
            return value
        if value is True:
            return cls.PASS
        if value is False:
            return cls.FAIL
        if isinstance(value, str):
            normalized = value.strip().upper()
            aliases = {
                "TRUE": cls.PASS,
                "FALSE": cls.FAIL,
                "PASS": cls.PASS,
                "FAIL": cls.FAIL,
                "REVIEW": cls.REVIEW,
            }
            return aliases.get(normalized)
        return None

    @property
    def legacy_status(self) -> str:
        return "true" if self is InspectionVerdict.PASS else "false"
