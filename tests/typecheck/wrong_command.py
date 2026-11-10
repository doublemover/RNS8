"""Deliberately invalid checker input; never an ordinary passing source."""
from evidence_records import Command

invalid = Command(role="diagnostic", argv="must be a tuple of strings")
