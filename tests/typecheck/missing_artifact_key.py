"""Deliberately invalid checker input; raw JSON cannot omit its digest."""
from evidence_records import ArtifactReference

invalid: ArtifactReference = {"path": "authored-not-evidence"}
