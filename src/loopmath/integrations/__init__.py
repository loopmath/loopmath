"""Bridges from other tools' optimization loops to loopmath's gate.

Each module reads what a tool already holds (per-instance scores, costs) and asks `loopmath.gate.decide`.
None of them imports the tool it serves: the tool's objects are read by their field names.
"""
