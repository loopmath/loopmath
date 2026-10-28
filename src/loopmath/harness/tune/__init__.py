"""`loopmath harness` (experimental): tune a Claude Code harness for one task, or a small task set.

The engine makes base runs, has editors propose atomic edits, runs a 12-run Plackett-Burman screen of probes over
them, fits the edits' effects by least squares, builds the kept edits mechanically, runs the build once and applies
a delivery rule. One TOML file drives it (`schema.FIELDS` is its table); every field also has a flag.

    contracts  the records every module passes and the protocols each component implements
    schema     FIELDS: key, type, default, choices, help and source of every config field
    events     the event stream: types, the validator, the writer, the reader and the status fold
"""

CONFIG_SCHEMA = "loopmath.harness-tune/0.1"
EVENTS_SCHEMA = "loopmath.harness-events/0.1"
