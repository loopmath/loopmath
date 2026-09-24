# Native swarm graph fixtures

These three files are lossless, hash-checked compressed envelopes of the unedited
`Graph.to_dict()` byte outputs written by the native ingest, grade, price and
graph-extract path in `tools/eval_swarms.py 3 --graph-source native` for the
2026-08-31 A, B and C evaluation swarms. Each compact JSON envelope explicitly names
the `gzip+base64` encoding, records the SHA-256 and byte length of the uncompressed
payload, and carries deterministic gzip data produced with `mtime=0`. The test loader
strictly decodes, decompresses, and verifies both values before parsing the original
JSON bytes.

The envelopes are checked in so the OCP round-trip acceptance test uses the actual
evaluation graphs without depending on one developer's live session stores or parse
cache, while keeping code-review input below backend size limits.

The fixtures contain metadata only. They intentionally retain all native graph
fields, including graph counters, artifact write provenance, and source paths. They
are historical writer artifacts: their `nodes_missing_tokens` value is `1` because
the writer then counted only a wholly absent cost mapping. They remain committed
unchanged because they are the only committed inputs exercising the legacy ten-state
compatibility path; regenerating them would delete that evidence. Freshly generated
output uses the shared six-stream completeness rule. The round-trip test does not
hand-build, reduce, or normalize the reconstructed graph before calling `to_ocp`.
