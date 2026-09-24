# Rust step-1 parity fixtures

The JSON goldens are compact UTF-8 output from the live Python scan and price
stages, with one trailing newline. `tests/test_rust_parity.py` recomputes every
golden from its checked-in input, then asks the standalone Rust binary to load
the typed stage record and emit it again. Both comparisons are byte-for-byte.

The Claude/Codex scan inputs cover both top-level dictionary variants, nullable
base fields, every production event annotation, scanner counters, and both
present and absent origin metadata. The price input uses a local table so its
arithmetic and warning maps are deterministic; it covers malformed and complete
token blocks, copied extension fields, assignment into existing result-key
positions, Unicode/control escaping, and the GPT-5.6 warning counter.

Open price records are deserialized with lexical JSON-number preservation. This
is the step-1 parity contract: Rust does not calculate prices or regenerate their
numbers as `f64`. The golden pins CPython's midpoint mantissas, exponent spelling,
signed zero, threshold forms, and both adjacent IEEE patterns from the rejected
attempt's differing-mantissa review case.

These are parity instruments, not Rust implementations of scanning or pricing.
Python remains the oracle and owns both algorithms in step 1.
