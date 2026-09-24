# OCP versioning promise

1. The `ocp` field is a scalar version string of the form `0.x`.
2. A reader accepts every published version (0.1, 0.2 and 0.3) and must ignore fields it does not know at any level of the document, including the top level, run, node, attempt, artifact, edge, event, and `ext` objects.
3. 0.3 is a strict superset of 0.2: every field 0.3 adds is optional, so a valid 0.2 document stays valid when its `ocp` value is changed to `0.3`, unless it uses a name that 0.3 defines as an unknown field of its own. 0.2 is not a superset of 0.1: 0.2 requires `tier` on every edge, so 0.1 documents need `loopmath ocp migrate`, which writes the 0.3 form.
4. A field is never removed without a deprecation notice that stays in the specification for at least two minor versions and names the replacement.
5. Extension namespaces use reverse-domain identifiers. The specification's own namespace begins `io.orchestrationcontextprotocol.` and loopmath producer extensions begin `dev.loopmath.` from v0.3. The pre-rename `dev.dagr.` namespace stays readable, with warning W182, through v0.4. The checker rejects retired namespace prefixes and names the required replacement.
