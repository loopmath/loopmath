# OCP versioning promise

1. The `ocp` field is a scalar version string of the form `0.x`.
2. Within the `0.x` line every change is additive, so a reader must ignore fields it does not know at any level of the document, including the top level, run, node, attempt, artifact, edge, event, and `ext` objects.
3. A field is never removed without a deprecation notice that stays in the specification for at least two minor versions and names the replacement.
4. Extension namespaces use registered reverse-domain identifiers. The specification's own namespace begins `io.orchestrationcontextprotocol.` and loopmath producer extensions begin `dev.loopmath.` from v0.3. The pre-rename `dev.dagr.` namespace stays readable, with warning W182, through v0.4. The checker rejects retired namespace prefixes and names the required replacement.
