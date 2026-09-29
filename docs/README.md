# Diagrams

`aardvark-architecture.html` is a single self-contained file. Open it in a
browser; nothing is fetched at runtime.

`aardvark-architecture.candidate.json` is its source. Every component carries
the repository paths and line ranges the claim rests on, and `meta.repository`
pins the commit those lines were read at, so a reader can check the diagram
against the code rather than trusting it.

Regenerate after the code moves:

```bash
node <archify>/bin/archify.mjs finalize architecture \
  docs/aardvark-architecture.candidate.json \
  docs/aardvark-architecture.html \
  --repo-root . --quality showcase --json
```

Update `meta.repository.revision` in the same edit. A line range that has moved
is a stale citation, and a diagram whose citations are stale is worse than one
with none: it looks checkable and is not.

The HTML is committed because it is the artifact people actually open, and it
is generated, so it changes wholesale on every regeneration. If that churn
becomes annoying, drop it and keep the candidate.
