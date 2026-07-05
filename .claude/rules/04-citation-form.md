# Citation Form

Use a three-layer citation model.

## Layer 1 — Prose Source Identity

When a source is important to trust, name it in prose:

- "The FlashAttention paper frames the bottleneck as..."
- "The vLLM paper reports..."
- "The course lecture on distributed training introduces..."

## Layer 2 — Inline Source Marker

Use short source-ledger markers:

```markdown
[CITE: flashattention-paper]
[CITE: llmsys-14-distributed-training]
```

The marker contains only one or more source card slugs separated by semicolons.

## Layer 3 — Generated References

Later build steps should generate references from source-ledger cards. Do not put full citation metadata inline in chapter prose.

## Missing Evidence

Use:

```markdown
[EVIDENCE NEEDED: describe the missing source]
```

Replace it with `[CITE: slug]` after a source card exists.

