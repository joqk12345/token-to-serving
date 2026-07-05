# Chapter 6 Online Softmax Formula Check

Date: 2026-07-04

Scope: `book/chapters/06-flashattention-transformer-acceleration.md`

## Verdict

The online softmax formulas added to Chapter 6 are suitable for draft use and match the FlashAttention paper's blockwise rescaling idea.

They should still receive final notation review before publication, because the chapter uses a simplified one-row presentation rather than the paper's full block notation.

## Checked Formula

For old accumulated state:

```text
m_old, l_old, O_old
```

and new block:

```text
m_block = max(s_block)
p_block = exp(s_block - m_block)
l_block = sum(p_block)
O_block = p_block V_block
```

the merge uses:

```text
m_new = max(m_old, m_block)

l_new =
  exp(m_old - m_new) * l_old
  + exp(m_block - m_new) * l_block

O_new =
  (exp(m_old - m_new) * l_old * O_old
   + exp(m_block - m_new) * O_block)
  / l_new
```

This is internally consistent if `O_old` is interpreted as the normalized accumulated output for the blocks seen so far, while `O_block` is the unnormalized weighted value sum for the new block.

## Required Wording Constraint

The prose must make clear that:

- `O_old` is already normalized by `l_old`;
- `O_block` is not yet normalized by the global denominator;
- the formulas are for one row, while implementation operates on blocks.

## Follow-up Edit

Add one sentence after the formula in Chapter 6 to state the normalized/unnormalized convention explicitly.

Owner: Technical Reviewer  
Purpose: Formula review for Chapter 6 draft  
Evidence grade: A for review process; source claim anchored to `dao-2022-online-softmax`  
Open questions: Final publication may prefer paper notation if the chapter becomes more mathematical  
Handoff: Systems Explainer for wording adjustment
