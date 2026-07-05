# Source Note: llmsys-20 GPTQ

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-20-quantization2-ba573d7e5d82e68027bbd3a92c3cd819.pdf`

## Scope

This source is about GPTQ as a post-training quantization method for generative Transformer models. It discusses layer-wise weight quantization, column/block-wise processing, rounding-error compensation, Hessian inverse/Cholesky precomputation, OBQ background, and lazy batch updates.

## Key Claims

- GPTQ targets quantization for very large models while maintaining accuracy.
- GPTQ revisits layer-wise quantization of weight matrices.
- GPTQ quantizes one column block at a time.
- GPTQ updates not-yet-quantized weights to compensate for the error introduced by quantizing earlier weights.
- GPTQ uses precomputed information from the Hessian inverse of layer inputs.
- Cholesky precomputation helps provide needed inverse-Hessian information in a numerically manageable way.
- Lazy batch updates improve practical efficiency by batching updates to remaining columns.
- GPTQ evaluation uses calibration data and per-row asymmetric quantization in the lecture setup.

## Chapter 11 Use

- Use GPTQ as the main example of second-order post-training quantization.
- Explain the mechanism qualitatively before formulas: quantize, measure error, compensate future weights.
- Avoid reproducing complex formulas unless the notation is checked line by line.
- Do not quote model-size, runtime, or inference speed results without full paper conditions.

## Do Not Use As

- A source for unconditional claims that GPTQ dominates other quantization methods across workloads.
- A benchmark source without hardware/model/calibration/kernel details.
- A proof-oriented treatment of OBS or OBQ.

## Candidate Source Cards

- `llmsys-20-gptq-goal`
- `llmsys-20-gptq-blockwise-compensation`
- `llmsys-20-gptq-hessian-cholesky`
- `llmsys-20-gptq-lazy-updates`

Owner: Technical Researcher  
Purpose: Chapter 11 GPTQ source extraction  
Evidence grade: A for course framing; GPTQ paper needed for publication-level claims  
Assumptions: Chapter 11 explains GPTQ as a systems-aware PTQ method, not a full optimization-theory derivation  
Open questions: How much Hessian notation should be in the chapter draft  
Handoff: Book Architect for Chapter 11 brief
