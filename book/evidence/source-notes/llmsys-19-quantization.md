# Source Note: llmsys-19 Quantization Basic Methods

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-19-quantization-da7a2abad092c802b03672ce1cc7bee9.pdf`

## Scope

This source covers low-precision formats, direct quantization, absmax and zero-point quantization, post-training versus training-time quantization approaches, layer-wise quantization objectives, ZeroQuant, and LLM.int8 outlier handling.

## Key Claims

- Quantization stores parameters and layer outputs in lower precision to reduce memory and potentially improve throughput, with possible accuracy loss.
- Direct quantization from FP32 to INT8/INT4 can lose information through precision loss, clipping/range mismatch, and rounding error.
- Absmax quantization scales values according to maximum absolute value.
- Zero-point quantization uses a scale and offset/zero point to map a value range into an integer range.
- Quantization approaches include training-time and post-training methods.
- Layer-wise quantization can frame the problem as minimizing the output difference between full-precision and quantized linear projections.
- ZeroQuant uses layer-by-layer knowledge distillation with the full-precision model as teacher and the quantized model as student.
- LLM.int8 keeps outlier features in higher precision while quantizing most work to 8-bit.

## Chapter 11 Use

- Use as the grounding source for quantization vocabulary and error modes.
- Explain quantization as a data representation and systems tradeoff, not just a compression trick.
- Use absmax and zero-point quantization as simple mechanisms before GPTQ.
- Treat ZeroQuant and LLM.int8 as examples of scaling quantization to large transformers without quoting benchmark numbers.

## Do Not Use As

- A source for unconditional speedup claims.
- A benchmark source without model, hardware, precision, batch, and implementation conditions.
- A full current survey of quantization methods.

## Candidate Source Cards

- `llmsys-19-quantization-purpose`
- `llmsys-19-direct-quantization-errors`
- `llmsys-19-absmax-zeropoint`
- `llmsys-19-quantization-approaches`
- `llmsys-19-layerwise-objective`
- `llmsys-19-zeroquant`
- `llmsys-19-llmint8-outliers`

Owner: Technical Researcher  
Purpose: Chapter 11 quantization basics extraction  
Evidence grade: A for course framing; original papers needed for named-system details  
Assumptions: Chapter 11 focuses on adaptation/compression mechanisms, not serving benchmarks  
Open questions: Whether to include simple quantization equations in the main draft  
Handoff: Book Architect for Chapter 11 brief
