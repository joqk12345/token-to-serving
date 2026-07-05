# Source Note: Yu et al. 2022 ORCA

Source PDF: `downloads/llmsystem2026spring/source_pdfs/osdi22-yu.pdf`

## Scope

This source covers ORCA, a distributed serving system for Transformer-based generative models. The extracted sections discuss autoregressive multi-iteration inference, limitations of request-level scheduling, iteration-level scheduling, selective batching, and handling variable request lengths.

## Key Claims

- Autoregressive generative models process an inference request over multiple model iterations, with each iteration generating one output token.
- Request-level batching is inefficient for variable-length generative workloads because finished requests wait for the whole batch and newly arrived requests wait for the batch to complete.
- Iteration-level scheduling runs one model iteration on a batch and allows the scheduler to update the batch between iterations.
- Selective batching applies batching to operations where it is effective and handles attention separately when sequence lengths differ.
- ORCA is the primary paper source for continuous/iteration-level scheduling in the Chapter 12 cost model.

## Chapter 12 Use

- Use to support prefill/decode and iteration-level scheduling claims.
- Use as the paper anchor for continuous batching / selective batching.
- Avoid ORCA benchmark numbers unless full setup is carried.

## Do Not Use As

- A current benchmark comparison source.
- A complete source for modern KV-cache block management.
- A source for serving systems outside Transformer autoregressive generation without explicit qualification.

## Candidate Source Cards

- `yu-2022-orca-autoregressive-iterations`
- `yu-2022-orca-request-level-limitation`
- `yu-2022-orca-iteration-level-scheduling`
- `yu-2022-orca-selective-batching`

Owner: Technical Researcher  
Purpose: Chapter 12 ORCA source extraction  
Evidence grade: A for primary paper claims  
Assumptions: Chapter 12 uses ORCA mechanisms without reproducing benchmark numbers  
Open questions: Whether to name ORCA in main prose or use it mostly as citation support  
Handoff: Book Architect for Chapter 12 brief
