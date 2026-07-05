# Source Note: llmsys-27 LMCache

Source PDF: `downloads/llmsystem2026spring/source_pdfs/llmsys-27-LMCache_junchenjiang-c21828fe582270cb5e08b4a21a002956.pdf`

## Scope

This source covers KV cache as reusable serving data, external KV-cache storage, LMCache as a separated KV-cache management service, zero-copy CPU sharing, remote KV pools, storage plugins, and research directions such as CacheBlend and KV-cache compression.

## Key Claims

- KV cache avoids repeated computation by preserving attention state for reuse.
- KV cache can become an AI-native data object that may be stored, moved, blended, compressed, and shared.
- LMCache separates KV-cache management from the inference engine instead of running only as an in-process library.
- LMCache can use a separate multi-process CPU pool and remote KV pool to reduce extra copies when moving KV between GPU workers.
- LMCache exposes hooks/storage plugins for get/put KV-cache operations, including integration with storage or transfer systems.
- CacheBlend and selective prefill are presented as research ideas for combining reused KV cache with recomputation of selected non-prefix tokens.
- KV-cache compression can reduce stored cache size and transfer volume, but benchmark claims need separate conditions.

## Chapter 14 Use

- Use LMCache to explain why KV cache becomes an object managed outside one inference process.
- Use zero-copy CPU sharing as a mechanism example, not a benchmark claim.
- Mention CacheBlend/compression only as optional advanced caching directions unless the draft creates deeper source cards.

## Candidate Source Cards

- `llmsys-27-kv-cache-reuse`
- `llmsys-27-kv-cache-ai-native-data`
- `llmsys-27-lmcache-separated-service`
- `llmsys-27-zero-copy-cpu-sharing`
- `llmsys-27-storage-plugin-interface`
- `llmsys-27-cacheblend-selective-prefill`
- `llmsys-27-kv-cache-compression`

Owner: Technical Researcher
Purpose: Chapter 14 LMCache source extraction
Evidence grade: A for course lecture framing; benchmark claims require conditions
Assumptions: Chapter 14 uses LMCache to discuss external KV-cache management, not as a current product survey
Open questions: Whether to add LMCache paper/docs/source-code cards for production implementation details
Handoff: Book Architect for Chapter 14 brief
