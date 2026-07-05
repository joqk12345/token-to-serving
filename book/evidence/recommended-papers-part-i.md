# Recommended Papers — Part I

This file extracts paper recommendations and citations from the Part I course slides:

- `llmsys-06-transformer`
- `llmsys-07-llms`
- `llmsys-08-tokenization`
- `llmsys-09-decoding`

It separates papers into required anchors, useful supporting sources, and later-chapter sources already present locally.

## Required Anchors for Chapters 1-3

| Priority | Paper                                                                                                                                      | Where it appears             | Use in book                                                                        | Local PDF? | Action                                                  |
| -------: | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | ---------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------- |
|        1 | Vaswani et al., "Attention Is All You Need", 2017                                                                                          | Lecture 6                    | Transformer architecture, attention, positional encoding, encoder-decoder baseline | no         | Added card: `vaswani-2017-attention-is-all-you-need`    |
|        2 | Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", 2016                                                       | Lectures 7, 8                | BPE origin for NMT and subword algorithm wording                                   | no         | Added card: `sennrich-2016-bpe-rare-words`              |
|        3 | Kudo and Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", 2018 | Lecture 7 reading, Lecture 8 | Modern tokenizer practice and raw-sentence whitespace handling                     | no         | Added card: `kudo-richardson-2018-sentencepiece`        |
|        4 | Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", 2020                                   | Lecture 7                    | T5 / text-to-text encoder-decoder example                                          | no         | Added card: `raffel-2020-t5`                            |
|        5 | Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023                                                               | Lecture 7                    | decoder-only modern LLM example, pre-norm, SwiGLU, RoPE                            | no         | Added card: `touvron-2023-llama`                        |
|        6 | Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021                                                           | Lecture 7                    | RoPE mechanism                                                                     | no         | Added card: `su-2021-roformer-rope`                     |
|        7 | Brown et al., "Language Models are Few-Shot Learners", 2020                                                                                | Lecture 7                    | GPT-3 example, sparse attention note, model-scale historical anchor                | no         | Add source card if GPT-3 remains beyond passing mention |

## Decoding and Generation Sources

| Priority | Paper / Source                                                                                                                  | Where it appears | Use in book                                                | Local PDF? | Action                                                                 |
| -------: | ------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------------------- | ---------- | ---------------------------------------------------------------------- |
|        1 | Freitag and Al-Onaizan, "Beam Search Strategies for Neural Machine Translation", 2017                                           | Lecture 9        | beam pruning and beam-search caveats                       | no         | Optional source card if beam search gets more than a basic explanation |
|        2 | Kool et al., "Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement", 2019 | Lecture 9        | Gumbel sampling sidebar                                    | no         | Optional                                                               |
|        3 | Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", 2022                                             | Lecture 9        | draft/target validation explanation                        | no         | Added card: `leviathan-2022-speculative-decoding`                      |
|        4 | Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", 2023                                       | Lecture 9        | distribution-preserving speculative sampling corroboration | no         | Added card: `chen-2023-speculative-sampling`                           |
|        5 | Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", 2024                                          | Lecture 9        | feature-level speculative decoding sidebar                 | no         | Added card: `li-2024-eagle`                                            |

## Tokenization Advanced Sources

| Priority | Paper                                                                                                          | Where it appears | Use in book                                  | Local PDF? | Action                                             |
| -------: | -------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------------------- | ---------- | -------------------------------------------------- |
|        1 | Xu, Zhou, Gan, Zheng, Li, "Vocabulary Learning via Optimal Transport for Neural Machine Translation", ACL 2021 | Lecture 8        | VOLT sidebar                                 | no         | Optional; add only if VOLT remains                 |
|        2 | Dagan et al., "Getting the Most out of Your Tokenizer for Pre-training and Domain Adaptation", ICML 2024       | Lecture 8        | code/domain tokenizer considerations         | no         | Optional                                           |
|        3 | Golkar et al., "xVal: A Continuous Numerical Tokenization for Scientific Language Models", 2023                | Lecture 8        | numerical tokenization sidebar               | no         | Optional                                           |
|        4 | Liang et al., "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models", EMNLP 2023 | Lecture 8        | multilingual vocabulary allocation           | no         | Optional                                           |
|        5 | Yuan et al., "How Vocabulary Sharing Facilitates Multilingualism in LLaMA?", ACL 2024                          | Lecture 8        | vocabulary sharing and multilingual behavior | no         | Optional                                           |
|        6 | Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens", 2024                              | Lecture 8        | tokenizer-free model sidebar                 | no         | Added card: `pagnoni-2024-byte-latent-transformer` |

## Hugging Face Daily Papers 2026-07-03 Scan

The user requested a supplemental check against `https://huggingface.co/papers/date/2026-07-03`. Most trending papers that day are agent, benchmark, vision, or application papers rather than Part I foundations.

One paper is relevant to later LLM systems chapters:

| Paper                                                     | HF / arXiv           | Use in book                                                                          | Card                                   |
| --------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------ | -------------------------------------- |
| Lan et al., "Morphing into Hybrid Attention Models", 2026 | HF page `2606.30562` | Forward-looking long-context / hybrid attention material for Chapter 6 or Chapter 15 | `lan-2026-flashmorph-hybrid-attention` |

## Local Supporting Papers Already in `source_pdfs`

These are not Part I anchors, but they are already downloaded and should be used later.

| File               | Paper                                                                                      | Likely chapter              |
| ------------------ | ------------------------------------------------------------------------------------------ | --------------------------- |
| `146.pdf`          | Frostig, Johnson, Leary, "Compiling machine learning programs via high-level tracing"      | Chapter 7, frameworks / JAX |
| `1910.02054.pdf`   | Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" | Chapter 10, ZeRO            |
| `2201.05596.pdf`   | Rajbhandari et al., "DeepSpeed-MoE"                                                        | Chapter 10, MoE             |
| `2205.14135.pdf`   | Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"  | Chapter 6, FlashAttention   |
| `osdi16-abadi.pdf` | Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning"                      | Chapter 7, frameworks       |
| `osdi22-yu.pdf`    | Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"    | Chapter 12/14, serving      |
| `p3005-li.pdf`     | Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training"       | Chapter 8, DDP              |

## Immediate Recommendation

Before revising Part I prose, create source-ledger cards for:

```text
vaswani-2017-attention-is-all-you-need
sennrich-2016-bpe-rare-words
kudo-richardson-2018-sentencepiece
raffel-2020-t5
touvron-2023-llama
su-2021-roformer-rope
leviathan-2022-speculative-decoding
chen-2023-speculative-sampling
li-2024-eagle
```

Done on 2026-07-04. Next: run a second review pass and decide whether GPT-3, VOLT, EAGLE, multilingual tokenizer behavior, and tokenizer-free models stay in main text or become sidebars.
