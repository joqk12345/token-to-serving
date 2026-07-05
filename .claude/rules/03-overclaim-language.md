# Technical Overclaim Language

Hedge every technical claim whose validity depends on workload, hardware, implementation, or benchmark setup.

## Flagged Phrases

These phrases require conditions or a source:

| Phrase | Safer treatment |
|---|---|
| always faster | State workload and hardware, or use "can be faster when..." |
| solves the problem | Use "addresses", "reduces", or specify the remaining bottleneck. |
| optimal | Name the objective and assumptions. |
| best | Name the metric and comparison set. |
| proves | Use "shows in this setting", unless it is a formal proof. |
| negligible | Quantify or remove. |
| bottleneck is | State how measured or inferred. |
| memory-bound / compute-bound | Provide arithmetic intensity, profiler result, or clear reasoning. |

## Required Conditions for Numbers

Any speed, memory, throughput, latency, or scaling number should carry as many of these as available:

- model size;
- sequence length;
- batch size;
- hardware;
- precision;
- software/runtime;
- baseline;
- dataset or workload.

