# Source Card: official-pytorch-pipeline-parallelism

Source ID: official-pytorch-pipeline-parallelism  
Title: Pipeline Parallelism — PyTorch 2.12 documentation  
Author/issuer: PyTorch documentation  
Date: Accessed 2026-07-05  
Source type: official-docs  
File path / URL: `https://docs.pytorch.org/docs/2.12/distributed.pipelining.html`  
Pages / slides / sections: Pipeline Parallelism overview, PipelineStage, PipelineSchedule  
Claim supported: PyTorch `torch.distributed.pipelining` provides pipeline model splitting and distributed runtime support for micro-batch scheduling, communication, gradient propagation, GPipe, 1F1B, and interleaved 1F1B schedules; docs mark it alpha.  
Exact quote: "alpha state"  
Paraphrase: The official docs describe pipeline parallelism as partitioning model execution so multiple micro-batches can execute different parts concurrently, implemented with `PipelineStage` and schedules such as `ScheduleGPipe` and `Schedule1F1B`.  
Evidence grade: A  
Technical sensitivity: implementation  
Conditions:
  model:
  hardware:
  batch size:
  sequence length:
  precision:
  software version: PyTorch 2.12 documentation
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Preserve alpha/API-change caveat in draft.
