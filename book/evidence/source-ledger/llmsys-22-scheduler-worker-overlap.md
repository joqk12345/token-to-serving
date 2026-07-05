# Source Card: llmsys-22-scheduler-worker-overlap

Source ID: llmsys-22-scheduler-worker-overlap  
Title: Design of Efficient LLM Inference Server  
Author/issuer: Lei Li / CMU LLM Systems  
Date: 2026  
Source type: lecture PDF  
File path / URL: `downloads/llmsystem2026spring/source_pdfs/llmsys-22-llm-serving-scheduler-radixattention-dfa87a4515092525676277a85bc4425d.pdf`  
Pages / slides / sections: Slides 40-46  
Claim supported: Scheduler CPU work can become overhead, and serving systems can overlap CPU scheduling with GPU model-worker execution.  
Exact quote: "Overlap CPU scheduler and GPU worker"  
Paraphrase: Request scheduling itself can enter the critical path, so server design may overlap scheduler and worker activity.  
Evidence grade: A  
Technical sensitivity: scheduler overhead  
Conditions:
  model: LLM serving loop
  hardware: CPU scheduler plus GPU worker
  batch size:
  sequence length:
  precision:
  software version:
Dispute status: none  
Checked by: Codex  
Checked date: 2026-07-05  
Notes: Avoid performance-number claims without setup.
