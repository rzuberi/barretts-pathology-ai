# Follow-up F addendum: perturbation prediction (fixed 2026-10-16 17:05, before any training)

**Benchmark (confirmed from the papers).** PerturbQA (Wu et al., ICLR 2025; github.com/genentech/PerturbQA, commit df2fe3a), as
used by SynthPert (arXiv 2509.25346). Four CRISPRi Perturb-seq cell lines: K562, RPE1, HepG2, Jurkat. Differential expression (DE):
614,479 (perturbation, gene) pairs; direction of change (DIR): 84,550 DE-positive pairs, label 1 = increase, 0 = decrease. Split
75:25 **by perturbation** (the test perturbations are unseen in training), as distributed.

**Task (SynthPert's three-class formulation).** Given cell line, perturbed gene and target gene → `up`, `down` or `none` (not
differentially expressed). DE pairs with label 0 → none; label 1 → up/down from the DIR file.

**Metrics (as both papers).** DE AUROC per cell line = PerturbQA `auc_per_gene` with score P(up) + P(down); DIR AUROC per cell line =
`auc_per_gene` on the DIR test pairs with score P(up) / (P(up) + P(down)); three-class accuracy (argmax) on the DE test pairs.
Probabilities = the model's next-token probabilities of the three answer words.

**Model.** Qwen3-8B in non-thinking mode, LoRA r = 16 (SynthPert used an 8B R1-distilled Llama; Qwen3-8B is chosen to keep GRPO
rollouts short). Prompt: cell line, perturbed gene, target gene, the three options; answer one word.
- zero-shot;
- SFT: 1 epoch on 48,000 training examples (16,000 per class, sampled across the four cell lines' training splits, seed 20261016);
- SFT→GRPO: 300 steps from the SFT model, 16 prompts × 8 completions, class-balanced training prompts, reward 1 for the right word,
  0 otherwise, −0.5 for no valid word.
Checkpoint choice on a 10% validation split of training **perturbations** (perturbation-disjoint from training, as the paper's
validation), by three-class accuracy.

**Cross-cell test (secondary).** The same three arms retrained without RPE1 (training on K562, HepG2, Jurkat), tested on RPE1's test split.

**Primary endpoint.** Mean DE and DIR AUROC over the four cell lines, SFT→GRPO minus SFT; reported beside SynthPert's published
table (their "LLM + SFT on data": DE 0.59–0.60, DIR 0.47–0.55; SynthPert: DE 0.70–0.79, DIR 0.65–0.73; three-class 89%).
Bootstrap over perturbations (2,000) for every CI. Stopping rule: fixed steps; a null result is reported as such.
