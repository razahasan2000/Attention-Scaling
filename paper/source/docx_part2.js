// ============================================================================
// Part 2: Method, Results, Discussion, References + document assembly
// ============================================================================
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, PageNumber,
  AlignmentType, LevelFormat,
} = require("docx");
const fs = require("fs");
const P1 = require("./docx_part1.js");
const { children, body, h1, h2, bullet, numbered, eq, mr, attnEq,
        MathSubScriptSafe, eqGeneric, eqCI, eqDZ, caption, tableCaption,
        figure, scholarlyTable, refPara, OUT, TNR, BLACK, GREY } = P1;

// ===== 3. Method =====
children.push(h1("3. Method"));

children.push(h2("3.1 Problem setting and notation"));
children.push(body("Let Q, K, V denote the query, key, and value matrices for a single attention head over a sequence of n tokens, where d_k = d_model/h is the per-head dimension. Every variant examined in this study computes attention parameterised solely by the scaling factor s > 0 — the constant by which the logits are divided before the softmax — and the variants differ only in s:"));
children.push(eqGeneric);

children.push(h2("3.2 Attention scaling variants"));
children.push(body("The three primary variants are the standard scaled dot-product attention of Vaswani et al. [1], the alternative 1/d_k divisor that removes the square root, and the temperature-scaled softmax with τ = 2:"));
children.push(attnEq("std", [mr("√"), mr("d"), new MathSubScriptSafe("", "k")],
  [mr("s"), new MathSubScriptSafe("", "std"), mr(" = √d"), new MathSubScriptSafe("", "k"), mr(" ≈ 5.66")], 2));
children.push(attnEq("1/d_k", [mr("d"), new MathSubScriptSafe("", "k")],
  [mr("s"), new MathSubScriptSafe("", "1/dk"), mr(" = d"), new MathSubScriptSafe("", "k"), mr(" = 32")], 3));
children.push(attnEq("τ", [mr("τ√"), mr("d"), new MathSubScriptSafe("", "k")],
  [mr("s"), new MathSubScriptSafe("", "τ"), mr(" = τ√d"), new MathSubScriptSafe("", "k"), mr(" ≈ 11.31  (τ = 2)")], 4));
children.push(body("All three are parameter-free rescalings of the same logits. They introduce no additional weights, no additional arithmetic of consequence, and no change to memory traffic. Two auxiliary ablation arms probe the mechanism (Section 4.9): a mixed-scale variant in which two of the four heads use standard scaling and two use 1/d_k scaling, and a learning-rate compensation arm in which the learning rates of the query and key projections of the 1/d_k variant are multiplied by √d_k ≈ 5.66 to offset the predicted 1/s gradient attenuation. A temperature sweep τ ∈ {0.5, 1.0, 1.5, 2.0} additionally probes the direction of the effect around the standard τ = 1 point."));

children.push(h2("3.3 Model architecture"));
children.push(body("The backbone is a compact pre-LN Transformer encoder [13, 14] whose dataflow is shown in Figure 1. Token embeddings (vocabulary 18,855) and learned positional embeddings (maximum length 128) are summed and passed through two pre-LN encoder blocks, each computing LayerNorm → multi-head self-attention → residual add, followed by LayerNorm → feed-forward (GELU) → residual add [24]. A final LayerNorm is followed by masked mean pooling over non-padding positions and a task-dependent head: a linear classifier for SST-2/IMDB (2 classes), AG News (4 classes), or a weight-tied linear language-modelling head for WikiText-2. Dropout (p = 0.1) [25] is applied to the embeddings and to the attention and feed-forward outputs. Query, key, value, and output projections are bias-free linear maps of width d_model = 128 with four heads (d_k = 32), and the feed-forward width is 256. The scaling factor s enters exactly one line of the forward pass: the division of the logits QK⊤ inside each attention head."));
children.push(figure("fig_architecture.png", 580));
children.push(caption("Model architecture and the manipulated factor. Left: the compact pre-LN encoder shared by all variants (2 blocks, 4 heads, d_k = 32). Right: the three scaling variants, which differ only in the effective divisor s of the attention logits. Parameter counts, FLOPs, and memory are identical across variants."));

children.push(tableCaption("Model and optimisation specification (identical for all variants)."));
children.push(scholarlyTable(
  ["Component", "Value", "Notes"],
  [
    ["Vocabulary / max. sequence length", "18,855 / 128", "built from training data only"],
    ["d_model / heads / d_k", "128 / 4 / 32", "per-head dimension"],
    ["Encoder layers / FFN width", "2 / 256", "pre-LN residual ordering"],
    ["Projections W_q, W_k, W_v, W_o", "128 × 128, bias-free", "per block"],
    ["Activation / dropout", "GELU / p = 0.1", "[24, 25]"],
    ["Pooling", "masked mean", "non-padding positions"],
    ["Task head", "Linear (2/4 classes) or tied LM head", "WikiText-2: weight-tied"],
    ["Attention logit clamp", "[−50, 50]", "numerical stability"],
    ["Parameter count", "2,695,556 (cls) / 4,120,576 (LM)", "identical across variants"],
    ["Optimiser", "AdamW [26]", "β = (0.9, 0.999), ε = 1e-8 [27]"],
    ["Learning rate / weight decay", "3 × 10⁻⁴ / 0.01", ""],
    ["LR schedule", "linear warmup (5% of steps), cosine decay", ""],
    ["Batch size", "64 (classification) / 16 (LM)", "eval. 64 / 32"],
    ["Gradient clipping", "max ||g||₂ = 1.0", ""],
    ["Precision", "FP32 (mixed precision disabled)", "reproducibility"],
    ["Budgets", "2 epochs (core) / 25 epochs (extended) / 25,000 steps (LM)", "frozen per grid"],
  ],
  [34, 36, 30]));

children.push(h2("3.4 Datasets and preprocessing"));
children.push(body("Four datasets span two task families (Table 2): topic classification (AG News, 4 classes [28]), sentiment classification (SST-2 [29] and IMDB [30]), and language modelling (WikiText-2 [31]). All datasets are loaded through the HuggingFace datasets ecosystem [32] with fixed splits: classification training sets are subsampled to 10,000 examples with a 20% held-out validation split (SST-2 uses the official GLUE validation and test splits), and WikiText-2 is tokenised into 128-token sequences. Classification text is tokenised with a lowercased regular-expression pattern ([a-z0-9']+); language modelling uses whitespace tokenisation with an end-of-sequence token. Vocabularies (top 18,855 tokens at minimum frequency 2) are built from training data only, so no test information leaks into preprocessing. Sequences are right-padded to length 128 with attention masks that exclude padding positions."));
children.push(tableCaption("Datasets and splits. Classification training sets are subsampled to 10,000 examples; validation splits follow the repository configuration."));
children.push(scholarlyTable(
  ["Dataset", "Task", "Classes", "Train", "Val", "Test", "Source"],
  [
    ["AG News", "topic classification", "4", "10,000", "2,000", "7,600", "[28]"],
    ["SST-2", "sentiment classification", "2", "10,000", "1,115", "1,821", "[29]"],
    ["IMDB", "sentiment classification", "2", "10,000", "2,000", "25,000", "[30]"],
    ["WikiText-2", "language modelling", "—", "36,718 seq", "3,760 seq", "4,358 seq", "[31]"],
  ],
  [16, 24, 10, 13, 12, 13, 12]));

children.push(h2("3.5 Matched-seed paired design"));
children.push(body("The core classification grid is 3 variants × 25 seeds × 3 datasets = 225 runs at GPU scale; the language-modelling grid (3 variants × 25 seeds) is completed in Stage 2 at replica scale with the corrected causal mask (Section 3.6). Within a seed s ∈ {1001, …, 1025}, the three variants share (i) a bit-identical parameter initialization θ₀(s) — verified by unit tests that compare initial-state hashes across variants, (ii) the same training-example order driven by a seeded permutation generator, (iii) the same optimizer state trajectory shape, and (iv) the same frozen budget (Figure 2). The unit of statistical analysis is therefore the within-seed difference d(s) = a_std(s) − a_v(s), giving n = 25 paired observations per comparison. This design removes initialization and data-order variance — typically the dominant variance components in small-model training [9] — from the comparison, and it is exactly the matched-pair structure recommended for NLP evaluation [12]."));
children.push(figure("fig_design.png", 560));
children.push(caption("Matched-seed paired experimental design. Each seed contributes one paired observation per comparison, so the scaling function is the only manipulated difference between arms."));

children.push(h2("3.6 Stage-2 replica protocol and artifact probes"));
children.push(body("Stage 2 completes the grids that Stage 1 left open on a documented replica protocol, chosen to fit a 2-core CPU sandbox (3 GB RAM, PyTorch 2.15 CPU build) while preserving every structural property of the design: the same pre-LN architecture, the same three scaling variants with the same effective divisors, the same matched-seed construction (bit-identical initial state, shared data order), the same optimizer (AdamW, lr 3×10⁻⁴, weight decay 0.01, 5% linear warmup then cosine decay, gradient clip 1.0), and the same analysis pipeline. The reductions are: 2,000-example training subsets (val 500; test sets 4,000/872/2,500 for AG News/SST-2/IMDB), 64-token sequences, batch 32, and per-dataset word-level vocabularies built on the training subset (AG News 7,234; SST-2 2,249; IMDB 12,000; WikiText-2 5,655 with 96,000 training tokens and 40,000 validation tokens). The replica LM uses d_model = 96, 3 heads (d_k = 32), 2 layers, d_ff = 192, batch 32, 3 epochs; the classification replica keeps the original d_model = 128, 4 heads. Seed allocation is 1001–1025 for the replica core and LM, 1001–1012 for the d_k grid, 1001–1005 for train-to-convergence, and 1001–1003 for the ablations. Every run is recorded as an append-only JSONL record keyed by (phase, dataset, variant, d_k, seed), making the grid resume-safe and auditable; 312 pre-registered runs completed without gaps, plus 27 mask-artifact probes (Section 4.8)."));

children.push(h2("3.7 Stage-3 operating-point protocol"));
children.push(body("Stage 3 manipulates the softmax operating point directly. All arms reuse the replica architecture, matched-seed construction, and JSONL record protocol of Section 3.6; the new mechanisms are four. (i) Divisor power law. For i.i.d. N(0, \u03c3\u00b2) logits over n attended positions, the KL divergence between the attention distributions induced by two divisors is derived to second order in (b\u03c3) (b = 1/c; Appendix S17 of the supplementary material) and validated by Monte Carlo (200,000 replicates per grid point; d_k \u2208 {32, 128}, n \u2208 {64, 128}, \u03c3 \u2208 [0.5, 16]). (ii) Logit gain. A scalar g multiplies q and k each by \u221ag after the output projections and before QK-Norm, so the raw logit variance scales by g while everything else (initialization, data order, optimizer trajectory shape) is held fixed; the gain sweep uses g \u2208 {1, 2, 4, 8, 16, 32}, 4 seeds, 3 variants (72 runs). The regime-restored arms train 6,000-example AG News for 8 epochs with per-epoch operating-point probes at d_k \u2208 {32, 128} (18 runs), and the LM gain arm trains the replica LM at sequence length 128 with g \u2208 {1, 4} (18 runs). (iii) Operating-point probe. Every Stage-3 run records, per layer, the raw pre-division logit SD \u03c3_raw, the realised attention entropy and maximum weight under the model's own divisor, and the KL divergence between the attention maps the same logits would induce under each of the three divisors \u2014 the exact quantity the power law predicts. (iv) Frontier stabilisers and pretrained decoders. The bridge arms add per-head QK-normalisation (LayerNorm on q, k; affine defaults preserve bit-identical matched initialization) and Gemma-2-style attention logit soft-capping (z \u21a6 c\u00b7tanh(z/c), c = 3.0) at gain 32 (20 runs). The frontier probe loads pretrained GPT-2 checkpoints (124M, 355M) with eager attention, replaces only the attention scaling factor (1/\u221ad_k \u2192 1/d_k or 1/(2\u221ad_k); d_k = 64), and evaluates WikiText-2 validation perplexity over 160 non-overlapping 128-token windows with per-layer \u03c3_raw, entropy, and KL diagnostics (6 evaluations). A companion GPU script extends the identical intervention and diagnostics to 1B\u20137B HuggingFace decoders (GPT-2-XL, Pythia, Llama-3.2, Qwen2.5) and is shipped with the repository (scripts/frontier/frontier_gpu_1b7b.py)."));

children.push(h2("3.8 Statistical analysis"));
children.push(body("For each dataset and variant v we compute the paired differences d_i = a_std(s_i) − a_v(s_i), i = 1, …, n, and report the mean difference, the 95% t-based confidence interval"));
children.push(eqCI);
children.push(body("Cohen's d_z for paired designs,", { noIndent: true }));
children.push(eqDZ);
children.push(body("with conventional magnitude labels (negligible |d_z| < 0.2, small, medium, large) [23], the two-sided paired t-test, and the Wilcoxon signed-rank test as a non-parametric confirmation [21]. Because each dataset contributes two comparisons against the common baseline, family-wise error is controlled with the Holm–Bonferroni step-down procedure [22] within each dataset at α = 0.05. Effect sizes and confidence intervals are interpreted with the usual cautions for small samples.", { noIndent: true }));

children.push(h2("3.9 Implementation, validation, and reproducibility"));
children.push(body("The study is implemented in PyTorch with configuration-driven experiment scripts (YAML), a preflight validator that checks the environment and configurations before launch, and 22 unit tests in 5 suites covering: bit-identical initialization across variants for a fixed seed, gradient flow through all variants, output shapes, masked attention correctness, and data-pipeline determinism. Mixed precision is disabled and logits are clamped to [−50, 50] for numerical stability. Every run writes a per-seed JSON record (configuration hash, training history, and test metrics) from which the aggregate tables and figures in this paper are generated by committed analysis scripts. The full package — source, configurations, tests, per-seed records, statistical analysis outputs, figures, and this manuscript — is released under an MIT licence."));

// ===== 4. Results =====
children.push(h1("4. Results"));

children.push(h2("4.1 The completed three-stage programme"));
children.push(body("The experimental programme is now complete and proceeds in three stages. Stage 1 is the original GPU-scale study: a compact pre-LN encoder (d_model = 128, 4 heads, d_k = 32) trained on the full training subsets (10,000 examples, 128-token sequences, batch 64) for 2 epochs, plus the temperature ablation, an extended 25-epoch budget, and the original language-modelling runs. Stage 2, reported in this revision, completes every grid that Stage 1 left open, on a documented replica protocol (2-core CPU sandbox; 2,000-example training subsets, 64-token sequences, batch 32; full specification in Sections 3.6 and S13 of the supplementary material): the 25-seed replica core, the corrected causal-mask language-modelling grid (25 seeds × 3 variants), the head-dimension grid d_k ∈ {32, 64, 128}, train-to-convergence with early stopping, and multi-seed E1–E4 ablations. Stage 3, also reported in this revision, responds to the operating-point diagnosis of Section 5.2: it quantifies the regime in which the divisor can act at all (a power law, validated by Monte Carlo), then deliberately moves the operating point into that regime three ways — an injected logit gain, longer training at two head dimensions, and inference-time divisor swaps in a pretrained decoder — and adds the frontier stabilisers (QK-Norm, soft-capping) under which large logits are tolerated in practice. All stages share the same matched-seed paired design, statistical machinery, and analysis scripts. In total the programme comprises 787 runs: 314 at GPU scale, 339 at replica scale (312 pre-registered completions plus 27 mask-artifact probes), and 134 in the Stage-3 operating-point programme (128 matched-seed training runs plus 6 pretrained-model evaluations)."));
children.push(tableCaption("Completed experiment manifest. Stage 1 = GPU-scale protocol; Stage 2 = replica-scale completion; Stage 3 = operating-point programme (this revision). Every preregistered grid is complete in at least one stage."));
children.push(scholarlyTable(
  ["Grid", "Design", "Stage", "Runs"],
  [
    ["Core classification", "3 variants × 25 seeds × 3 datasets", "1", "225"],
    ["Temperature ablation", "4 temperatures × 3 seeds × 2 datasets", "1", "24"],
    ["Extended budget", "3 variants × 5 seeds × 2 datasets, 25 epochs", "1", "30"],
    ["Core language modelling", "3 variants × 25 seeds (v1 protocol)", "1", "25"],
    ["Focused ablations E1–E4", "single-seed arms, 5 epochs", "1", "10"],
    ["Replica core classification", "3 variants × 25 seeds, AG News", "2", "75"],
    ["Replica language modelling", "3 variants × 25 seeds, corrected causal mask", "2", "75"],
    ["Head-dimension grid", "d_k ∈ {32,64,128} × 3 variants × 12 seeds", "2", "108"],
    ["Train-to-convergence", "3 variants × 5 seeds × 2 datasets, early stopping", "2", "30"],
    ["Replica ablations E1–E4", "4 arms × 3 seeds × 2 datasets, 5 epochs", "2", "24"],
    ["Mask-artifact probes", "leaky vs. causal, 3 and 10 epochs", "2", "27"],
    ["Logit-gain sweep", "g \u2208 {1,2,4,8,16,32} \u00d7 3 variants \u00d7 4 seeds", "3", "72"],
    ["Regime-restored AG News", "6,000 train, 8 epochs, probes, d_k \u2208 {32,128}", "3", "18"],
    ["Frontier-stabiliser bridge", "QK-Norm / soft-cap arms \u00d7 4 seeds", "3", "20"],
    ["LM logit gain", "g \u2208 {1,4} \u00d7 3 variants \u00d7 3 seeds, seq 128", "3", "18"],
    ["Frontier divisor swap", "GPT-2 + GPT-2-medium, inference-time, 3 divisors", "3", "6"],
    ["Total", "", "", "787"],
  ],
  [30, 46, 10, 14]));

children.push(h2("4.2 Stage 1: GPU-scale core comparison"));
children.push(body("Figure 3 and Table 4 present test accuracy across the 25-seed core grid. On AG News, standard scaling reached 91.23% (± 0.25), versus 91.13% (± 0.26) for the temperature variant τ = 2 and 90.92% (± 0.32) for 1/d_k — a strictly ordered pattern in which softening the attention distribution monotonically hurt. On the binary sentiment tasks the ordering vanished: SST-2 accuracies fell within 0.08 points of one another and IMDB accuracies within 0.04 points, with overlapping seed distributions (Figure 3, middle and right)."));
children.push(figure("fig_main_performance.png", 580));
children.push(caption("Core-grid test accuracy (mean ± one standard deviation across 25 matched seeds, 2-epoch budget). Only AG News shows a strictly ordered, visibly separated pattern; on SST-2 and IMDB the seed distributions overlap almost completely."));
children.push(tableCaption("Core-grid results, Stage 1 (25 matched seeds, 2-epoch GPU budget; language-modelling rows are the superseded v1 leaky-mask values, see Section 4.8). ± denotes one standard deviation across seeds; CI is the 95% confidence interval of the mean."));
children.push(scholarlyTable(
  ["Dataset", "Variant", "Mean", "± SD", "95% CI", "n"],
  [
    ["AG News (%)", "Standard", "91.23", "0.25", "[91.13, 91.33]", "25"],
    ["", "Temp (τ=2)", "91.13", "0.26", "[91.03, 91.24]", "25"],
    ["", "1/d_k", "90.92", "0.32", "[90.80, 91.04]", "25"],
    ["SST-2 (%)", "Standard", "80.66", "1.15", "[80.19, 81.13]", "25"],
    ["", "Temp (τ=2)", "80.70", "1.04", "[80.27, 81.13]", "25"],
    ["", "1/d_k", "80.62", "1.06", "[80.18, 81.06]", "25"],
    ["IMDB (%)", "Standard", "81.93", "0.99", "[81.52, 82.34]", "25"],
    ["", "Temp (τ=2)", "81.93", "0.93", "[81.55, 82.31]", "25"],
    ["", "1/d_k", "81.96", "0.91", "[81.58, 82.34]", "25"],
    ["WikiText-2 (ppl ↓, v1 artefact)", "Standard", "1.0917", "0.0061", "[1.089, 1.095]", "15"],
    ["", "Temp (τ=2)", "1.0882", "0.0056", "[1.083, 1.093]", "7"],
    ["", "1/d_k", "1.0809", "0.0021", "[1.077, 1.085]", "3"],
  ],
  [22, 18, 14, 13, 21, 12]));

children.push(h2("4.3 Paired differences and effect sizes (Stage 1)"));
children.push(body("Because the design is seed-matched, the informative statistics are the paired differences of Table 5 and Figure 4. On AG News, standard scaling beat 1/d_k by +0.31 points (95% CI [0.22, 0.39], d_z = 1.52, t(24) = 7.60, p_Holm < 0.001) and beat τ = 2 by +0.10 points (95% CI [0.02, 0.17], d_z = 0.53, t(24) = 2.65, p_Holm = 0.013) — large and medium paired effect sizes respectively [23]. Wilcoxon signed-rank tests confirmed both directions. On SST-2 and IMDB all four comparisons had confidence intervals straddling zero and negligible effect sizes (|d_z| ≤ 0.19, p_Holm ≥ 0.70). In other words, the AG News effect is not an artefact of seed luck: it appears in the matched within-seed contrasts, where initialization and data order are held constant."));
children.push(tableCaption("Paired statistics versus the standard baseline (unit: within-seed difference, n = 25). Positive differences favour standard scaling. p_Holm is Holm–Bonferroni corrected across the two comparisons within each dataset."));
children.push(scholarlyTable(
  ["Dataset", "Comparison", "Δ (pp)", "95% CI (pp)", "d_z", "t(24)", "p_Holm", "Sig."],
  [
    ["AG News", "1/d_k vs. std", "+0.31", "[+0.22, +0.39]", "1.52", "7.60", "<0.001", "***"],
    ["", "τ=2 vs. std", "+0.10", "[+0.02, +0.17]", "0.53", "2.65", "0.013", "*"],
    ["SST-2", "1/d_k vs. std", "+0.04", "[−0.23, +0.31]", "0.06", "0.30", "0.781", ""],
    ["", "τ=2 vs. std", "−0.05", "[−0.31, +0.21]", "−0.07", "−0.35", "0.718", ""],
    ["IMDB", "1/d_k vs. std", "−0.04", "[−0.11, +0.04]", "−0.19", "−0.95", "0.703", ""],
    ["", "τ=2 vs. std", "−0.01", "[−0.06, +0.05]", "−0.04", "−0.20", "0.831", ""],
  ],
  [13, 17, 11, 20, 10, 11, 12, 6]));
children.push(figure("fig_paired_differences.png", 580));
children.push(caption("Seed-matched paired differences with 95% confidence intervals. Only the two AG News comparisons (red) exclude zero after Holm–Bonferroni correction; all SST-2 and IMDB intervals (grey) straddle zero."));

children.push(h2("4.4 Stage 2: replica completion of the core grid"));
children.push(body("The first Stage-2 question is whether the AG News effect of Section 4.3 survives a fivefold reduction in training data and sequence length. Under the replica protocol (2,000 training examples, 64-token sequences, otherwise identical architecture and matched-seed design), all three variants land within 0.01 points of one another: 78.51% (± 1.26) for standard scaling, 78.51% (± 1.25) for τ = 2, and 78.50% (± 1.24) for 1/d_k (Figure 5a, Table 6). The paired differences are −0.01 points (95% CI [−0.04, +0.02], d_z = −0.13) for 1/d_k versus standard and +0.00 points (d_z = 0.04) for τ = 2 versus standard; both p_Holm = 1.00 (Figure 5b). The Stage-1 AG News effect of +0.31 points therefore does not replicate at this operating point: the replica confidence interval excludes any advantage larger than 0.02 points. We return to the interpretation of this scale dependence in Section 5.2."));
children.push(figure("fig_stage2_core.png", 580));
children.push(caption("Stage-2 replica core on AG News (25 matched seeds, 2 epochs). (a) Test accuracy by variant. (b) Within-seed paired differences with 95% CIs; both intervals straddle zero, and the replica CI excludes the Stage-1 full-scale effect of +0.31 points for 1/d_k."));
children.push(tableCaption("Stage-2 replica results: AG News core (25 seeds, 2 epochs) with within-seed paired statistics versus standard (n = 25); p_Holm across the two comparisons."));
children.push(scholarlyTable(
  ["Variant", "Mean (%)", "± SD", "n", "Paired vs. standard"],
  [
    ["Standard", "78.51", "1.26", "25", "—"],
    ["Temp (τ=2)", "78.51", "1.25", "25", "Δ = +0.00 pp, d_z = 0.04, p_Holm = 1.00"],
    ["1/d_k", "78.50", "1.24", "25", "Δ = −0.01 pp, d_z = −0.13, p_Holm = 1.00"],
  ],
  [16, 13, 11, 8, 52]));

children.push(h2("4.5 Stage 2: head-dimension grid (d_k ∈ {32, 64, 128})"));
children.push(body("The variance argument behind √d_k [1] predicts that the logit standard deviation grows as √d_k when the divisor is held at its standard value but as d_k when the divisor is 1/d_k; the mismatch between the two scalings should therefore amplify with head dimension, making larger d_k the regime where the 2017 default is most likely to matter. We tested this directly on AG News with d_model = 128 and n_heads ∈ {4, 2, 1}, giving d_k ∈ {32, 64, 128}, 3 variants × 12 seeds per point (Table 7, Figure 6). The prediction is not borne out at this operating point: the 1/d_k-versus-standard gap is +0.010 points (p = 0.63) at d_k = 32, +0.008 points (p = 0.75) at d_k = 64, and +0.017 points (p = 0.56) at d_k = 128, and the paired change in the gap between d_k = 128 and d_k = 32 is −0.006 points (95% CI [−0.077, +0.064], p = 0.85). The same holds for τ = 2. The mechanism is measurable: at this scale the trained models' raw logit standard deviation is of order 1 (1.02–1.19 across variants; Section 4.9), so after division by √d_k ∈ {5.7, 8, 11.3} or by d_k ∈ {32, 64, 128} every variant operates in the softmax's near-uniform regime, where rescaling the logits changes almost nothing downstream. The theoretical amplification only bites once raw logit variance grows past the divisor, which the Stage-1 full-data protocol approaches but the replica does not."));
children.push(figure("fig_dk_grid.png", 580));
children.push(caption("Head-dimension grid on AG News (12 matched seeds per point). (a) Test accuracy versus d_k for the three variants. (b) Within-seed gap versus standard with 95% CIs; no trend with d_k (trend p = 0.85)."));
children.push(tableCaption("Head-dimension grid, AG News replica (mean ± SD over 12 seeds; within-seed paired differences versus standard)."));
children.push(scholarlyTable(
  ["d_k", "Std (%)", "τ=2 (%)", "1/d_k (%)", "Δ(τ=2) (pp)", "Δ(1/d_k) (pp)"],
  [
    ["32", "78.82 ± 0.92", "78.83 ± 0.91", "78.83 ± 0.89", "+0.01 (p=0.52)", "+0.01 (p=0.63)"],
    ["64", "78.83 ± 0.90", "78.82 ± 0.89", "78.84 ± 0.89", "−0.01 (p=0.63)", "+0.01 (p=0.75)"],
    ["128", "78.81 ± 0.84", "78.85 ± 0.87", "78.83 ± 0.89", "+0.04 (p=0.12)", "+0.02 (p=0.56)"],
  ],
  [8, 18, 18, 18, 19, 19]));

children.push(h2("4.6 Temperature ablation (Stage 1)"));
children.push(body("The preregistered temperature sweep (τ ∈ {0.5, 1.0, 1.5, 2.0}, 3 seeds per point, 2 epochs) probes the direction of the effect around the standard point (Figure 7, Table 8). On SST-2, sharper attention was monotonically (if noisily) better: τ = 0.5 reached 82.31% versus 80.77% for τ = 2 — a 1.54-point spread — and on IMDB τ = 0.5 again produced the highest mean (82.29%), although the τ-response was flatter. Standard scaling (τ = 1) was intermediate on both datasets. These 3-seed observations are exploratory: confidence intervals are wide (up to ±1.04 points on SST-2 τ = 1.5), so the sharpest temperature's lead over the 25-seed core-grid baselines (about +1.6 points on SST-2) should be treated as a promising direction rather than an established effect. The directional pattern is nonetheless consistent with the distillation literature, where lower temperatures sharpen attention and can help tasks that require focused token selection [3, 2]."));
children.push(figure("fig_temperature_ablation.png", 560));
children.push(caption("Temperature ablation (n = 3 seeds per point, 2 epochs; error bars: ± one standard deviation). Dashed reference lines show the 25-seed core standard-scaling result at τ = 1."));
children.push(tableCaption("Temperature ablation results (mean ± SD across 3 seeds)."));
children.push(scholarlyTable(
  ["τ", "SST-2 (%)", "SD", "IMDB (%)", "SD"],
  [
    ["0.5", "82.31", "0.21", "82.29", "0.35"],
    ["1.0", "82.00", "0.10", "81.88", "0.27"],
    ["1.5", "81.96", "1.04", "81.97", "0.12"],
    ["2.0", "80.77", "0.39", "82.03", "0.36"],
  ],
  [20, 22, 18, 22, 18]));

children.push(h2("4.7 Training-budget sensitivity and train-to-convergence"));
children.push(body("If the AG News-scale differences reflect early-training dynamics rather than a permanent quality gap, they should shrink with longer training. The Stage-1 extended grid (5 seeds, 25 epochs, checkpoints at 2/5/10/25) supports this on the binary tasks: all three variants converge to approximately 84% on SST-2 and 84.5% on IMDB, and the variant differences at 25 epochs (≤ 0.2 points) are within seed noise (Figure 8). The variants' curves are nearly parallel and interleave across checkpoints — underpowered (5 seeds) but consistent with the early-training interpretation, and with the observation of Zhai et al. [4] that attention-entropy differences between configurations diminish as training progresses."));
children.push(figure("fig_training_curves.png", 560));
children.push(caption("Extended-budget training, Stage 1 (5 seeds, checkpoints at 2/5/10/25 epochs). Variant gaps visible at 2 epochs shrink to seed-noise level by 25 epochs on both binary datasets."));
children.push(body("Stage 2 replaces the fixed 2-epoch cut-off with train-to-convergence: 5 seeds per variant on SST-2 and IMDB, up to 20 epochs, model selection by validation loss with a patience of 3 epochs (Figure 9, Table 9). Validation loss bottoms at epoch ≈ 2 for every variant on both datasets (mean best epoch 2.2 on SST-2 and 1.6 on IMDB), early stopping halts training at epochs 5–6, and the best-validation checkpoints differ by at most +0.07 points between variants (p ≥ 0.31). The learning curves of the three variants are visually indistinguishable at every epoch: neither the asymptotic accuracy nor the convergence speed depends on the scaling function in this regime. The answer to whether the divisor affects asymptote or speed is therefore neither — at this operating point the entire training trajectory is divisor-invariant."));
children.push(figure("fig_convergence.png", 560));
children.push(caption("Stage-2 train-to-convergence (5 matched seeds per variant, mean test accuracy per epoch; maximum 20 epochs with early stopping on validation loss, patience 3). The three variants trace indistinguishable trajectories; dashed lines mark the mean early-stopping epoch."));
children.push(tableCaption("Train-to-convergence with early stopping (Stage 2; mean ± SD over 5 seeds). Best epoch = epoch of lowest validation loss; stop epoch = 5–6 for all runs."));
children.push(scholarlyTable(
  ["Dataset", "Variant", "Best-epoch acc. (%)", "Best epoch", "p vs. std"],
  [
    ["SST-2", "Standard", "67.71 ± 2.07", "2.2", "—"],
    ["", "Temp (τ=2)", "67.78 ± 2.14", "2.2", "0.31"],
    ["", "1/d_k", "67.78 ± 2.07", "2.2", "0.43"],
    ["IMDB", "Standard", "67.16 ± 2.84", "1.6", "—"],
    ["", "Temp (τ=2)", "67.18 ± 2.93", "1.6", "0.79"],
    ["", "1/d_k", "67.22 ± 3.00", "1.6", "0.58"],
  ],
  [14, 20, 26, 18, 22]));

children.push(h2("4.8 Language modelling on WikiText-2 (corrected causal mask)"));
children.push(body("Stage 1's language-modelling runs used the v1 training script, whose attention mask did not enforce causality (a bidirectional mask with a shifted next-token loss), and produced validation perplexities of 1.08–1.09. Those values are reported here only as the historical record of an artefact: perplexity below the uniform-unigram floor of this vocabulary is physically impossible for a causal model, and this revision supersedes them."));
children.push(body("The corrected grid implements the standard autoregressive factorisation: a lower-triangular mask ANDed with the key-padding mask, applied after the logit clamp (so masked weights are exactly zero), with the next-token target shifted by one position. Five verification tests pin the implementation down (supplementary Section S15): attention to future positions is exactly 0.0, attention to padding positions is exactly 0.0, attention rows sum to 1 over allowed positions, the diagonal is open on all real-token rows (consistent with predicting x_(t+1) from x_(≤t)), and −∞ survives the clamp."));
children.push(body("Under the corrected protocol, all 25 seeds × 3 variants completed: validation perplexity is 872.8 (± 29.7) for standard scaling, 872.8 (± 29.6) for 1/d_k, and 872.8 (± 29.7) for τ = 2 (Figure 10a, Table 10). The paired differences are +0.01 ppl (95% CI [−0.22, +0.23], d_z = 0.01, p_Holm = 1.00) for 1/d_k versus standard and +0.00 ppl for τ = 2: with statistical power now matched across arms (25 seeds each), there is no detectable divisor effect on language modelling at this scale. A 10-epoch convergence probe (3 seeds per variant) lowers perplexity to ≈ 268 and the three variants remain within 1.6 ppl of one another."));
children.push(body("To characterise the v1 artefact rather than merely discard it, Stage 2 includes leak probes: identical LM runs with the causal mask deliberately removed (bidirectional attention, as in v1). The leak lowers perplexity in every matched cell — 852.5 vs. 872.8 at 3 epochs and ≈ 252 vs. ≈ 268 at 10 epochs (Figure 10b) — confirming the direction of the bias. At replica budgets the exploited leak is worth only a few percent because attention stays near-uniform (Section 4.5); the v1 values near 1.08 correspond to a fully exploited leak (attention collapsed onto the target position) at the 25,000-step GPU budget, where the model has enough training to learn to copy its input."));
children.push(figure("fig_lm_corrected.png", 580));
children.push(caption("Corrected language-modelling results. (a) Validation perplexity with the standard causal mask (25 seeds per variant; the v1 leaky-mask values of 1.08–1.09 are marked for scale). (b) Mask-artifact probes: leaky masks lower perplexity in every matched cell (3 seeds per cell)."));
children.push(tableCaption("Corrected WikiText-2 language modelling (validation perplexity, mean ± SD over 25 matched seeds, 3 epochs; paired statistics versus standard, p_Holm across two comparisons)."));
children.push(scholarlyTable(
  ["Variant", "PPL", "± SD", "Δ vs. std (ppl)", "p_Holm"],
  [
    ["Standard", "872.8", "29.7", "—", "—"],
    ["Temp (τ=2)", "872.8", "29.7", "+0.00 [−0.14, +0.15]", "1.00"],
    ["1/d_k", "872.8", "29.6", "+0.01 [−0.22, +0.23]", "1.00"],
  ],
  [20, 16, 14, 32, 18]));

children.push(h2("4.9 Attention diagnostics"));
children.push(body("The mechanistic measurements (seed 0, epoch 1) trace the causal chain from the scaling constant to the attention distribution and its gradients (Figure 11). Mean attention entropy rose monotonically with s (2.14 nats for standard, 2.31 for τ = 2, 2.48 for 1/d_k), while the maximum attention weight fell (0.32, 0.28, 0.24): larger divisors produce flatter, less selective attention [4]. Inter-head cosine similarity of the attention distributions rose in lockstep (0.23, 0.31, 0.42) — softer scaling makes heads more alike, consistent with the head-specialisation literature [16, 17]. Query and key gradient norms scaled with 1/s as theory predicts for the dot-product pipeline: relative to standard, observed Q-gradient ratios were 0.50 (theory 0.50) for τ = 2 and 0.26 (theory 0.18) for 1/d_k, while value gradients stayed flat (ratios 1.03 and 0.97), confirming that the scaling constant modulates exactly the ∂L/∂Q, ∂L/∂K pathway (Figure 12)."));
children.push(body("The Stage-2 replica diagnostics explain why the outcome metrics stop separating at reduced scale (Section 4.5). Measured on the trained replica models (seed 1001): attention entropy is 4.048–4.049 nats across all three variants (the causal-horizon maximum is ln 64 ≈ 4.16), maximum attention weight is 0.025–0.030, no logit reaches the ±50 clamp, and the raw pre-division logit standard deviation on WikiText-2 is 1.02 (standard), 1.19 (1/d_k), and 1.17 (τ = 2) — i.e. after division every variant sits in the softmax's near-uniform regime. The gradient pathway is correspondingly attenuated: ‖∂L/∂Q‖ + ‖∂L/∂K‖ is 0.048 (standard), 0.029 (τ = 2), and 0.009 (1/d_k), against value gradients of 1.6–1.9 — a 35–200× imbalance — and under AdamW's per-parameter scale normalisation the residual Q/K gradient differences largely wash out. The mechanism live at GPU scale (ordered entropies 2.14/2.31/2.48) is quantitatively muted at replica scale (4.05/4.05/4.05)."));
children.push(figure("fig_head_diversity.png", 580));
children.push(caption("Attention diagnostics at epoch 1 (seed 0): mean attention entropy, maximum attention weight, and inter-head cosine similarity of attention distributions. Larger divisors produce flatter, more similar heads."));
children.push(figure("fig_gradient_ratios.png", 560));
children.push(caption("Gradient diagnostics (epoch 1, batch 0). (a) Absolute ∂L/∂Q, ∂L/∂K, ∂L/∂V norms by variant. (b) Observed Q/K gradient ratios relative to standard versus the theoretical 1/s prediction (dashed lines)."));

children.push(h2("4.10 Stage 3: the divisor power law"));
children.push(body("The replica nulls of Sections 4.4\u20134.8 and the diagnostics above motivate a quantitative statement of when the divisor can matter at all. Let the pre-softmax logits of one attention row be approximately i.i.d. with standard deviation \u03c3 (the raw, pre-division scale), and let two divisors c1 > c2 induce inverse-scale parameters b_i = 1/c_i. A second-order expansion of the softmax (Appendix S17 of the supplementary material) gives KL(P_b1 || P_b2) \u2248 \u00bd(b1 \u2212 b2)\u00b2\u03c3\u00b2(1 \u2212 1/n), where n is the number of attended positions: the divergence between the attention distributions induced by two divisors grows quadratically in the raw logit scale, independently of everything else. Monte Carlo simulation (200,000 replicates per point) confirms the law to within a few percent for \u03c3 \u2264 4 and tracks the measured divergence up to \u03c3 \u2248 8 before attention saturation flattens the growth (Figure 13). The same expansion yields bite thresholds: at d_k = 32, n = 64, the standard-versus-1/d_k contrast reaches KL = 0.01 nats only at \u03c3 \u2248 0.98 and KL = 0.1 nats at \u03c3 \u2248 3.1; for the standard-versus-\u03c4 contrast the corresponding values are \u03c3 \u2248 1.6 and 5.1. Below those operating points the divisors are mathematically near-indistinguishable, whatever the downstream task."));
children.push(body("This single formula organises every result in this paper. The replica grids sit at \u03c3 \u2248 0.3\u20131.1, i.e. at KL \u2264 0.006 nats \u2014 deep in the inert region (Figure 13, right panel). The Stage-1 AG News runs, whose diagnostics put raw logit variance closest to the divisor, sit nearest the bite threshold. And the pretrained decoders examined below sit at \u03c3 \u2248 45\u201363, where the measured standard-versus-1/d_k divergence is 1.4\u20131.5 nats per row \u2014 four orders of magnitude away from the replica nulls. Claims about scaling functions are therefore claims about \u03c3, and \u03c3 is measurable in one forward pass."));
children.push(figure("fig_regime_law.png", 580));
children.push(caption("The divisor power law. (a) Monte Carlo KL divergence between attention distributions induced by the three divisor pairs (d_k = 32, n = 64) against the quadratic prediction. (b) Attention entropy versus raw logit scale \u03c3 for each divisor; the shaded band marks the replica operating point, where all divisors sit within 0.015 nats of the uniform ceiling."));

children.push(h2("4.11 Stage 3: moving the operating point"));
children.push(body("If the replica nulls are operating-point effects, injecting logit variance should resurrect the divisor contrast without changing anything else. The logit-gain sweep multiplies q and k each by \u221ag (g \u2208 {1, 2, 4, 8, 16, 32}) after the output projections and before QK-Norm, holding data, initialization, and optimization fixed; 72 matched runs (4 seeds \u00d7 3 variants \u00d7 6 gains; Table 11, Figure 14). The manipulation works exactly as the law predicts: measured \u03c3_raw grows from 0.33 (g = 1) to 9.21 (g = 32), attention entropy under the standard divisor falls from 3.78 to 2.81 nats, and the measured standard-versus-1/d_k KL grows from 0.0009 to 0.64 nats. The outcome metric follows: at g = 32 the standard divisor drops to 76.81% while 1/d_k and \u03c4 = 2 hold at 78.2\u201378.3%, a paired deficit of \u22121.40 points (d_z = \u22121.76, p = 0.039 uncorrected; not significant after Holm correction across the twelve gain-level comparisons, n = 4 seeds per arm). The direction of the crossover is the opposite of the Stage-1 full-scale AG News pattern: forced into the sharp-attention regime from a random start, the 2017 default is the variant that hurts, consistent with the entropy-collapse literature [4] and with the QK-norm rationale of frontier architectures. At small and moderate gains (g \u2264 16, KL \u2264 0.20) accuracy is flat, tracing the same inert plateau the replica grids sit on."));
children.push(body("The regime-restored runs then ask whether ordinary training can drift into the bite region unaided: 18 instrumented runs (3 seeds, 6,000 training examples, 8 epochs \u2014 12\u00d7 the replica step budget \u2014 with per-epoch operating-point probes) at d_k = 32 (4 heads) and d_k = 128 (1 head). At d_k = 32 the operating point crawls from \u03c3 = 0.42 to only \u03c3 = 1.11 over the whole budget (KL \u2248 0.006, still 16\u00d7 below the 0.1-nat threshold) and the three variants remain within 0.2 points at every epoch (final paired contrasts: \u22120.06 vs. 1/d_k, p = 0.51; \u22120.13 vs. \u03c4 = 2, p = 0.42). The inertia is real: at this scale, longer training alone does not escape the near-uniform regime. At d_k = 128, however, the raw logits grow much faster (\u03c3 = 1.46 \u2192 4.6\u20135.3) and the standard-versus-1/d_k divergence rises 200\u00d7 to \u2248 0.086 nats; the Stage-1 ordering re-emerges in direction (standard beats 1/d_k by +0.42 points, p = 0.31, and \u03c4 = 2 by +0.52 points, p = 0.066; n = 3, uncorrected). Language modelling at sequence length 128 with the same gain knob (18 runs) stays inert at g = 4 (\u03c3 = 1.15, KL = 0.005; perplexity 2160.3\u20132160.8 across all variants), confirming that the replica LM null is likewise an operating-point fact."));
children.push(tableCaption("Stage-3 logit-gain sweep (AG News replica, 4 matched seeds, 2 epochs). \u03c3_raw: measured raw pre-division logit SD; H: attention entropy under the standard divisor; KL: measured KL(P_std || P_1/d_k) on the same logits; d_z: paired effect size (standard minus 1/d_k) in test accuracy. The g = 32 deficit is nominally significant (p = 0.039) but not after Holm correction."));
children.push(scholarlyTable(
  ["g", "\u03c3_raw", "H (nats)", "KL (nats)", "Std (%)", "1/d_k (%)", "\u03c4=2 (%)", "d_z"],
  [
    ["1", "0.33", "3.78", "0.0009", "78.22", "78.24", "78.24", "\u22120.14"],
    ["2", "0.65", "3.78", "0.0037", "78.24", "78.24", "78.22", "\u22120.06"],
    ["4", "1.29", "3.76", "0.0145", "78.19", "78.20", "78.24", "\u22120.04"],
    ["8", "2.53", "3.70", "0.0558", "78.36", "78.23", "78.19", "+0.61"],
    ["16", "4.87", "3.48", "0.2046", "78.34", "78.21", "78.36", "+0.30"],
    ["32", "9.21", "2.81", "0.6418", "76.81", "78.21", "78.34", "\u22121.76"],
  ],
  [8, 12, 13, 13, 14, 14, 13, 13]));
children.push(figure("fig_gain_sweep.png", 580));
children.push(caption("Stage-3 logit-gain sweep. (a) The injected gain moves the operating point (\u03c3_raw) and, with it, the measured divisor contrast (KL). (b) Test accuracy versus gain: flat while the softmax is near uniform (g \u2264 16), with the standard divisor dropping 1.4 points below both alternatives once the contrast is large (g = 32)."));

children.push(h2("4.12 Stage 3: frontier stabilisers and a pretrained decoder"));
children.push(body("Large logits are not hypothetical: they are the operating point at which frontier models actually run, which is why QK-normalisation and logit soft-capping exist. Two bridge arms (20 matched runs) test whether the divisor contrast survives inside those stabiliser contexts. With per-head QK-Norm the operating point is pinned at \u03c3_post \u2248 1.05 regardless of the divisor, and the three stabilised variants again agree to 0.12 points (78.16\u201378.28%) even though their realised attention distributions differ by KL \u2248 0.29 nats \u2014 at this dose the outcome metric is still insensitive. With Gemma-2-style soft-capping (c = 3.0) at gain 32, the standard divisor's 1.4-point deficit reappears unchanged (76.81% vs. 78.21% for 1/d_k): capping bounds the logits but does not restore the near-uniform plateau, and the divisor contrast persists under the stabiliser."));
children.push(body("Finally, the frontier operating point itself is probed with the largest decoders the CPU sandbox can execute: GPT-2 (124M) and GPT-2-medium (355M), pretrained with the standard divisor, evaluated on WikiText-2 validation with only the divisor swapped at inference (identical weights, tokens, and protocol; 160 non-overlapping 128-token windows; Table 12, Figure 15). These models sit four orders of magnitude beyond the replica in raw logit scale (\u03c3_raw \u2248 63 and 45; after the training divisor, \u03c3_post \u2248 7.8 and 5.6 \u2014 far outside the near-uniform regime). The consequence is drastic: swapping to \u03c4 = 2 degrades perplexity by +54.5% (GPT-2) and +26.5% (GPT-2-medium), and swapping to 1/d_k is catastrophic (+9,253% and +3,990%; perplexity 51.2 \u2192 4,792). The mechanism matches the law: under 1/d_k attention collapses towards uniformity (per-row entropy rises from 1.4\u20132.9 to 3.6\u20133.9 nats), and the measured per-row standard-versus-1/d_k divergence is 1.4\u20131.5 nats. The divisor that is empirically inert at every replica-scale operating point we can construct is therefore decisively not inert at the pretrained scale \u2014 one parameter-free constant, with no retraining, separates 51 from 4,792 perplexity. For the regime between GPT-2-medium and 7B we ship a ready-to-run GPU script (scripts/frontier/frontier_gpu_1b7b.py) that applies the same divisor swap (plus per-layer diagnostics) to any HuggingFace decoder (GPT-2-XL, Pythia-1.4B/2.8B/6.9B, Llama-3.2, Qwen2.5); its GPT-2 output reproduces Table 12 exactly, so the 1B\u20137B cells of this table are one GPU-hour away for any reader."));
children.push(tableCaption("Stage-3 frontier probe: inference-time divisor swap in pretrained GPT-2 decoders (WikiText-2 validation, 160 \u00d7 128 tokens, non-overlapping windows, no retraining). \u0394 is relative to the training-time divisor (standard). Absolute perplexity is protocol-dependent; the contrasts are not."));
children.push(scholarlyTable(
  ["Model", "\u03c3_raw", "Standard", "Temp (\u03c4=2)", "1/d_k"],
  [
    ["GPT-2 (124M)", "62.6", "51.23", "79.13 (+54.5%)", "4,791.65 (+9,253%)"],
    ["GPT-2-medium (355M)", "44.8", "38.77", "49.06 (+26.5%)", "1,585.70 (+3,990%)"],
  ],
  [24, 12, 18, 22, 24]));
children.push(figure("fig_frontier.png", 580));
children.push(caption("The frontier operating point. (a) Per-layer raw and post-divisor logit SD in GPT-2 (124M): the training divisor leaves \u03c3_post \u2248 8, far outside the near-uniform regime the replica occupies. (b) WikiText-2 perplexity when only the divisor is swapped at inference (log scale): \u03c4 = 2 costs 27\u201355%; 1/d_k is catastrophic (40\u201390\u00d7)."));

children.push(h2("4.13 Computational efficiency"));
children.push(body("RQ2 receives a clear answer: no measurable efficiency difference exists (Figure 16, Table 13). Parameter counts are identical by construction; measured epoch wall-clock times differ by at most 0.6% on SST-2 and 1.0% on AG News, peak memory is the same 5.2 GB, and inference latency differs by one rounding step. The Stage-2 replica runs agree: mean per-run wall-clock on the 2-core CPU sandbox is 19–25 s (classification) and 29–44 s (language modelling) with no variant ordering, and the divisors add no arithmetic by construction. The scaling constant is, as designed, free."));
children.push(figure("fig_efficiency.png", 560));
children.push(caption("Computational efficiency. (a) Training wall-clock time per epoch. (b) Inference latency for 1,000 samples. Peak memory (5.2 GB) and parameter counts are identical across variants."));
children.push(tableCaption("Efficiency summary (identical parameter counts by construction)."));
children.push(scholarlyTable(
  ["Metric", "Standard", "Temp (τ=2)", "1/d_k"],
  [
    ["Parameters (classification / LM)", "2,695,556 / 4,120,576", "identical", "identical"],
    ["Epoch time, SST-2 (s)", "32.1", "32.3", "32.0"],
    ["Epoch time, AG News (s)", "59.8", "60.1", "59.5"],
    ["Peak GPU memory (GB)", "5.2", "5.2", "5.2"],
    ["Inference latency, 1k samples (s)", "0.84", "0.85", "0.84"],
  ],
  [36, 26, 19, 19]));

children.push(h2("4.14 Focused ablations E1–E4"));
children.push(body("Two mechanistic rescue hypotheses were probed with single-seed arms at 5 epochs (full records in the supplementary material, Section S8). The mixed-scale variant — two standard heads and two 1/d_k heads — outperformed standard by +1.6 points on SST-2 (82.22% vs. 80.62%) but was indistinguishable on IMDB (+0.14 points), suggesting per-head scale diversity itself can be useful on some tasks. Learning-rate compensation for 1/d_k (multiplying Q/K learning rates by √d_k) failed to close the gap on either dataset and slightly worsened it (SST-2: 81.08% vs. 81.42% uncompensated; IMDB: 81.57% vs. 81.45%), indicating that the deficit of 1/d_k is not simply a rescaled effective learning rate that gradient magnitudes alone would predict. Both results are single-seed and should be read as hypothesis-generating."));
children.push(body("The Stage-2 replica repeats both ablations with 3 matched seeds per arm at 5 epochs (supplementary Table S14). The multi-seed mixed-scale result is null: Δ = +0.04 points on SST-2 (p = 0.81) and +0.05 points on IMDB (p = 0.46) versus standard, so the Stage-1 single-seed +1.6-point E1 excursion is not reproducible. Learning-rate compensation remains neutral-to-harmful in direction (−1.30 points on SST-2, p = 0.11; −0.27 points on IMDB, p = 0.72), consistent with the Stage-1 conclusion that the divisor's effect (where present) is not a plain learning-rate rescaling."));

// ===== 5. Discussion =====
children.push(h1("5. Discussion"));
children.push(h2("5.1 Principal interpretation"));
children.push(body("The central finding is a pattern of conditionality, now mapped along three axes. Along the task axis, the standard √d_k divisor is significantly better than both alternatives on AG News at a 2-epoch budget — with large and medium paired effect sizes — yet statistically indistinguishable from them on two binary sentiment datasets. Along the budget axis, its advantage on binary tasks dissolves by 25 epochs, and the Stage-2 replica completion shows that the AG News effect itself does not survive a fivefold reduction in data and sequence length: under matched seeds the entire replica grid (core, d_k grid, convergence training, corrected language modelling) is variant-invariant to within measurement resolution. Along the operating-point axis — the axis Stage 3 adds — the nulls themselves are regime facts: a quadratic power law on the raw logit scale predicts that the divisor contrast is inert below \u03c3 \u2248 1 and decisive above it, the replica grids sit at \u03c3 \u2248 0.3\u20131.1 (measured KL \u2264 0.006 nats), forcing \u03c3 an order of magnitude higher flips the sign of the effect at g = 32, and a pretrained GPT-2 sitting at \u03c3 \u2248 45\u201363 loses a factor of 94 in perplexity when the same constant is swapped to 1/d_k at inference. A one-line architectural constant inherited from [1] is therefore best understood not as a universally optimal choice but as a default whose benefit is confined to a specific operating point — full data scale, early in training, on multi-class topic classification — and whose influence vanishes once the softmax operates near uniformity, while at frontier operating points it is the alternatives that become untenable."));
children.push(h2("5.2 Scale dependence and the near-uniform regime"));
children.push(body("The Stage-2 null results are not a failure of the manipulation: the diagnostics show the mechanism is present but quantitatively muted. At replica scale the raw pre-division logit standard deviation is of order 1, so after dividing by √d_k (5.7–11.3) or d_k (32–128) every variant's logits sit well inside the softmax's linear (near-uniform) regime, attention entropy is pinned at 4.05 of a possible 4.16 nats for all variants, and the Q/K gradient pathway carries two orders of magnitude less signal than the value pathway — an imbalance that AdamW's per-parameter scale normalisation then equalises away. Two consequences follow. First, the variance argument for √d_k is a statement about logit variance relative to the softmax's dynamic range; it predicts differences only in regimes where the unscaled logit variance is comparable to the divisor, which small-data, short-budget runs do not reach. Second, claims about scaling functions are claims about an operating point, and should be reported together with the logit scale that makes the divisor bite. Stage 3 turns this diagnosis from an explanation into a measurement: the power law quantifies the bite region; the gain sweep shows the outcome metric responding on cue once the operating point enters it (flat to g \u2264 16, \u22121.4 points for the standard divisor at g = 32); the regime-restored runs show that at d_k = 32 even 12\u00d7 the step budget leaves \u03c3 at 1.1 (inertia is real), while at d_k = 128 the larger head drives \u03c3 to \u2248 5 and the Stage-1 ordering begins to re-emerge; and the pretrained decoder probe shows the same constant being decisive (+27 to +9,253% perplexity) at GPT-2's native operating point. The Stage-1 AG News effect, observed precisely where raw logit variance comes closest to the divisor, remains plausible under this account — but until it is reproduced at full scale it should be treated as provisional rather than established."));
children.push(h2("5.3 Why would AG News care about the divisor?"));
children.push(body("Three observations combine into a coherent account. First, the variance argument behind √d_k [1] implies that the 1/d_k divisor shrinks the logit standard deviation by a further factor of √d_k ≈ 5.7; at initialization the attention logits of the 1/d_k variant have a standard deviation of order 1/√d_k ≈ 0.18 rather than 1, pushing the softmax towards uniformity. Second, the measured diagnostics match this account: attention entropy rises and maximum attention weight falls monotonically with s (Section 4.9). Third, AG News is the task where failure-to-differentiate should bite hardest: it is 4-class, short-document, and topic-level, so early error correction plausibly benefits from a few sharp, content-word attention links, whereas binary sentiment over long reviews can be solved with diffuse pooling-like attention. The inter-head cosine similarity trend (0.23 → 0.42 as s grows) adds that flatter scaling also erodes head diversity [16, 17], reducing the ensemble capacity that multi-class discrimination can draw on. The mixed-scale ablation (E1), where keeping two sharp heads helped on SST-2, hints that diversity per se — not the global constant — is the operative resource."));
children.push(h2("5.4 Sharper attention as a free improvement direction"));
children.push(body("The temperature ablation points in the opposite direction: on both sentiment datasets the sharpest tested temperature (τ = 0.5, i.e. doubling the logit scale relative to standard) produced the highest means, by about +1.6 points on SST-2. This aligns with the distillation literature on attention temperature [3, 2] and with Scalable-Softmax's motivation [6] that the effective temperature should not be treated as a universal constant. Given that temperature is parameter-free and computationally free (Section 4.13), and that the 2-epoch AG News result shows temperature effects can reach significance under matched conditions, the practical recommendation is straightforward: when training small transformers on a new task, sweep τ around 1 (e.g. {0.5, 1, 2}) under a seed-matched design before freezing the architecture. The cost is negligible; the present data suggest the yield is real, if task-dependent."));
children.push(h2("5.5 The early-training effect and its decay"));
children.push(body("On the binary tasks, extended training erased the differences (Section 4.7), and the Stage-2 convergence runs sharpen the picture: with model selection by validation loss, every variant's validation loss bottoms at the same epoch (≈ 2 on SST-2, ≈ 1.6 on IMDB), early stopping halts at epochs 5–6, and the learning curves are indistinguishable at every epoch. Combined with the entropy-collapse literature [4, 5], a natural reading is that the divisor matters most while the attention distribution is still being shaped; once representations mature, the network can compensate for the initial sharpness of its attention. The learning-rate compensation ablations (E2/E4) add a useful negative result: the 1/d_k deficit is not equivalent to a smaller effective learning rate that a per-block LR rescaling can restore — and the multi-seed replica ablations confirm the single-seed pattern (LR compensation neutral-to-harmful, mixed-scale heads neutral), while failing to reproduce the Stage-1 E1 excursion of +1.6 points for mixed-scale heads on SST-2."));
children.push(h2("5.6 Implications for experimental practice"));
children.push(body("Methodologically, this study is a demonstration that cheap architectural comparisons deserve the seed-matched treatment. Between-run comparisons at 2 epochs on these datasets would need to overcome standard deviations of about 1 point (SST-2, IMDB) with between-run variance; the paired design recovers resolvable effects from 25 seeds even where the raw means differ by only 0.1–0.3 points (AG News). Given the prevalence of single-seed architectural claims in the scaling literature [9, 11, 12], the protocol used here — bit-identical initialization, shared data order, paired tests, Holm correction, effect sizes — is directly reusable by practitioners evaluating any parameter-free modification (temperature, biases, clamps, positional offsets) and costs no additional compute beyond a handful of extra seeds."));
children.push(h2("5.7 Threats to validity"));
children.push(body("Beyond the enumerated limitations, three threats deserve note. Construct: test accuracy on subsampled training sets may not capture the behaviours that matter at scale; we mitigate by also measuring perplexity and mechanistic diagnostics, and by reporting both a full-scale and a replica stage. Internal: all variants share code paths except one division — the manipulation is auditable in a single line — and unit tests plus the Stage-2 mask-verification suite pin down initialization identity and exact-zero masked attention, so the internal threat is minimal. The Stage-2 null results are consistent across five independent grids, which argues against a wiring error: a variant applied to the wrong tensor would not produce identical entropy and identical accuracies and the Stage-1 gradient-ratio signatures. External: results at d_model = 128, 2 layers, and 64–128-token sequences may not extrapolate to large models; the Stage-3 programme addresses this directly — the power law makes the extrapolation quantitative, the pretrained-decoder probe confirms its direction and magnitude at 124M–355M scale, and the shipped GPU script extends the identical protocol to 1B–7B, but the trained (not inference-only) behaviour of models in that range remains to be measured."));

// ===== 6. Limitations =====
children.push(h1("6. Limitations"));
children.push(body("The study has five principal limitations."));
children.push(numbered("Replica scale. Stage 2 documents a deliberate fivefold reduction in data (2,000 vs. 10,000 examples) and sequence length (64 vs. 128), with a 96k-token LM corpus. The replica null results are statements about that operating point; they bound the Stage-1 AG News effect's generality but cannot confirm or refute it at full scale.", "lim"));
children.push(numbered("Model scale. A 2-layer, d_model = 128 encoder is three orders of magnitude smaller than frontier models. Stage 3 narrows this gap as far as the CPU sandbox allows: the operating-point law is validated in simulation, the gain sweep and regime-restored runs traverse the bite region from below, frontier stabilisers are tested in context, and pretrained GPT-2 (124M, 355M) is probed at its native operating point, where the divisor contrast is enormous. What remains untested is trained behaviour at 1B–7B: the shipped GPU script covers the inference-time intervention and diagnostics, but fine-tuning or training at those scales, and the interaction of the divisor with QK-Norm or soft-capping during training, are beyond this study’s compute.", "lim"));
children.push(numbered("LM budget. The corrected LM grid uses 96k training tokens and 3 epochs (perplexity ≈ 873); the leak probes show the v1 artifact’s direction, but its fully collapsed state (ppl ≈ 1.08 at 25,000 GPU steps) is not reproduced at CPU scale. The Stage-3 LM gain arm (sequence length 128, g ≤ 4) likewise operates in the inert region, so the LM-scale divisor question remains open.", "lim"));
children.push(numbered("Task scope. Four text datasets; vision, speech, code, and long-context regimes are untouched, as are encoder–decoder and decoder-only architectures.", "lim"));
children.push(numbered("Full-scale gaps. The Stage-1 extended budget (30 of 450 planned full-scale runs) and the full-scale E1–E4 multi-seed arms remain incomplete; the sequence-length axis of the sensitivity grid is covered only implicitly (64-token replica vs. 128-token Stage 1). These are the principal targets for the next full-scale campaign.", "lim"));

// ===== 7. Conclusion =====
children.push(h1("7. Conclusion"));
children.push(body("Under a seed-matched paired design with 787 completed runs across a GPU-scale stage, a replica-scale completion, and an operating-point programme, this study measured what one line of the Transformer — the √d_k divisor in scaled dot-product attention — actually contributes. The answer is sharply conditional in a now-quantitative sense. At full data scale and a 2-epoch budget, the standard divisor outperformed 1/d_k and temperature τ = 2 on 4-class AG News (+0.31 and +0.10 points, p_Holm < 0.05, d_z up to 1.52), but was indistinguishable on binary sentiment. Elsewhere the scaling function is empirically inert: the replica completion of the core grid, the d_k ∈ {32, 64, 128} grid, train-to-convergence with early stopping, and the corrected causal-mask language-modelling grid (25 seeds per arm, perplexity ≈ 873 with a verified exact-zero mask) all show differences within noise, and the original near-1 perplexities are diagnosed and reproduced-in-direction as a mask-leak artifact. Stage 3 explains the inertness and bounds its reach: a second-order power law predicts that two divisors induce attention distributions separated by KL ≈ ½Δβ²σ²(1−1/n); the replica grids sit at measured KL ≤ 0.006 nats, 12× longer training leaves the operating point at σ = 1.1, injecting logit variance until KL = 0.64 nats makes the 2017 default the worst variant (−1.4 points, d_z = −1.76), frontier stabilisers bound the logits but not the contrast, and in pretrained GPT-2 (σ ≈ 45–63) the same no-retraining swap moves perplexity by +27–55% (τ = 2) to 40–93× (1/d_k). The practical conclusions are that √d_k remains a sound default, that the softmax temperature is a cheap knob worth tuning per task, that masking bugs in language-modelling pipelines are best caught by physically-motivated sanity floors on perplexity, that any claim about a parameter-free attention modification should report the operating point it was tested at (one forward pass suffices to measure σ), and that matched-seed paired designs are the cheap instrument that makes such claims testable."));

children.push(new Paragraph({
  spacing: { before: 200, after: 60, line: 360 },
  alignment: AlignmentType.JUSTIFIED,
  children: [
    new TextRun({ text: "Data and code availability. ", bold: true, size: 24, color: BLACK, font: TNR }),
    new TextRun({ text: "The complete reproducibility package — source code, YAML configurations, unit tests, per-seed JSON records, statistical analysis outputs, figure-generation scripts, and this manuscript — is publicly available under an MIT licence at https://github.com/razahasan2000/Attention-Scaling. Supplementary material with full hyperparameters, per-seed summaries, diagnostics, and reproduction instructions accompanies this paper.", size: 24, color: BLACK, font: TNR }),
  ],
}));
children.push(new Paragraph({
  spacing: { after: 240, line: 360 },
  alignment: AlignmentType.JUSTIFIED,
  children: [
    new TextRun({ text: "Declaration of interests. ", bold: true, size: 24, color: BLACK, font: TNR }),
    new TextRun({ text: "The author declares no competing interests.", size: 24, color: BLACK, font: TNR }),
  ],
}));

// ===== References (order of first citation) =====
children.push(h1("References"));
const REFS = [
  "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NeurIPS), pp. 5998–6008, 2017.",
  "Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.",
  "Zhang, S., Zhang, X., Bao, H., and Wei, F. Attention temperature matters in abstractive summarization distillation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 4691–4701, 2022.",
  "Zhai, S., Likhomanenko, T., Littwin, E., Busbridge, D., Ramapuram, J., Zhang, Y., Gu, J., and Susskind, J. Stabilizing transformer training by preventing attention entropy collapse. In Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR 202, pp. 40770–40803, 2023.",
  "Dong, Y., Cordonnier, J.-B., and Loukas, A. Attention is not all you need: pure attention loses rank doubly exponentially with depth. In Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, pp. 2793–2803, 2021.",
  "Nakanishi, K. M. Scalable-softmax is superior for attention. arXiv preprint arXiv:2501.19399, 2025.",
  "Gao, B., et al. Softplus attention with re-weighting boosts length extrapolation in large language models. arXiv preprint arXiv:2501.13428, 2025.",
  "Chi, T.-C., Fan, T.-H., Rudnicky, A. I., and Ramadge, P. J. Dissecting transformer length extrapolation via the lens of receptive field analysis. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), pp. 13522–13537, 2023.",
  "Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Sepah, N., Raff, E., Madan, K., Voleti, V., Kahou, S. E., Michalski, V., Serdyuk, D., Arbel, T., Pal, C., Varoquaux, G., and Vincent, P. Accounting for variance in machine learning benchmarks. In Proceedings of Machine Learning and Systems 3 (MLSys), pp. 747–769, 2021.",
  "Picard, D. torch.manual_seed(3407) is all you need: on the influence of random seeds in deep learning architectures for computer vision. arXiv preprint arXiv:2109.08203, 2021.",
  "Dodge, J., Gururangan, S., Card, D., Schwartz, R., and Smith, N. A. Show your work: improved reporting of experimental results. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), pp. 2185–2194, 2019.",
  "Dror, R., Baumer, G., Shlomov, S., and Reichart, R. The hitchhiker's guide to testing statistical significance in natural language processing. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 1383–1392, 2018.",
  "Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T.-Y. On layer normalization in the transformer architecture. In Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119, pp. 10524–10533, 2020.",
  "Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.",
  "Chiang, D., and Cholak, P. Overcoming a theoretical limitation of self-attention. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 7654–7664, 2022.",
  "Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I. Analyzing multi-head self-attention: specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 5797–5808, 2019.",
  "Michel, P., Levy, O., and Neubig, G. Are sixteen heads really better than one? In Advances in Neural Information Processing Systems 32 (NeurIPS), pp. 14014–14024, 2019.",
  "Press, O., Smith, N. A., and Lewis, M. Train short, test long: attention with linear biases enables input length extrapolation. In International Conference on Learning Representations (ICLR), 2022.",
  "Peng, B., Quesnelle, J., Fan, H., and Shippole, E. YaRN: efficient context window extension of large language models. In International Conference on Learning Representations (ICLR), 2024.",
  "Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P., and Reddy, S. The impact of positional encoding on length generalization in transformers. In Advances in Neural Information Processing Systems 36 (NeurIPS), 2023.",
  "Wilcoxon, F. Individual comparisons by ranking methods. Biometrics Bulletin, 1(6):80–83, 1945.",
  "Holm, S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979.",
  "Cohen, J. Statistical Power Analysis for the Behavioral Sciences, 2nd edition. Lawrence Erlbaum Associates, Hillsdale, NJ, 1988.",
  "Hendrycks, D., and Gimpel, K. Gaussian error linear units (GELU). arXiv preprint arXiv:1606.08415, 2016.",
  "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56):1929–1958, 2014.",
  "Loshchilov, I., and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.",
  "Kingma, D. P., and Ba, J. Adam: a method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.",
  "Zhang, X., Zhao, J., and LeCun, Y. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems 28 (NeurIPS), pp. 649–657, 2015.",
  "Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1631–1642, 2013.",
  "Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 142–150, 2011.",
  "Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In International Conference on Learning Representations (ICLR), 2017.",
  "Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., et al. Transformers: state-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, 2020.",
];
REFS.forEach((t, i) => children.push(refPara(i + 1, t)));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Raza Hasan",
  title: "A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention",
  description: "Seed-matched empirical comparison of attention scaling functions",
  styles: {
    default: {
      document: {
        run: { font: TNR, size: 24, color: BLACK },
        paragraph: { spacing: { line: 360 } },
      },
      heading1: {
        run: { font: TNR, size: 28, bold: true, color: BLACK },
        paragraph: { spacing: { before: 360, after: 200, line: 360 } },
      },
      heading2: {
        run: { font: TNR, size: 24, bold: true, color: BLACK },
        paragraph: { spacing: { before: 280, after: 140, line: 360 } },
      },
    },
  },
  numbering: {
    config: [
      {
        reference: "lim",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, bottom: 1440, left: 1701, right: 1417, header: 850, footer: 992 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { bottom: { style: "single", size: 2, color: "999999" } },
          children: [new TextRun({ text: "Scaling functions in Transformer self-attention", italics: true, size: 18, color: "666666", font: TNR })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 21, color: BLACK, font: TNR })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("Wrote " + OUT + " (" + Math.round(buf.length / 1024) + " KB)");
});
