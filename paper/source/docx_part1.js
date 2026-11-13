// ============================================================================
// Word (.docx) generator for:
// "A Seed-Matched Empirical Comparison of Scaling Functions in Transformer
//  Self-Attention" (Raza Hasan, 2026)
// Mirrors the LaTeX manuscript: same structure, numbers, figures, equations.
// ============================================================================
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, PageNumber, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, ShadingType, ExternalHyperlink,
  Math: OoxmlMath, MathRun, MathFraction, MathRadical, MathSuperScript,
} = require("docx");
const fs = require("fs");

// ---------------------------------------------------------------- constants
const OUT = require("path").join(__dirname, "manuscript.docx");
const FIGDIR = require("path").join(__dirname, "..", "..", "figures");
const BLACK = "000000";
const GREY = "333333";
const TNR = { ascii: "Times New Roman", eastAsia: "Times New Roman" };

// PNG dimension reader (IHDR)
function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

// ---------------------------------------------------------------- helpers
function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: opts.noIndent ? 0 : 480 },
    spacing: { line: 360, after: 80 },
    children: [new TextRun({ text, size: 24, color: BLACK, font: TNR, ...opts.run })],
  });
}

// paragraph from mixed runs (for in-text math fragments)
function bodyRuns(runs, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: opts.noIndent ? 0 : 480 },
    spacing: { line: 360, after: 80 },
    children: runs,
  });
}
function r(text, o = {}) {
  return new TextRun({ text, size: 24, color: BLACK, font: TNR, italics: o.i, bold: o.b, superScript: o.sup, subScript: o.sub });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200, line: 360 },
    children: [new TextRun({ text, bold: true, size: 28, color: BLACK, font: TNR })],
  });
}
function h1b(text) { // unnumbered H1 (backmatter)
  return h1(text);
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140, line: 360 },
    children: [new TextRun({ text, bold: true, size: 24, color: BLACK, font: TNR })],
  });
}
function bullet(text, boldLead = null) {
  const runs = [];
  if (boldLead) runs.push(new TextRun({ text: boldLead, bold: true, size: 24, color: BLACK, font: TNR }));
  runs.push(new TextRun({ text, size: 24, color: BLACK, font: TNR }));
  return new Paragraph({
    bullet: { level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: 60 },
    children: runs,
  });
}
function numbered(text, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: 60 },
    children: [new TextRun({ text, size: 24, color: BLACK, font: TNR })],
  });
}

// equation paragraph (centered, with right-aligned number via tab)
function eq(mathChildren, number) {
  return new Paragraph({
    tabStops: [{ type: "right", position: 8700 }],
    spacing: { before: 120, after: 120, line: 360 },
    children: [
      new OoxmlMath({ children: mathChildren }),
      new TextRun({ text: "\t(" + number + ")", size: 24, color: BLACK, font: TNR }),
    ],
  });
}
const mr = (t) => new MathRun(t);

// caption
function caption(text, kind = "Figure") {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 60, after: 200, line: 280 },
    children: [
      new TextRun({ text: kind + ". ", bold: true, size: 21, color: GREY, font: TNR }),
      new TextRun({ text, size: 21, color: GREY, font: TNR }),
    ],
  });
}
function tableCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    keepNext: true,
    spacing: { before: 200, after: 80, line: 280 },
    children: [
      new TextRun({ text: "Table. ", bold: true, size: 21, color: GREY, font: TNR }),
      new TextRun({ text, size: 21, color: GREY, font: TNR }),
    ],
  });
}

// figure embed
function figure(file, displayWidthPx) {
  const buf = fs.readFileSync(`${FIGDIR}/${file}`);
  const { w, h } = pngSize(buf);
  const dw = Math.min(displayWidthPx, 580);
  const dh = Math.round(dw * h / w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 160, after: 40 },
    children: [new ImageRun({ data: buf, transformation: { width: dw, height: dh }, type: "png" })],
  });
}

// three-line (booktabs style) table
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function scholarlyTable(headers, rows, colPct, alignFirstLeft = true) {
  const mkCell = (text, isHeader, pct, alignLeft) => new TableCell({
    width: { size: pct, type: WidthType.PERCENTAGE },
    borders: {
      top: isHeader ? { style: BorderStyle.SINGLE, size: 8, color: BLACK } : NB,
      bottom: { style: BorderStyle.SINGLE, size: isHeader ? 4 : 8, color: BLACK },
      left: NB, right: NB,
    },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "FFFFFF" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: alignLeft ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { line: 276 },
      children: [new TextRun({ text, bold: isHeader, size: 20, color: BLACK, font: TNR })],
    })],
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB },
    rows: [
      new TableRow({
        tableHeader: true, cantSplit: true,
        children: headers.map((t, i) => mkCell(t, true, colPct[i], alignFirstLeft && i === 0)),
      }),
      ...rows.map((cells, ri) => new TableRow({
        cantSplit: true,
        children: cells.map((t, i) => mkCell(t, false, colPct[i], alignFirstLeft && i === 0)),
      })),
    ],
  });
}

// reference entry
function refPara(num, text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { left: 480, hanging: 480 },
    spacing: { line: 300, after: 60 },
    children: [
      new TextRun({ text: "[" + num + "]\t", size: 22, color: BLACK, font: TNR }),
      new TextRun({ text, size: 22, color: BLACK, font: TNR }),
    ],
  });
}

// ---------------------------------------------------------------- equations
const MSuper = require("docx").MathSuperScript;
const MRadical = require("docx").MathRadical;

// QK^⊤ fragment
const QKtop = () => new MSuper({ children: [mr("QK")], superScript: [mr("⊤")] });

const eqGeneric = eq([
  mr("Attn(Q, K, V) = softmax"),
  new MathFraction({ numerator: [QKtop()], denominator: [mr("s")] }),
  mr(" V"),
], 1);

function attnEq(name, denomRuns, sRuns, number) {
  return eq([
    mr("Attn"), new MathSubScriptSafe("", name), mr("(Q, K, V) = softmax"),
    new MathFraction({ numerator: [QKtop()], denominator: denomRuns }),
    mr(" V,   "), ...sRuns,
  ], number);
}

const eqCI = eq([
  mr("d̄ ± "),
  new MathSubScriptSafe("t", "0.975, n−1"),
  mr(" · "),
  new MathFraction({ numerator: [mr("SD(d)")], denominator: [new MRadical({ children: [mr("n")] })] }),
], 4);

// small helper for subscript math runs
function MathSubScriptSafe(base, sub) {
  return new (require("docx").MathSubScript)({
    children: [mr(base)], subScript: [mr(sub)],
  });
}

const eqDZ = eq([
  new MathSubScriptSafe("d", "z"), mr(" = "),
  new MathFraction({ numerator: [mr("d̄")], denominator: [mr("SD(d)")] }),
], 5);

// ---------------------------------------------------------------- content
const children = [];

// ===== Title block =====
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 240, line: Math.ceil(20 * 23), lineRule: "atLeast" },
  children: [new TextRun({ text: "A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention", bold: true, size: 40, color: BLACK, font: TNR })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60, line: 320 },
  children: [new TextRun({ text: "Raza Hasan", size: 28, color: BLACK, font: TNR })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60, line: 280 },
  children: [new TextRun({ text: "Independent Researcher", size: 22, color: GREY, font: TNR })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60, line: 280 },
  children: [
    new TextRun({ text: "razahasan2000@gmail.com  ·  ", size: 20, color: GREY, font: TNR }),
    new ExternalHyperlink({
      children: [new TextRun({ text: "https://github.com/razahasan2000/Attention-Scaling", size: 20, color: "0563C1", underline: {}, font: TNR })],
      link: "https://github.com/razahasan2000/Attention-Scaling",
    }),
  ],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 280, line: 280 },
  children: [new TextRun({ text: "Preprint — November 2026", italics: true, size: 20, color: GREY, font: TNR })],
}));

// ===== Abstract =====
children.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 80 },
  children: [
    new TextRun({ text: "Abstract. ", bold: true, size: 24, color: BLACK, font: TNR }),
    new TextRun({ text: "The scaled dot-product attention of the Transformer divides query–key logits by √d_k, a design choice inherited from the original architecture and rarely tested in isolation. This study reports a seed-matched empirical comparison of three parameter-free scaling functions for softmax self-attention: the standard 1/√d_k divisor, an alternative 1/d_k divisor that removes the square root, and an explicit softmax temperature τ = 2 applied on top of the standard divisor. Because all three variants rescale the same logits, every seed yields bit-identical initialization, identical batch order, and identical optimization settings across variants, so paired per-seed differences isolate the scaling function as the only manipulated factor. The completed programme comprises 787 runs in three stages. At GPU scale (314 runs; 25 matched seeds per variant on AG News, SST-2, and IMDB, plus ablations), standard scaling significantly outperformed both alternatives on AG News (+0.31 percentage points over 1/d_k, d_z = 1.52, Holm-adjusted p < 0.001; +0.10 points over τ = 2) and was indistinguishable on the binary tasks, and a τ = 0.5 temperature looked most promising on sentiment. This revision completes the preregistered grids that Stage 1 left open, on a documented replica protocol (339 further seed-matched runs on a 2-core CPU sandbox): the 25-seed replica core, a corrected causal-mask WikiText-2 language-modelling grid (25 seeds × 3 variants), a head-dimension grid d_k ∈ {32, 64, 128}, train-to-convergence with early stopping, and multi-seed E1–E4 ablations. Three results stand out. First, the corrected mask restores physically plausible perplexity (≈ 873 versus the artefactual 1.08–1.09 of the original non-causal pipeline; deliberate leak probes reproduce the artifact's direction) and shows no divisor effect at matched power (|d_z| ≤ 0.01). Second, at replica scale the AG News advantage does not reproduce (paired difference −0.01 points, 95% CI [−0.04, +0.02]), the d_k grid shows no amplification (trend p = 0.85), and early stopping selects the same epoch with indistinguishable learning curves for all variants. Third, diagnostics explain the nulls: at reduced scale the raw logit standard deviation (≈ 1) sits far below every divisor, so all variants operate in the softmax's near-uniform regime where the scaling constant is inert. The √d_k divisor's benefit is therefore conditional on the operating point — clearest at full data scale, early in training, on multi-class classification — the softmax temperature remains a parameter-free knob worth tuning, and architectural comparisons should report the compute regime that makes the manipulation bite. A third stage quantifies that regime: a Monte-Carlo-validated power law on the raw logit scale predicts where the divisor contrast is inert (KL below 0.01 nats at raw logit SD ≤ 1), logit-gain and regime-restored runs cross that threshold on cue, and an inference-time divisor swap in pretrained GPT-2 decoders — whose raw logit scale is two orders of magnitude larger — moves perplexity by +27 to +55% (τ = 2) and by 40 to 93× (1/d_k) without any retraining, with a GPU script extending the identical protocol to 1B–7B models. The divisor is thus neither useless nor universal: it is an operating-point-dependent constant, and the operating point is measurable in one forward pass.", size: 24, color: BLACK, font: TNR }),
  ],
}));
children.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 160 },
  children: [
    new TextRun({ text: "Keywords: ", bold: true, size: 24, color: BLACK, font: TNR }),
    new TextRun({ text: "Transformer; self-attention; scaling functions; softmax temperature; seed-matched design; paired statistics; effect sizes; reproducibility.", size: 24, color: BLACK, font: TNR }),
  ],
}));

// ===== 1. Introduction =====
children.push(h1("1. Introduction"));
children.push(body("Self-attention is the computational core of the Transformer architecture [1]. In its standard form it computes, for every pair of positions, a dot product between a query and a key vector, divides that logit by √d_k, applies a softmax normalisation, and mixes value vectors with the resulting weights. The 1/√d_k factor is justified by a variance argument: if the components of d_k-dimensional queries and keys are treated as independent random variables with unit variance, the dot product has variance d_k, so dividing by √d_k keeps the logit variance constant and prevents the softmax from saturating prematurely [1]. This rationale has been quoted across hundreds of follow-up works, yet the scaling constant itself is seldom the object of a controlled empirical test: it is inherited, not chosen."));
children.push(body("Alternatives are mathematically trivial to implement. Dividing by d_k directly removes the square root; introducing an explicit softmax temperature τ [2] rescales the same logits by a tunable constant. Both leave the parameter count, the computational graph, and the memory traffic unchanged, and both plausibly alter training dynamics by reshaping the entropy of the attention distribution [3–5]. Recent architectural work has begun to treat the scaling function as a first-class design axis — proposing input-size-aware softmax rescaling [6], softplus-based alternatives [7], and temperature adjustments for length extrapolation [8] — but the empirical base remains thin: most of these studies compare architectural systems end-to-end, where many factors change at once and a single seed per configuration is common."));
children.push(body("Whether a different scaling constant helps or hurts is therefore an empirical question, and answering it credibly requires more than running two configurations once. Small models trained for short budgets exhibit substantial seed-to-seed variability [9, 10], so single-run comparisons mostly measure noise [11]. Matched-pair designs, in which competing variants share the same random initialization and the same data order, are a more sensitive instrument: they remove variance sources that a between-run comparison cannot control, and they support paired statistical tests on per-seed differences [12]."));
children.push(body("This study reports a comparison of exactly that kind. We implemented a compact pre-LN Transformer encoder [13, 14] whose attention module admits three scaling functions, verified by unit tests that the variants share bit-identical initialization and produce correctly masked computations, and then trained fixed grids of 3 variants × 25 matched seeds on four text datasets under frozen hyperparameter budgets. Four research questions structure the investigation:"));
children.push(bullet("Does standard scaled dot-product attention outperform the alternative scaling functions on predictive accuracy under a matched budget?", "RQ1. "));
children.push(bullet("Do the variants differ in computational efficiency (training time, inference latency, memory) when the scaling change is parameter-free and adds no arithmetic?", "RQ2. "));
children.push(bullet("How do attention entropy, head diversity, and gradient magnitudes respond to the scaling factor, and do those responses explain the performance outcomes?", "RQ3. "));
children.push(bullet("Are the observed performance differences statistically significant under paired tests with multiple-comparison correction?", "RQ4. "));
children.push(body("The preregistered hypothesis H1 — that at least one alternative scaling function would significantly outperform standard attention on a relevant metric — is partially falsified at the Stage-1 operating point: standard attention significantly outperformed both alternatives on AG News, the alternatives were not significantly worse on SST-2 or IMDB, and no variant separated from the rest on WikiText-2 with sufficient statistical power. This revision completes the programme and sharpens the picture further: with the corrected causal mask the WikiText-2 comparison is properly powered (25 seeds per arm) and shows no divisor effect; the head-dimension grid shows no amplification with d_k; and at a documented replica operating point the Stage-1 AG News effect does not reproduce, bounding its generality. The variants were exactly equivalent in measured computational cost, so the accuracy deficit (where present) is not offset by any efficiency gain."));
children.push(body("The contribution of this paper is sixfold. (i) A clean, parameter-free experimental manipulation — a single divisor in one line of code — evaluated under a seed-matched paired design across 787 completed runs in three stages (GPU scale, replica completion, and an operating-point programme), with paired t-tests, Wilcoxon signed-rank tests, Holm–Bonferroni correction, and effect sizes. (ii) Evidence that the standard √d_k divisor's advantage is conditional: significant on a 4-class topic-classification task at full data scale and a short budget, absent on binary sentiment tasks, absent in the replica completion of the core grid, and not offset by any computational trade-off. (iii) A corrected causal-mask language-modelling protocol with formal mask-verification tests and leak probes that explain — rather than merely discard — the artifactually low perplexities of the original runs. (iv) Mechanistic diagnostics linking the scaling factor to attention entropy, inter-head similarity, and gradient magnitudes, with the measured 1/s gradient scaling matching theory, and a replica-scale regime analysis (raw logit scale versus divisor) that explains when the divisor can and cannot matter. (v) A quantitative operating-point programme (Stage 3): a Monte-Carlo-validated power law for the divergence between divisors as a function of raw logit variance, a logit-gain sweep and regime-restored training runs that cross the bite threshold, frontier stabilisers (QK-Norm, soft-capping) tested in context, and an inference-time divisor swap in pretrained GPT-2 decoders showing the contrast is inert at replica scale but worth a factor of 94 in perplexity at pretrained scale, with a GPU script extending the identical protocol to 1B–7B models. (vi) A fully reproducible package — configurations, unit tests, analysis scripts, per-seed records, and this manuscript — released under an MIT licence."));

// ===== 2. Related Work =====
children.push(h1("2. Related Work"));
children.push(h2("2.1 Attention scaling and softmax temperature"));
children.push(body("The scaled dot-product formulation with the √d_k divisor was introduced with the Transformer [1] and inherited by essentially every successor, including pre-LN variants that reorganised the residual stream for stability [13, 14]. The temperature of a softmax — the constant that divides the logits before exponentiation — entered deep learning through knowledge distillation, where Hinton et al. [2] showed that elevated temperatures expose more of the teacher's class-similarity structure. In attention, Zhang et al. [3] demonstrated that a tuned attention temperature matters for summarization distillation, and subsequent work has made the temperature task- or layer-dependent. Nakanishi [6] proposed Scalable-Softmax, which rescales logits as a function of the input vector size to counteract attention fading over long contexts, and Gao et al. [7] replaced softmax with a length-scaled softplus for similar reasons. On the theoretical side, Dong et al. [5] showed that pure attention networks lose rank doubly exponentially with depth — a collapse that residual connections and, indirectly, the sharpness of the attention distribution modulate — while Chiang and Cholak [15] analysed how softmax attention can plateau in probability space and how the scaling interacts with easy fixes. Zhai et al. [4] connected training stability to attention entropy, showing that entropy collapse (near-uniform attention) can be prevented by reparameterisation, which positions the scaling constant as a direct lever on the same quantity. What is missing from this literature is a controlled, seed-matched measurement of how much the classical constant √d_k itself matters on standard NLP benchmarks — the gap this study addresses."));
children.push(h2("2.2 Attention heads, diversity, and their measurement"));
children.push(body("Multi-head attention analyses show that individual heads specialise, and that many heads can be pruned with little loss [16, 17]. Head specialisation is precisely the property that a flatter attention distribution should erode: if all heads attend near-uniformly, their outputs become similar. We therefore measure inter-head cosine similarity of attention distributions and mean attention entropy as mechanistic mediators between the scaling constant and task performance (Section 4.7). Related measurement tools — attention-map similarity and pruning sensitivity — have a long history, but they have not previously been applied to isolate the effect of the logit scaling constant under matched training conditions."));
children.push(h2("2.3 Positional and architectural context"));
children.push(body("Scaling interacts with other attention modifiers studied for length generalisation: additive attention biases (ALiBi) [18], rotary rescaling methods such as YaRN [19], and positional-encoding choices [20, 8]. These works modify attention for long-context capability, whereas the present study deliberately freezes the architecture — including a fixed maximum sequence length of 128 — and varies only the scalar divisor. The two lines are complementary: a reliable answer about the default divisor is a precondition for reasoning about more exotic rescalings."));
children.push(h2("2.4 Experimental methodology: variance and significance"));
children.push(body("Deep-learning benchmarks are dominated by variance across random seeds, compute-minority reports, and incomplete hyperparameter searches [9, 11]. In NLP specifically, Dror et al. [12] found that significance testing is often ignored or misused, and recommended matched designs with paired tests where possible. Even seed sensitivity alone can flip conclusions in computer-vision benchmarks [10]. The methodology of this study follows these recommendations directly: 25 matched seeds per variant, within-seed paired differences as the unit of analysis, paired t-tests with Wilcoxon signed-rank confirmation [21], Holm–Bonferroni family-wise correction across the two comparisons per dataset [22], and paired effect sizes d_z with conventional magnitude labels [23]. To our knowledge this is the first application of a fully seed-matched paired protocol to the question of attention logit scaling."));

module.exports = { children, body, bodyRuns, r, h1, h2, bullet, numbered, eq, mr, attnEq, MathSubScriptSafe, eqGeneric, eqCI, eqDZ, caption, tableCaption, figure, scholarlyTable, refPara, OUT, TNR, BLACK, GREY };
