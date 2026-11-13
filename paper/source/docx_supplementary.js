// ============================================================================
// Word (.docx) generator for the SUPPLEMENTARY MATERIAL of:
// "A Seed-Matched Empirical Comparison of Scaling Functions in Transformer
//  Self-Attention" (Raza Hasan, 2026)
// Mirrors supplementary.tex: sections S1-S12, tables S1-S11, equations, code.
// ============================================================================
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, PageNumber, AlignmentType, HeadingLevel, WidthType,
  BorderStyle, ShadingType, ExternalHyperlink, VerticalAlign,
  Math: OoxmlMath, MathRun, MathFraction, MathRadical, MathSuperScript,
  MathSubScript,
} = require("docx");
const fs = require("fs");
const S2R = require("./s2_rows.js");

const OUT = require("path").join(__dirname, "supplementary.docx");
const BLACK = "000000";
const GREY = "333333";
const TNR = { ascii: "Times New Roman", eastAsia: "Times New Roman" };
const MONO = { ascii: "Consolas", eastAsia: "Consolas" };

// ---------------------------------------------------------------- helpers
function bodyR(runs, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: opts.noIndent ? 0 : 480 },
    spacing: { line: 360, after: 80 },
    children: runs,
  });
}
function rt(text, o = {}) {
  return new TextRun({
    text, size: o.sz || 24, color: o.c || BLACK, font: o.mono ? MONO : TNR,
    italics: o.i, bold: o.b, superScript: o.sup, subScript: o.sub,
  });
}
function mono(t) { return rt(t, { mono: true, sz: 22 }); }

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200, line: 360 },
    children: [new TextRun({ text, bold: true, size: 28, color: BLACK, font: TNR })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140, line: 360 },
    children: [new TextRun({ text, bold: true, size: 24, color: BLACK, font: TNR })],
  });
}
function bullet(runs, boldLead = null) {
  const all = [];
  if (boldLead) all.push(new TextRun({ text: boldLead, bold: true, size: 24, color: BLACK, font: TNR }));
  all.push(...runs);
  return new Paragraph({
    bullet: { level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: 60 },
    children: all,
  });
}
const mr = (t) => new MathRun(t);
const sub = (base, s) => new MathSubScript({ children: [mr(base)], subScript: [mr(s)] });
const QKtop = () => new MathSuperScript({ children: [mr("QK")], superScript: [mr("⊤")] });

// display equation, centered, optional bold label before it
function eqMath(label, mathChildren) {
  const kids = [];
  if (label) kids.push(new TextRun({ text: label, bold: true, size: 24, color: BLACK, font: TNR }));
  kids.push(new OoxmlMath({ children: mathChildren }));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120, line: 360 },
    children: kids,
  });
}

// caption for tables (S-numbered)
function tableCaption(num, runs) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    keepNext: true,
    spacing: { before: 200, after: 80, line: 280 },
    children: [
      new TextRun({ text: "Table S" + num + ". ", bold: true, size: 21, color: GREY, font: TNR }),
      ...runs.map((o) => new TextRun({
        text: o.t, size: o.sz || 21, color: GREY, font: o.mono ? MONO : TNR,
        italics: o.i, subScript: o.sub, superScript: o.sup,
      })),
    ],
  });
}

// booktabs-style three-line table
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function runsOfCell(spec, defSize) {
  const mk = (o) => new TextRun({
    text: o.t, bold: o.b, italics: o.i, superScript: o.sup, subScript: o.sub,
    size: o.sz || defSize, color: BLACK, font: o.mono ? MONO : TNR,
  });
  if (typeof spec === "string") return [mk({ t: spec })];
  if (spec.runs) return spec.runs.map(mk);
  return [mk(spec)];
}
function scholarlyTable(headers, rows, colPct) {
  const mkCell = (spec, isHeader, pct, alignLeft, rowSpan) => new TableCell({
    width: { size: pct, type: WidthType.PERCENTAGE },
    borders: {
      top: isHeader ? { style: BorderStyle.SINGLE, size: 8, color: BLACK } : NB,
      bottom: { style: BorderStyle.SINGLE, size: isHeader ? 4 : 8, color: BLACK },
      left: NB, right: NB,
    },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "FFFFFF" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    verticalAlign: VerticalAlign.CENTER,
    rowSpan: rowSpan || undefined,
    children: [new Paragraph({
      alignment: alignLeft ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { line: 276 },
      children: runsOfCell(spec, 20),
    })],
  });
  const cellSpec = (s) => (s && typeof s === "object" && !s.runs && !Array.isArray(s)) ? s : {};
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB },
    rows: [
      new TableRow({
        tableHeader: true, cantSplit: true,
        children: headers.map((s, i) => mkCell(s, true, colPct[i], i === 0, cellSpec(s).rowSpan)),
      }),
      ...rows.map((cells) => new TableRow({
        cantSplit: true,
        children: cells.map((s, i) => mkCell(s, false, colPct[i], i === 0, cellSpec(s).rowSpan)),
      })),
    ],
  });
}

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

function codeLine(t) {
  return new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "F5F5F5" },
    spacing: { after: 0, line: 252 },
    indent: { left: 240, right: 240 },
    children: [new TextRun({ text: t.length ? t : " ", size: 18, color: "222222", font: MONO })],
  });
}

// ---------------------------------------------------------------- content
const children = [];

// ===== Title block =====
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 120 },
  children: [new TextRun({ text: "Supplementary Material", bold: true, size: 40, color: BLACK, font: TNR })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 160, line: 320 },
  children: [new TextRun({ text: "A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention", size: 26, color: BLACK, font: TNR })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60, line: 280 },
  children: [new TextRun({ text: "Raza Hasan — Independent Researcher", size: 22, color: BLACK, font: TNR })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60, line: 280 },
  children: [new ExternalHyperlink({
    children: [new TextRun({ text: "https://github.com/razahasan2000/Attention-Scaling", size: 20, color: "0563C1", underline: {}, font: TNR })],
    link: "https://github.com/razahasan2000/Attention-Scaling",
  })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 160, line: 280 },
  children: [new TextRun({ text: "Preprint — November 2026", italics: true, size: 20, color: GREY, font: TNR })],
}));
children.push(new Paragraph({
  spacing: { after: 200 },
  border: { bottom: { style: "single", size: 4, color: "000000" } },
  children: [new TextRun({ text: "", size: 2 })],
}));

// ===== Intro =====
children.push(bodyR([rt("This supplement provides the complete experimental documentation for the main manuscript: the full architecture specification (S1), dataset and preprocessing details (S2), hyperparameters (S3), the experiment manifest (S4), per-seed summaries for the Stage-1 GPU-scale grids (S5), the complete statistical analysis (S6), training-budget sensitivity (S7), focused ablations E1–E4 (S8), attention and optimisation diagnostics (S9), efficiency measurements (S10), the figure index (S11), and reproduction instructions (S12). This revision adds the Stage-2 replica-completion documentation: replica protocol specification (S13), Stage-2 per-seed records (S14), mask verification and leak probes (S15), and Stage-2 diagnostics with the LM convergence probe (S16). The operating-point programme (Stage 3) adds the divisor power law with its derivation and Monte Carlo validation (S17) and the Stage-3 operating-point records (S18). All numbers are generated from the committed per-seed records in the repository; the analysis scripts that recompute every table are included in the package.")], { noIndent: true }));

// ===== S1 =====
children.push(h1("S1. Full architecture specification"));
children.push(h2("S1.1 Layer inventory"));
children.push(bullet([rt("nn.Embedding(18855, 128, padding_idx=0)", { mono: true, sz: 22 }), rt(".")], "Token embeddings: "));
children.push(bullet([rt("learned "), rt("nn.Parameter", { mono: true, sz: 22 }), rt(" of shape (1, 128, 128), initialised as N(0, 0.02"), rt("2", { sup: true }), rt(").")], "Positional embeddings: "));
children.push(bullet([rt("pre-LN Transformer block — LayerNorm → multi-head self-attention → residual add, then LayerNorm → feed-forward (GELU) → residual add. Query, key, value, and output projections are bias-free "), rt("nn.Linear(128, 128)", { mono: true, sz: 22 }), rt("; the feed-forward is 128 → 256 → 128 with GELU.")], "Encoder blocks (×2): "));
children.push(bullet([rt("nn.LayerNorm(128)", { mono: true, sz: 22 }), rt(".")], "Final LayerNorm: "));
children.push(bullet([rt("nn.Linear(128, num_classes)", { mono: true, sz: 22 }), rt(".")], "Classification head: "));
children.push(bullet([rt("nn.Linear(128, 18855, bias=False)", { mono: true, sz: 22 }), rt(" with weight tying to the token embedding.")], "Language-modelling head: "));

children.push(h2("S1.2 Dimensions and counts"));
children.push(tableCaption(1, [{ t: "Architecture dimensions and parameter counts (identical across all scaling variants)." }]));
children.push(scholarlyTable(
  ["Parameter", "Value", "Notes"],
  [
    [{ runs: [{ t: "d" }, { t: "model", sub: true }] }, "128", "embedding width"],
    ["Heads h", "4", "per block"],
    [{ runs: [{ t: "d" }, { t: "k", sub: true }, { t: " = d" }, { t: "v", sub: true }] }, "32", "per-head dimension"],
    ["Layers L", "2", "pre-LN"],
    [{ runs: [{ t: "d" }, { t: "ff", sub: true }] }, "256", "feed-forward width"],
    ["Dropout", "0.1", "embeddings, attention output, FFN"],
    ["Max sequence length", "128", "right padding, attention mask"],
    ["Vocabulary", "18,855", "classification and LM"],
    ["Logit clamp", "[−50, 50]", "numerical stability"],
    [{ t: "Parameters (classification)", b: true }, { t: "2,695,556", b: true }, { t: "all variants identical", b: true }],
    [{ t: "Parameters (LM)", b: true }, { t: "4,120,576", b: true }, { t: "tied LM head", b: true }],
  ],
  [32, 22, 46]
));

children.push(h2("S1.3 Scaling variants"));
children.push(bodyR([rt("For Q, K, V ∈ ℝ"), rt("n×dk", { sup: true }), rt(" and effective divisor s:")]));
children.push(eqMath("Standard:      ", [
  mr("softmax"),
  new MathFraction({ numerator: [QKtop()], denominator: [new MathRadical({ children: [sub("d", "k")] })] }),
  mr(" V,   s = "),
  new MathRadical({ children: [mr("32")] }),
  mr(" ≈ 5.6569,"),
]));
children.push(eqMath("Temperature:   ", [
  mr("softmax"),
  new MathFraction({ numerator: [QKtop()], denominator: [mr("τ"), new MathRadical({ children: [sub("d", "k")] })] }),
  mr(" V,   s = τ"),
  new MathRadical({ children: [mr("32")] }),
  mr(" ≈ 11.3137  (τ = 2),"),
]));
children.push(eqMath("Alternative:   ", [
  mr("softmax"),
  new MathFraction({ numerator: [QKtop()], denominator: [sub("d", "k")] }),
  mr(" V,   s = 32."),
]));
children.push(bodyR([
  rt("The mixed-scale ablation arm ("), mono("multiscale_v1"), rt(") divides the logits of heads 1–2 by √d"),
  rt("k", { sub: true }), rt(" (standard) and heads 3–4 by d"), rt("k", { sub: true }),
  rt("; the learning-rate compensation arm ("), mono("dk_lrcomp"),
  rt(") multiplies the learning rates of W"), rt("q", { sub: true }), rt(" and W"), rt("k", { sub: true }),
  rt(" in the 1/d"), rt("k", { sub: true }), rt(" variant by √d"), rt("k", { sub: true }), rt(" ≈ 5.66."),
]));

children.push(h2("S1.4 Unit tests"));
children.push(bodyR([
  rt("Five test suites (22 tests) cover: bit-identical initialization across variants for a fixed seed; gradient flow through every variant; output shapes and dtype contracts; masked-attention correctness (padding positions receive negligible weight); and data-pipeline determinism (identical batch order for identical seeds). Run with "),
  mono("pytest tests/"), rt("."),
]));

// ===== S2 =====
children.push(h1("S2. Dataset and preprocessing details"));
children.push(h2("S2.1 Sources"));
children.push(tableCaption(2, [{ t: "HuggingFace dataset identifiers." }]));
children.push(scholarlyTable(
  ["Dataset", "HF identifier", "Configuration"],
  [
    ["AG News", { t: "fancyzhx/ag_news", mono: true }, { t: "default", mono: true }],
    ["SST-2", { t: "nyu-mll/glue", mono: true }, { t: "sst2", mono: true }],
    ["IMDB", { t: "stanfordnlp/imdb", mono: true }, { t: "default", mono: true }],
    ["WikiText-2", { t: "Salesforce/wikitext", mono: true }, { t: "wikitext-2-v1", mono: true }],
  ],
  [24, 44, 32]
));
children.push(h2("S2.2 Splits and tokenization"));
children.push(bodyR([
  rt("Classification training sets are subsampled to 10,000 examples with a 20% held-out validation split (SST-2 uses the official GLUE validation/test splits: 1,115 / 1,821); test sets are the frozen official test partitions. Text is tokenised with the regular expression "),
  mono("[a-z0-9']+"), rt(" over lowercased input (classification) or whitespace splitting (LM, with an "), mono("<eos>"),
  rt(" token). Vocabularies keep the top 18,855 tokens at minimum frequency 2 (plus special tokens) and are built from training data only. Label mappings preserve the original class identities (AG News 1–4 → 0–3; IMDB neg/pos → 0/1). Sequences are right-padded to 128 positions; attention masks exclude padding."),
]));

// ===== S3 =====
children.push(h1("S3. Full hyperparameters"));
children.push(tableCaption(3, [{ t: "Optimisation configuration (all grids)." }]));
children.push(scholarlyTable(
  ["Parameter", "Value"],
  [
    ["Optimiser", "AdamW [1]"],
    ["Learning rate", { runs: [{ t: "3 × 10" }, { t: "−4", sup: true }] }],
    ["Weight decay", "0.01"],
    [{ runs: [{ t: "Betas / ε" }] }, { runs: [{ t: "(0.9, 0.999) / 10" }, { t: "−8", sup: true }] }],
    ["LR schedule", "linear warmup over the first 5% of steps, then cosine decay to 0"],
    ["Gradient clipping", { runs: [{ t: "max ‖g‖" }, { t: "2", sub: true }, { t: " = 1.0" }] }],
    ["Logit clamp", "[−50, 50]"],
    ["Mixed precision", "disabled (reproducibility)"],
    ["Batch size (classification / LM)", "64 / 16"],
    ["Eval batch size (classification / LM)", "64 / 32"],
    ["Budgets", "core: 2 epochs (classification), 25,000 steps (LM); extended: 25 epochs; ablations E1–E4: 5 epochs"],
    ["Checkpoints", "epochs 2/5/10/15/20/25 (extended); steps 5k/10k/15k/20k/25k (LM)"],
    ["Metrics", "accuracy, macro-F1, loss (classification); perplexity, loss (LM)"],
    ["Seeds", "1001–1025 (core); 1001–1003 (temperature ablation)"],
  ],
  [40, 60]
));

// ===== S4 =====
children.push(h1("S4. Complete experiment manifest"));
children.push(bodyR([rt("The programme is complete in three stages (Table 4): 787 runs in total, 314 at GPU scale (Stage 1), 339 at replica scale (Stage 2; 312 pre-registered completions plus 27 mask-artifact probes), and 134 in the Stage-3 operating-point programme (128 matched-seed training runs plus 6 pretrained-model evaluations; Sections S17-S18). Relative to the original 951-run full-scale plan, the Stage-1 extended budget (30 of 450 planned runs) and the full-scale multi-seed E1–E4 arms remain open; every other grid is complete in at least one stage, and the replica stage adds the head-dimension grid that the original plan sketched as model-size sensitivity.")]));
children.push(tableCaption(4, [{ t: "Completed experiment manifest (both stages)." }]));
children.push(scholarlyTable(
  ["Grid", "Design", "Stage", "Runs"],
  [
    ["Core classification", "3 variants × 25 seeds × 3 datasets", "1", "225"],
    ["Temperature ablation", "4 τ × 3 seeds × 2 datasets", "1", "24"],
    ["Extended budget", "3 variants × 5 seeds × 2 datasets, 25 epochs", "1", "30"],
    ["Core LM (v1 protocol)", "3 variants, 25,000 steps", "1", "25"],
    ["Focused ablations E1–E4", "single-seed arms, 5 epochs", "1", "10"],
    ["Replica core classification", "3 variants × 25 seeds, AG News", "2", "75"],
    ["Replica LM (corrected causal mask)", "3 variants × 25 seeds", "2", "75"],
    ["Head-dimension grid", "d", "2", "108"],
    ["Train-to-convergence", "3 variants × 5 seeds × 2 datasets", "2", "30"],
    ["Replica ablations E1–E4", "4 arms × 3 seeds × 2 datasets", "2", "24"],
    ["Mask-artifact probes", "leaky vs. causal, 3 and 10 epochs", "2", "27"],
    ["Logit-gain sweep (Stage 3)", "g \u2208 {1,2,4,8,16,32} \u00d7 3 variants \u00d7 4 seeds", "3", "72"],
    ["Regime-restored AG News (Stage 3)", "6,000 train, 8 epochs, probes, d_k \u2208 {32,128}", "3", "18"],
    ["Frontier-stabiliser bridge (Stage 3)", "QK-Norm / soft-cap arms \u00d7 4 seeds", "3", "20"],
    ["LM logit gain (Stage 3)", "g \u2208 {1,4} \u00d7 3 variants \u00d7 3 seeds, seq 128", "3", "18"],
    ["Frontier divisor swap (Stage 3)", "GPT-2 + GPT-2-medium, inference-time", "3", "6"],
    [{ t: "Total", b: true }, "", "", { t: "787", b: true }],
  ],
  [30, 44, 10, 16]
));

// ===== S5 =====
const vStd = "Standard";
const vTemp = { runs: [{ t: "Temp (τ = 2)" }] };
const vDk = { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] };
children.push(h1("S5. Per-seed results summary"));
children.push(tableCaption(5, [{ t: "Core classification grids: per-seed statistics over 25 matched seeds (accuracy fraction). SD is the across-seed standard deviation." }]));
children.push(scholarlyTable(
  ["Dataset", "Variant", "Mean", "SD", "Min", "Max"],
  [
    ["AG News", vStd, "0.9123", "0.0025", "0.9068", "0.9156"],
    ["", vTemp, "0.9113", "0.0026", "0.9058", "0.9149"],
    ["", vDk, "0.9092", "0.0032", "0.9021", "0.9141"],
    ["SST-2", vStd, "0.8066", "0.0115", "0.7826", "0.8310"],
    ["", vTemp, "0.8070", "0.0104", "0.7854", "0.8295"],
    ["", vDk, "0.8062", "0.0106", "0.7831", "0.8267"],
    ["IMDB", vStd, "0.8193", "0.0099", "0.7980", "0.8378"],
    ["", vTemp, "0.8193", "0.0093", "0.8002", "0.8364"],
    ["", vDk, "0.8196", "0.0091", "0.7995", "0.8352"],
    ["WikiText-2 (ppl, v1 artefact)", { runs: [{ t: "Standard (" }, { t: "n" }, { t: " = 15)" }] }, "1.0917", "0.0061", "—", "—"],
    ["", { runs: [{ t: "Temp (τ = 2, " }, { t: "n" }, { t: " = 7)" }] }, "1.0882", "0.0056", "—", "—"],
    ["", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " (" }, { t: "n" }, { t: " = 3)" }] }, "1.0809", "0.0021", "—", "—"],
  ],
  [20, 24, 14, 14, 14, 14]
));
children.push(bodyR([
  rt("The WikiText-2 rows are the superseded v1 leaky-mask values (historical record); the corrected 25-seed causal-mask protocol is tabulated in S14. Stage-2 per-seed records (replica core, corrected LM, head-dimension grid, convergence training, and replica ablations) are tabulated in Section S14."),
]));

// ===== S6 =====
children.push(h1("S6. Full statistical analysis"));
children.push(h2("S6.1 Paired comparisons (core classification, n = 25)"));
children.push(tableCaption(6, [
  { t: "Complete paired statistics. diff = baseline − variant (positive favours standard scaling); " },
  { t: "p" }, { t: "Holm", sub: true },
  { t: " corrects across the two comparisons within each dataset." },
]));
children.push(scholarlyTable(
  ["Dataset", "Comparison", "diff", "95% CI", { runs: [{ t: "d" }, { t: "z", sub: true }] }, "p", { runs: [{ t: "p" }, { t: "Holm", sub: true }] }],
  [
    ["AG News", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " vs. std" }] }, "+0.0031", "[+0.0022, +0.0039]", "1.52", "<0.001", "<0.001"],
    ["", { runs: [{ t: "τ = 2 vs. std" }] }, "+0.0010", "[+0.0002, +0.0017]", "0.53", "0.013", "0.013"],
    ["SST-2", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " vs. std" }] }, "+0.0004", "[−0.0023, +0.0031]", "0.06", "0.780", "0.781"],
    ["", { runs: [{ t: "τ = 2 vs. std" }] }, "−0.0005", "[−0.0031, +0.0021]", "−0.07", "0.718", "0.718"],
    ["IMDB", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " vs. std" }] }, "−0.0004", "[−0.0011, +0.0004]", "−0.19", "0.352", "0.703"],
    ["", { runs: [{ t: "τ = 2 vs. std" }] }, "−0.0001", "[−0.0006, +0.0005]", "−0.04", "0.831", "0.831"],
  ],
  [13, 20, 13, 22, 10, 11, 11]
));
children.push(h2("S6.2 Effect-size interpretation"));
children.push(bodyR([
  rt("Following Cohen [2]: |d"), rt("z", { sub: true }), rt("| < 0.2 negligible; 0.2–0.5 small; 0.5–0.8 medium; > 0.8 large. Only the two AG News comparisons exceed the negligible band (d"),
  rt("z", { sub: true }), rt(" = 1.52 large; d"), rt("z", { sub: true }), rt(" = 0.53 medium)."),
]));

// ===== S7 =====
children.push(h1("S7. Training-budget sensitivity"));
children.push(tableCaption(7, [{ t: "Extended budget (5 seeds, checkpoints at 2/5/10/25 epochs): test accuracy by variant. Differences decay to seed-noise level by 25 epochs." }]));
children.push(scholarlyTable(
  ["Dataset", "Variant", "2 ep", "5 ep", "10 ep", "25 ep"],
  [
    ["SST-2", vStd, "0.813", "0.828", "0.835", "0.842"],
    ["", vTemp, "0.815", "0.829", "0.836", "0.843"],
    ["", vDk, "0.815", "0.827", "0.834", "0.841"],
    ["IMDB", vStd, "0.822", "0.831", "0.838", "0.845"],
    ["", vTemp, "0.822", "0.832", "0.839", "0.846"],
    ["", vDk, "0.822", "0.830", "0.837", "0.844"],
  ],
  [16, 26, 14, 14, 15, 15]
));

// ===== S8 =====
children.push(h1("S8. Focused ablations E1–E4"));
children.push(bodyR([
  rt("Single-seed arms (seed 0), 5 epochs. "), mono("multiscale_v1"),
  rt(" divides the logits of heads 1–2 by √d"), rt("k", { sub: true }), rt(" and heads 3–4 by d"),
  rt("k", { sub: true }), rt("; "), mono("dk_lrcomp"),
  rt(" multiplies the Q/K learning rates of the 1/d"), rt("k", { sub: true }), rt(" variant by √d"),
  rt("k", { sub: true }), rt("."),
]));
children.push(tableCaption(8, [{ t: "Focused ablation results (test accuracy)." }]));
children.push(scholarlyTable(
  ["ID", "Dataset", "Arm", "Acc.", { runs: [{ t: "Δ vs. std" }] }, "Reading"],
  [
    ["E1", "SST-2", "standard", "0.8062", "—", { t: "mixed-scale +1.6 pp", rowSpan: 2 }],
    ["", "", { t: "multiscale_v1", mono: true }, "0.8222", "+1.60 pp"],
    ["E2", "SST-2", "standard", "0.8268", "—", { t: "compensation fails", rowSpan: 3 }],
    ["", "", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] }, "0.8142", "−1.26 pp"],
    ["", "", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " + LR comp." }] }, "0.8108", "−1.60 pp"],
    ["E3", "IMDB", "standard", "0.8152", "—", { t: "no difference", rowSpan: 2 }],
    ["", "", { t: "multiscale_v1", mono: true }, "0.8166", "+0.14 pp"],
    ["E4", "IMDB", "standard", "0.8183", "—", { t: "compensation fails", rowSpan: 3 }],
    ["", "", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] }, "0.8145", "−0.38 pp"],
    ["", "", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " + LR comp." }] }, "0.8157", "−0.26 pp"],
  ],
  [7, 12, 24, 12, 16, 29]
));
children.push(bodyR([rt("All four Stage-1 arms are single-seed and hypothesis-generating. The Stage-2 replica advances both ablation questions to 3 matched seeds per arm (Section S14): the mixed-scale result is null (+0.04 pp SST-2, +0.05 pp IMDB), and learning-rate compensation remains neutral-to-harmful (−1.30 pp SST-2, −0.27 pp IMDB), consistent with the Stage-1 reading.")]));

// ===== S9 =====
children.push(h1("S9. Attention and optimisation diagnostics"));
children.push(tableCaption(9, [
  { t: "Diagnostics at epoch 1 (seed 0). Entropy and head similarity averaged over heads and positions; gradient norms are ℓ" },
  { t: "2", sub: true },
  { t: " norms at the first optimisation step." },
]));
children.push(scholarlyTable(
  ["Metric", "Standard", { runs: [{ t: "Temp (τ = 2)" }] }, vDk],
  [
    ["Mean attention entropy (nats)", "2.14", "2.31", "2.48"],
    ["Max attention weight", "0.32", "0.28", "0.24"],
    ["Inter-head cosine similarity", "0.23", "0.31", "0.42"],
    [{ runs: [{ t: "‖∂L/∂Q‖" }] }, "0.042", "0.021", "0.011"],
    [{ runs: [{ t: "‖∂L/∂K‖" }] }, "0.041", "0.020", "0.010"],
    [{ runs: [{ t: "‖∂L/∂V‖" }] }, "0.038", "0.039", "0.037"],
    [{ runs: [{ t: "Theoretical Q/K ratio (1/s vs. standard)" }] }, "1.000", "0.500", "0.177"],
    ["Observed Q ratio", "1.000", "0.500", "0.262"],
    ["Observed K ratio", "1.000", "0.488", "0.244"],
  ],
  [40, 20, 20, 20]
));
children.push(bodyR([
  rt("Larger divisors produce flatter (higher-entropy), more similar heads, and attenuate Q/K gradients by approximately 1/s; value gradients are unaffected, matching the analytical derivative of scaled dot-product attention."),
]));

// ===== S10 =====
children.push(h1("S10. Efficiency analysis"));
children.push(tableCaption(10, [{ t: "Efficiency measurements (identical parameter counts by construction; differences within run-to-run noise)." }]));
children.push(scholarlyTable(
  ["Metric", "Standard", { runs: [{ t: "Temp (τ = 2)" }] }, vDk],
  [
    ["Parameters (classification)", "2,695,556", "2,695,556", "2,695,556"],
    ["Parameters (LM)", "4,120,576", "4,120,576", "4,120,576"],
    ["Epoch time, SST-2 (s)", "32.1", "32.3", "32.0"],
    ["Epoch time, AG News (s)", "59.8", "60.1", "59.5"],
    ["Peak GPU memory (GB)", "5.2", "5.2", "5.2"],
    ["Inference latency, 1k samples (s)", "0.84", "0.85", "0.84"],
  ],
  [40, 20, 20, 20]
));

// ===== S11 =====
children.push(h1("S11. Figure index"));
children.push(tableCaption(11, [
  { t: "All figures are regenerated at 300 DPI from the committed per-seed records by " },
  { t: "scripts/generate_figures.py", mono: true },
  { t: "-equivalent analysis code." },
]));
children.push(scholarlyTable(
  ["File", "Content"],
  [
    [{ t: "fig_architecture.png", mono: true }, "Pre-LN encoder architecture and scaling variants"],
    [{ t: "fig_design.png", mono: true }, "Matched-seed paired experimental design"],
    [{ t: "fig_main_performance.png", mono: true }, "Core-grid accuracy by variant and dataset"],
    [{ t: "fig_paired_differences.png", mono: true }, "Paired differences with 95% CIs"],
    [{ t: "fig_temperature_ablation.png", mono: true }, { runs: [{ t: "Accuracy vs. temperature τ" }] }],
    [{ t: "fig_training_curves.png", mono: true }, "Extended-budget convergence"],
    [{ t: "fig_head_diversity.png", mono: true }, "Entropy, max weight, head cosine"],
    [{ t: "fig_gradient_ratios.png", mono: true }, { runs: [{ t: "Gradient norms vs. 1/s theory" }] }],
    [{ t: "fig_efficiency.png", mono: true }, "Epoch time and inference latency"],
    [{ t: "fig_lm_perplexity.png", mono: true }, "WikiText-2 perplexity by variant (v1 record)"],
    [{ t: "fig_stage2_core.png", mono: true }, "Replica core means and paired differences"],
    [{ t: "fig_dk_grid.png", mono: true }, { runs: [{ t: "Head-dimension grid: accuracy and gap vs. d" }, { t: "k", sub: true }] }],
    [{ t: "fig_lm_corrected.png", mono: true }, "Corrected LM perplexity and mask-artifact probes"],
    [{ t: "fig_convergence.png", mono: true }, "Train-to-convergence learning curves"],
  ],
  [40, 60]
));

// ===== S12 =====
children.push(h1("S12. Reproduction instructions"));
const CODE = [
  "# 1. Environment (Python >= 3.10)",
  "pip install torch torchvision torchaudio",
  "pip install numpy scipy matplotlib PyYAML datasets tqdm \\",
  "            transformers tokenizers accelerate scikit-learn",
  "",
  "# 2. Preflight validation (environment, configs, 22 unit tests)",
  "python scripts/validate.py",
  "",
  "# 3. Core grid: 3 variants x 25 seeds x 3 datasets (225 runs)",
  "python scripts/run_experiments.py --config-dir configs \\",
  "       --output-dir results --grid core",
  "",
  "# 4. Temperature ablation (tau in {0.5, 1.0, 1.5, 2.0}; 24 runs)",
  "python scripts/run_temp_ablation.py",
  "",
  "# 5. Extended 25-epoch budget and model-size sensitivity",
  "python scripts/run_experiments.py --config-dir configs \\",
  "       --output-dir results --grid extended",
  "python scripts/run_model_size.py",
  "",
  "# 6. Focused ablations E1-E4",
  "python scripts/run_ablations.py",
  "",
  "# 7. Statistical analysis and figures",
  "python scripts/analyse_results.py --results-dir results \\",
  "       --output-dir results/statistical_tests \\",
  "       --datasets agnews sst2 imdb wikitext2",
  "python scripts/generate_figures.py results/statistical_tests figures",
];
CODE.forEach((line, i) => children.push(codeLine(line)));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(bodyR([
  rt("Stage-2 replica completion (this revision). The replica stage ships as a self-contained suite with its own runner, aggregation, and verification scripts:"),
]));
const CODE2 = [
  "# Stage 2: data preparation (replica subsets and vocabularies)",
  "python scripts/experiments/prepare_data.py",
  "",
  "# Mask-verification suite (asserts exact-zero future/padding attention)",
  "python scripts/experiments/verify_mask.py",
  "",
  "# Grids (resume-safe; append-only JSONL records)",
  "python scripts/experiments/runner.py --phase lm     --procs 2   # 75 runs",
  "python scripts/experiments/runner.py --phase core   --procs 2   # 75 runs",
  "python scripts/experiments/runner.py --phase grid   --procs 2   # 108 runs",
  "python scripts/experiments/runner.py --phase conv   --procs 2   # 30 runs",
  "python scripts/experiments/runner.py --phase abl    --procs 2   # 24 runs",
  "python scripts/experiments/runner.py --phase lmleak --procs 2   # 9 probe runs",
  "python scripts/experiments/runner.py --phase lmc    --procs 2   # 18 probe runs",
  "",
  "# Attention diagnostics and paired statistics",
  "python scripts/experiments/diagnostics.py",
  "python scripts/experiments/aggregate.py",
];
CODE2.forEach((line) => children.push(codeLine(line)));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(bodyR([
  rt("Every run emits a per-seed JSON record (configuration hash, training history, test metrics); the committed "),
  mono("results/"),
  rt(" directory contains the records used in this paper, so all tables and figures can be regenerated without re-training."),
]));

// ===== S13 =====
children.push(h1("S13. Stage-2 replica protocol specification"));
children.push(bodyR([rt("Stage 2 executes every preregistered grid on a 2-core CPU sandbox (3 GB RAM, PyTorch 2.15.0 CPU, one torch thread per worker, two worker processes). The architecture, variant definitions, matched-seed construction, optimizer, and analysis pipeline are identical to Stage 1; only the data and step budgets are reduced, as follows.")]));
children.push(tableCaption(12, [{ t: "Stage-2 replica configuration." }]));
children.push(scholarlyTable(
  ["Component", "Classification replica", "LM replica"],
  [
    ["Training data", "2,000 examples per dataset", "96,000 tokens (WikiText-2 train head)"],
    ["Validation", "500 examples", "40,000 tokens (official validation head)"],
    ["Test", "AG News 4,000 / SST-2 872 / IMDB 2,500", "validation perplexity"],
    ["Sequence length", "64 (right-padded)", "64 (non-overlapping chunks)"],
    ["Vocabularies", "7,234 / 2,249 / 12,000", "5,655"],
    [{ runs: [{ t: "d" }, { t: "model", sub: true }, { t: " / heads" }] }, { runs: [{ t: "128 / 4 (d" }, { t: "k", sub: true }, { t: " = 32)" }] }, { runs: [{ t: "96 / 3 (d" }, { t: "k", sub: true }, { t: " = 32)" }] }],
    ["Layers / d_ff / dropout", "2 / 256 / 0.1", "2 / 192 / 0.1"],
    ["Batch size", "32", "32"],
    ["Budget", "2 epochs (core, grid); ≤20 epochs (convergence)", "3 epochs; 10-epoch probe"],
    ["Optimiser", { t: "AdamW, lr 3×10⁻⁴, wd 0.01, 5% warmup, cosine, clip 1.0", rowSpan: 4 }, ""],
    ["Seeds", "core/LM 1001–1025; grid 1001–1012; convergence 1001–1005; ablations 1001–1003", ""],
    ["Causal mask (LM)", "lower-triangular AND key-padding, applied after the [−50,50] clamp", ""],
    ["Records", "append-only JSONL keyed by (phase, dataset, variant, d_k, seed)", ""],
  ],
  [24, 40, 36]
));
children.push(bodyR([
  rt("Splits are drawn with a fixed RNG seed (42) from the official training partitions; test subsets are the first n rows of a fixed permutation of the official test partitions (SST-2 uses the official 872-example labelled dev set as test). Because the same subsets and order are shared by all variants of a seed, the paired contrasts remain exactly matched. Grid completion is checkpointed: any interrupted batch resumes from the JSONL key set, and the final manifest was audited for exactly-once execution of all 312 pre-registered runs."),
]));

// ===== S14 =====
children.push(h1("S14. Stage-2 per-seed records"));
children.push(bodyR([
  rt("Accuracy fractions (classification) or validation perplexity (LM) per matched seed. Every table below is generated directly from the append-only run records; the paired statistics in the main text are computed from exactly these values."),
]));
children.push(tableCaption(13, [{ t: "Replica core, AG News, 2 epochs: test accuracy per seed." }]));
children.push(scholarlyTable(
  ["Seed", "Standard", "Temp (τ=2)", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] }],
  S2R.CORE_ROWS, [25, 25, 25, 25]));
children.push(tableCaption(14, [{ t: "Corrected causal-mask language modelling, WikiText-2, 3 epochs: validation perplexity per seed (25 seeds × 3 variants, all completed)." }]));
children.push(scholarlyTable(
  ["Seed", "Standard", "Temp (τ=2)", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] }],
  S2R.LM_ROWS, [25, 25, 25, 25]));
children.push(tableCaption(15, [{ runs: [{ t: "Head-dimension grid, AG News, 2 epochs: test accuracy per seed. Columns group the three variants (Std / τ=2 / 1/d" }, { t: "k", sub: true }, { t: ") within each head dimension." }] }]));
children.push(scholarlyTable(
  ["Seed", "S32", "T32", "K32", "S64", "T64", "K64", "S128", "T128", "K128"],
  S2R.GRID_ROWS, [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]));
children.push(tableCaption(16, [{ t: "Train-to-convergence with early stopping (up to 20 epochs, patience 3 on validation loss): best-validation-epoch test accuracy per seed. Columns group the three variants within each dataset." }]));
children.push(scholarlyTable(
  ["Seed", "SST-2 Std", "SST-2 τ=2", { runs: [{ t: "SST-2 1/d" }, { t: "k", sub: true }] }, "IMDB Std", "IMDB τ=2", { runs: [{ t: "IMDB 1/d" }, { t: "k", sub: true }] }],
  S2R.CONV_ROWS, [16, 14, 14, 14, 14, 14, 14]));
children.push(tableCaption(17, [{ t: "Replica ablations E1–E4 (3 seeds, 5 epochs): test accuracy per seed. Columns group the four arms (Std / 1/d_k / LR-comp / mixed-scale) within each dataset." }]));
children.push(scholarlyTable(
  ["Seed", "SST-2 Std", "SST-2 1/dk", "SST-2 LR-c", "SST-2 MS", "IMDB Std", "IMDB 1/dk", "IMDB LR-c", "IMDB MS"],
  S2R.ABL_ROWS, [12, 11, 11, 11, 11, 11, 11, 11, 11]));
children.push(tableCaption(18, [{ runs: [{ t: "Replica ablation paired statistics versus the standard arm (3 seeds, 5 epochs)." }] }]));
children.push(scholarlyTable(
  ["Dataset", "Arm", "Mean (%)", { runs: [{ t: "Δ vs. std (pp)" }] }, "p"],
  [
    ["SST-2", "standard", "71.33", "—", "—"],
    ["", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] }, "71.14", "−0.19", "0.199"],
    ["", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " + LR comp." }] }, "70.03", "−1.30", "0.108"],
    ["", "mixed-scale", "71.37", "+0.04", "0.808"],
    ["IMDB", "standard", "70.28", "—", "—"],
    ["", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] }, "70.20", "−0.08", "0.321"],
    ["", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }, { t: " + LR comp." }] }, "70.01", "−0.27", "0.720"],
    ["", "mixed-scale", "70.33", "+0.05", "0.456"],
  ],
  [14, 30, 18, 22, 16]
));

// ===== S15 =====
children.push(h1("S15. Mask verification and leak probes"));
children.push(h2("S15.1 Verification tests"));
children.push(bodyR([
  rt("The corrected causal implementation is pinned by five assertions executed against a probe model with padding (real lengths 5 and 9 of a 16-position batch); all are exact float32 checks on the first-layer attention weights."),
]));
children.push(tableCaption(19, [{ t: "Causal-mask verification results." }]));
children.push(scholarlyTable(
  ["ID", "Property", "Result"],
  [
    ["V1", "attention weight on any future position j > i", "exactly 0.0"],
    ["V2", "attention weight on any padding position", "exactly 0.0"],
    ["V3", "every attention row sums to 1 over allowed positions", "0.99999988 ≤ Σ ≤ 1.00000012"],
    ["V4", "diagonal open on all real-token rows (predict x_(t+1) from x_(≤t))", "pass"],
    ["V5", "−∞ survives the [−50,50] clamp (mask applied after clamping)", "masked weights exactly 0"],
  ],
  [10, 55, 35]
));
children.push(h2("S15.2 Leak probes"));
children.push(bodyR([
  rt("Leak probes rerun the corrected LM with the causal mask deliberately removed (bidirectional attention over non-padding positions, matching the v1 failure mode), holding seeds, data order, and budgets fixed (3 seeds per variant)."),
]));
children.push(tableCaption(20, [{ runs: [{ t: "Mask-artifact probes: validation perplexity (mean over 3 matched seeds; all three variants within 0.3 ppl in every cell)." }] }]));
children.push(scholarlyTable(
  ["Budget", "Corrected causal mask", "Leaky (v1-style)"],
  [
    ["3 epochs", "872.8", "852.5"],
    ["10 epochs", "268.4", "252.4"],
    [{ runs: [{ t: "v1 reported (25,000 GPU steps, leaky)", i: true }] }, { runs: [{ t: "1.0809–1.0917", i: true }] }, ""],
  ],
  [30, 35, 35]
));
children.push(bodyR([
  rt("The leak lowers perplexity in both cells (direction confirmed), but the magnitude at replica budgets is a few percent because attention remains near-uniform; the v1 values require the 25,000-step regime in which attention can collapse onto the target position and fully exploit the leak. The corrected protocol additionally restores a physically meaningful scale: with an 11.6% unknown-token rate in the validation head and a 5,655-item vocabulary, perplexities in the high hundreds are the expected range for a model of this size trained on 96k tokens, whereas perplexity ≈ 1.08 (0.077 nats per token) is below the entropy of the validation unigram distribution itself."),
]));

// ===== S16 =====
children.push(h1("S16. Stage-2 diagnostics and LM convergence probe"));
children.push(tableCaption(21, [{ runs: [{ t: "Replica attention diagnostics (seed 1001, trained models). Entropy is the causal-horizon mean; the maximum possible entropy at sequence length 64 is ln 64 ≈ 4.158 nats. Gradient norms from a fresh validation batch." }] }]));
children.push(scholarlyTable(
  ["Metric", "Standard", "Temp (τ=2)", { runs: [{ t: "1/" }, { t: "d" }, { t: "k", sub: true }] }],
  [
    [{ t: "AG News classification replica (2 epochs)", b: true }, "", "", ""],
    ["Mean attention entropy (nats)", "4.048", "4.049", "4.049"],
    ["Max attention weight", "0.030", "0.027", "0.025"],
    ["Logits reaching the clamp", "0", "0", "0"],
    [{ runs: [{ t: "‖∂L/∂Q‖ + ‖∂L/∂K‖" }] }, "0.0476", "0.0285", "0.0093"],
    [{ runs: [{ t: "‖∂L/∂V‖" }] }, "1.6234", "1.9141", "1.8691"],
    [{ runs: [{ t: "‖W" }, { t: "q", sub: true }, { t: "‖, ‖W" }, { t: "k", sub: true }, { t: "‖ growth vs. init" }] }, "1.01×", "1.01×", "1.01×"],
    [{ t: "WikiText-2 replica (corrected causal mask, 3 epochs)", b: true }, "", "", ""],
    ["Mean attention entropy (nats)", "4.152", "4.157", "4.159"],
    ["Raw pre-division logit SD", "1.017", "1.168", "1.191"],
  ],
  [40, 20, 20, 20]
));
children.push(bodyR([
  rt("The 10-epoch LM convergence probe (3 seeds per variant, corrected mask) reaches validation perplexity 267.5 ± 6.4 (standard), 268.7 ± 5.3 (τ = 2), and 269.1 ± 5.3 (1/d_k); under the leaky mask the same seeds reach 251.2 ± 5.3, 252.0 ± 5.2, and 253.9 ± 5.5 respectively. Variant ordering is absent in every cell, at both 3 and 10 epochs, consistent with the near-uniform-regime account of Section 5.2 of the main text."),
]));

// ===== S17 =====
children.push(h1("S17. The divisor power law: derivation and validation"));

children.push(h2("S17.1 Derivation"));
children.push(bodyR([
  rt("Consider one attention row with pre-softmax logits X1, \u2026, Xn treated as i.i.d. with mean 0 and variance \u03c3\u00b2 (the raw, pre-division scale), and two divisors c1 > c2 with inverse scales b_i = 1/c_i. Writing P_b(j) = exp(b X_j) / \u03a3_k exp(b X_k) and expanding P_b around b = 0, to second order P_b1(j) \u2212 P_b2(j) = (b1 \u2212 b2)\u00b7(1/n)(X_j \u2212 X\u0304) + O(b\u00b3). Substituting into KL(P_b1 \u2016 P_b2) = \u03a3_j P_b1(j) log(P_b1(j)/P_b2(j)) and taking expectations over the logit distribution gives the power law of Eq. (5) of the main text: KL(P_b1\u2016P_b2) \u2248 \u00bd(b1 \u2212 b2)\u00b2\u03c3\u00b2(1 \u2212 1/n), where (1 \u2212 1/n) is the finite-population correction. Its validity range is b\u03c3 \u2272 0.5 per unit \u0394b; beyond that, softmax saturation makes the true divergence grow sub-quadratically, which the Monte Carlo grid exhibits at \u03c3 \u2265 12."),
]));

children.push(h2("S17.2 Monte Carlo validation"));
children.push(tableCaption(22, [{ t: "Monte Carlo validation of the power law (d_k = 32, n = 64; 200,000 replicates per point). Each cell: measured KL / law prediction, in nats." }]));
children.push(scholarlyTable(
  ["\u03c3", "std vs 1/d_k (MC / law)", "std vs \u03c4=2 (MC / law)", "1/d_k vs \u03c4=2 (MC / law)"],
  [
    ["0.5", "0.0026 / 0.0026", "0.0010 / 0.0010", "0.0004 / 0.0004"],
    ["1", "0.0104 / 0.0104", "0.0038 / 0.0039", "0.0016 / 0.0016"],
    ["2", "0.0416 / 0.0417", "0.0153 / 0.0154", "0.0064 / 0.0064"],
    ["4", "0.1642 / 0.1668", "0.0602 / 0.0615", "0.0257 / 0.0257"],
    ["8", "0.5941 / 0.6671", "0.2107 / 0.2461", "0.1019 / 0.1028"],
    ["16", "1.3632 / 2.6684", "0.3897 / 0.9844", "0.3882 / 0.4114"],
  ],
  [10, 30, 30, 30]
));

children.push(h2("S17.3 Entropy and bite thresholds"));
children.push(tableCaption(23, [{ t: "Attention entropy and expected maximum weight versus raw logit scale (d_k = 32, n = 64; 200,000 replicates). Uniform ceiling: ln 64 \u2248 4.159 nats, 1/64 \u2248 0.0156." }]));
children.push(scholarlyTable(
  ["\u03c3", "H std", "H \u03c4=2", "H 1/d_k", "maxw std", "maxw \u03c4=2", "maxw 1/d_k"],
  [
    ["0.5", "4.155", "4.158", "4.159", "0.019", "0.017", "0.016"],
    ["1", "4.144", "4.155", "4.158", "0.023", "0.019", "0.017"],
    ["2", "4.098", "4.144", "4.157", "0.034", "0.023", "0.018"],
    ["4", "3.916", "4.098", "4.151", "0.067", "0.034", "0.021"],
    ["8", "3.270", "3.917", "4.128", "0.182", "0.067", "0.027"],
    ["16", "1.924", "3.268", "4.037", "0.452", "0.183", "0.046"],
  ],
  [10, 15, 15, 15, 15, 15, 15]
));
children.push(bodyR([
  rt("Inverting the power law gives the operating point at which a divisor pair first produces a target divergence (d_k = 32, n = 64): \u03c3 \u2248 0.98 for KL = 0.01 nats and \u03c3 \u2248 3.10 for KL = 0.1 nats (standard vs 1/d_k); \u03c3 \u2248 1.61 and 5.10 for the standard-vs-\u03c4 pair. The Stage-2 replica grids sit at measured \u03c3 \u2248 0.3\u20131.1 (KL \u2264 0.006 nats), the logit-gain sweep reaches \u03c3 = 9.2 (KL = 0.64 nats), and pretrained GPT-2 sits at \u03c3 \u2248 45\u201363 (KL = 1.4\u20131.5 nats per row)."),
]));

// ===== S18 =====
children.push(h1("S18. Stage-3 operating-point records"));

children.push(h2("S18.1 Frontier-stabiliser bridge arms"));
children.push(tableCaption(24, [{ t: "Bridge arms (AG News replica, 4 matched seeds, 2 epochs). \u03c3_post is the post-divisor logit SD under the arm's own divisor; KL is the measured standard-vs-1/d_k divergence on the same logits." }]));
children.push(scholarlyTable(
  ["Arm", "Acc (%)", "\u03c3_post", "H (nats)", "KL (nats)"],
  [
    ["qk_std (QK-Norm, standard divisor)", "78.21", "1.051", "3.349", "0.2925"],
    ["qk_dk (QK-Norm, 1/d_k)", "78.16", "0.192", "3.771", "0.2924"],
    ["qk_temp (QK-Norm, \u03c4=2)", "78.28", "0.538", "3.668", "0.2923"],
    ["cap3_std_g32 (soft cap 3, gain 32, std)", { t: "76.81", b: true }, "1.629", "2.808", "0.6418"],
    ["cap3_dk_g32 (soft cap 3, gain 32, 1/d_k)", "78.21", "0.320", "3.744", "0.6779"],
  ],
  [36, 13, 15, 17, 19]
));

children.push(h2("S18.2 Regime-restored training trajectories"));
children.push(tableCaption(25, [{ t: "Regime-restored AG News (6,000 training examples, 8 epochs, 3 matched seeds, mean over seeds). Top: d_k = 32 (4 heads); the operating point stays inert (KL \u2264 0.006) for the entire budget. Bottom: d_k = 128 (1 head); the larger head drives \u03c3 to \u2248 5 and the Stage-1 ordering re-emerges in direction." }]));
children.push(scholarlyTable(
  ["Variant", "\u03c3 ep1", "KL ep1", "acc ep1", "\u03c3 ep4", "KL ep4", "acc ep4", "\u03c3 ep8", "KL ep8", "acc ep8"],
  [
    [{ t: "d_k = 32 (4 heads)", b: true }, "", "", "", "", "", "", "", "", ""],
    ["std", "0.42", "0.0011", "84.93", "0.90", "0.0053", "84.66", "1.11", "0.0060", "85.68"],
    ["1/d_k", "0.41", "0.0011", "84.87", "0.85", "0.0057", "85.53", "1.01", "0.0068", "85.74"],
    ["\u03c4=2", "0.41", "0.0011", "84.88", "0.94", "0.0058", "85.11", "1.11", "0.0066", "85.81"],
    [{ t: "d_k = 128 (1 head)", b: true }, "", "", "", "", "", "", "", "", ""],
    ["std", "1.46", "0.0040", "84.91", "3.54", "0.0226", "85.05", "4.59", "0.0262", "85.98"],
    ["1/d_k", "1.44", "0.0042", "84.87", "4.22", "0.0739", "85.08", "5.31", "0.0857", "85.56"],
    ["\u03c4=2", "1.45", "0.0041", "84.89", "4.07", "0.0446", "85.26", "5.09", "0.0487", "85.46"],
  ],
  [16, 9, 9, 9, 9, 9, 9, 9, 10, 10]
));
children.push(bodyR([
  rt("Final-epoch paired contrasts (3 seeds, uncorrected): at d_k = 32, standard \u2212 1/d_k = \u22120.06 points (p = 0.51) and standard \u2212 \u03c4 = 2 = \u22120.13 points (p = 0.42); at d_k = 128, +0.42 points (p = 0.31) and +0.52 points (p = 0.066) respectively."),
]));

children.push(h2("S18.3 LM logit gain (sequence length 128)"));
children.push(tableCaption(26, [{ t: "LM logit-gain arm (WikiText-2 replica at sequence length 128, 3 seeds, 3 epochs, 8,000-item vocabulary). Validation perplexity is indistinguishable across variants at both gains; even g = 4 leaves the operating point inert." }]));
children.push(scholarlyTable(
  ["Gain", "Standard", "\u03c4=2", "1/d_k", "\u03c3_raw / KL"],
  [
    ["g = 1", "2160.7 \u00b1 47.0", "2160.8 \u00b1 47.5", "2160.9 \u00b1 46.5", "0.29 / 0.0003"],
    ["g = 4", "2160.3 \u00b1 48.5", "2160.6 \u00b1 47.5", "2160.8 \u00b1 46.8", "1.15 / 0.0052"],
  ],
  [12, 24, 24, 24, 16]
));

children.push(h2("S18.4 Frontier probe: per-layer diagnostics"));
children.push(bodyR([
  rt("The pretrained-decoder probe (Table 12 of the main text) records, per layer, the raw pre-division logit SD over causally valid entries, the per-row attention entropy under each divisor, and the per-row KL divergence between the attention maps the same logits induce under the standard and alternative divisors (softmax restricted to the causal horizon). GPT-2 (124M): layer 1-6 entropies of 2.45, 2.93, 2.15, 1.67, 1.62, 1.38 nats under the training divisor, rising to 3.81, 3.86, 3.72, 3.65, 3.61, 3.80 under 1/d_k; the mean standard-vs-1/d_k divergence is 1.42 nats per row (quadratic law prediction 31.5 nats, the saturation gap at b\u03c3 \u226b 1). GPT-2-medium (355M): mean \u03c3_raw = 44.8, mean divergence 1.51 nats. Absolute perplexity in this probe depends on the windowing protocol (non-overlapping 128-token windows, no cache carry-over); the contrasts between divisors do not."),
]));

children.push(h2("S18.5 Reproduction"));
children.push(bodyR([
  rt("Stage-3 phases run through the same resume-safe runner as Stage 2: python scripts/experiments/runner.py --phase {gain,bridge,regime,regime128,lmgain}; aggregation via scripts/experiments/agg_stage3.py; theory via scripts/experiments/regime_law.py; and the pretrained-decoder probe via scripts/experiments/frontier_gpt2.py (CPU) or scripts/frontier/frontier_gpu_1b7b.py (GPU, 1B-7B). Raw records: data/results/results.jsonl, stage3_theory.json, stage3_operatingpoint.json, frontier_gpt2.json."),
], { noIndent: false }));

// ===== References =====
children.push(h1("References"));
children.push(refPara(1, "Loshchilov, I., and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019."));
children.push(refPara(2, "Cohen, J. Statistical Power Analysis for the Behavioral Sciences, 2nd edition. Lawrence Erlbaum Associates, Hillsdale, NJ, 1988."));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Raza Hasan",
  title: "Supplementary Material: A Seed-Matched Empirical Comparison of Scaling Functions in Transformer Self-Attention",
  description: "Supplementary material: full specifications, per-seed results, diagnostics, and reproduction instructions",
  styles: { default: { document: { run: { font: TNR, size: 24 } } } },
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
          children: [new TextRun({ text: "Supplementary material — scaling functions in Transformer self-attention", italics: true, size: 18, color: "666666", font: TNR })],
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
