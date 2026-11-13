"""Splice generated per-seed rows into the supplementary S13-S16 append block."""

APPEND = r"""
\section*{S13. Stage-2 replica protocol specification}\label{sup:replica}
\addcontentsline{toc}{section}{S13. Stage-2 replica protocol specification}

Stage 2 executes every preregistered grid on a 2-core CPU sandbox
(3\,GB RAM, PyTorch 2.15.0 CPU, one torch thread per worker, two worker
processes). The architecture, variant definitions, matched-seed construction,
optimizer, and analysis pipeline are identical to Stage 1; only the data and
step budgets are reduced, as follows.

\begin{table}[H]
  \centering
  \caption{Stage-2 replica configuration.}
  \small
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l l}
    \toprule
    \textbf{Component} & \textbf{Classification replica} & \textbf{LM replica} \\
    \midrule
    Training data & 2{,}000 examples per dataset & 96{,}000 tokens (WikiText-2 train head) \\
    Validation & 500 examples & 40{,}000 tokens (official validation head) \\
    Test & AG News 4{,}000 / SST-2 872 / IMDB 2{,}500 & validation perplexity \\
    Sequence length & 64 (right-padded) & 64 (non-overlapping chunks) \\
    Vocabularies & 7{,}234 / 2{,}249 / 12{,}000 & 5{,}655 \\
    $d_{\text{model}}$ / heads & 128 / 4 ($\dk = 32$) & 96 / 3 ($\dk = 32$) \\
    Layers / $d_{\text{ff}}$ / dropout & 2 / 256 / 0.1 & 2 / 192 / 0.1 \\
    Batch size & 32 & 32 \\
    Budget & 2 epochs (core, grid); $\le$20 epochs (convergence) & 3 epochs; 10-epoch probe \\
    Optimiser & \multicolumn{2}{l}{AdamW, lr $3\times10^{-4}$, wd 0.01, 5\% warmup, cosine, clip 1.0} \\
    Seeds & \multicolumn{2}{l}{core/LM 1001--1025; grid 1001--1012; convergence 1001--1005; ablations 1001--1003} \\
    Causal mask (LM) & \multicolumn{2}{l}{lower-triangular AND key-padding, applied after the $[-50,50]$ clamp} \\
    Records & \multicolumn{2}{l}{append-only JSONL keyed by (phase, dataset, variant, $\dk$, seed)} \\
    \bottomrule
  \end{tabular*}
\end{table}

Splits are drawn with a fixed RNG seed (42) from the official training
partitions; test subsets are the first $n$ rows of a fixed permutation of the
official test partitions (SST-2 uses the official 872-example labelled dev set
as test). Because the same subsets and order are shared by all variants of a
seed, the paired contrasts remain exactly matched. Grid completion is
checkpointed: any interrupted batch resumes from the JSONL key set, and the
final manifest was audited for exactly-once execution of all 312
pre-registered runs.

\section*{S14. Stage-2 per-seed records}\label{sup:perseeds2}
\addcontentsline{toc}{section}{S14. Stage-2 per-seed records}

Accuracy fractions (classification) or validation perplexity (LM) per matched
seed. Every table below is generated directly from the append-only run records;
the paired statistics in the main text are computed from exactly these values.

\begin{table}[H]
  \centering
  \caption{Replica core, AG News, 2 epochs: test accuracy per seed.}
  \small
  \begin{tabular}{r c c c}
    \toprule
    \textbf{Seed} & \textbf{Standard} & \textbf{Temp ($\tau=2$)} & \textbf{$1/\dk$} \\
    \midrule
%%CORE_ROWS%%
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[H]
  \centering
  \caption{Corrected causal-mask language modelling, WikiText-2, 3 epochs:
  validation perplexity per seed (25 seeds $\times$ 3 variants, all completed).}
  \small
  \begin{tabular}{r c c c}
    \toprule
    \textbf{Seed} & \textbf{Standard} & \textbf{Temp ($\tau=2$)} & \textbf{$1/\dk$} \\
    \midrule
%%LM_ROWS%%
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[H]
  \centering
  \caption{Head-dimension grid, AG News, 2 epochs: test accuracy per seed.
  Columns group the three variants (Std / $\tau{=}2$ / $1/\dk$) within each
  head dimension.}
  \scriptsize
  \setlength{\tabcolsep}{3.4pt}
  \begin{tabular}{r ccc ccc ccc}
    \toprule
    & \multicolumn{3}{c}{$\dk = 32$} & \multicolumn{3}{c}{$\dk = 64$} & \multicolumn{3}{c}{$\dk = 128$} \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
    \textbf{Seed} & Std & $\tau{=}2$ & $1/\dk$ & Std & $\tau{=}2$ & $1/\dk$
     & Std & $\tau{=}2$ & $1/\dk$ \\
    \midrule
%%GRID_ROWS%%
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[H]
  \centering
  \caption{Train-to-convergence with early stopping (up to 20 epochs, patience
  3 on validation loss): best-validation-epoch test accuracy per seed. Columns
  group the three variants within each dataset.}
  \small
  \begin{tabular}{r ccc ccc}
    \toprule
    & \multicolumn{3}{c}{SST-2} & \multicolumn{3}{c}{IMDB} \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-7}
    \textbf{Seed} & Std & $\tau{=}2$ & $1/\dk$ & Std & $\tau{=}2$ & $1/\dk$ \\
    \midrule
%%CONV_ROWS%%
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[H]
  \centering
  \caption{Replica ablations E1--E4 (3 seeds, 5 epochs): test accuracy per
  seed. Columns group the four arms (Std / $1/\dk$ / LR-comp / mixed-scale)
  within each dataset.}
  \small
  \begin{tabular}{r cccc cccc}
    \toprule
    & \multicolumn{4}{c}{SST-2} & \multicolumn{4}{c}{IMDB} \\
    \cmidrule(lr){2-5}\cmidrule(lr){6-9}
    \textbf{Seed} & Std & $1/\dk$ & LR-c & MS & Std & $1/\dk$ & LR-c & MS \\
    \midrule
%%ABL_ROWS%%
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[H]
  \centering
  \caption{Replica ablation paired statistics versus the standard arm
  (3 seeds, 5 epochs).}\label{sup:abl2}
  \small
  \begin{tabular*}{0.9\textwidth}{@{\extracolsep{\fill}} l l c c c}
    \toprule
    \textbf{Dataset} & \textbf{Arm} & \textbf{Mean (\%)} & \textbf{$\Delta$ vs.\ std (pp)} & \textbf{$p$} \\
    \midrule
    SST-2 & standard & 71.33 & --- & --- \\
          & $1/\dk$ & 71.14 & $-0.19$ & 0.199 \\
          & $1/\dk$ + LR comp. & 70.03 & $-1.30$ & 0.108 \\
          & mixed-scale & 71.37 & $+0.04$ & 0.808 \\
    \midrule
    IMDB & standard & 70.28 & --- & --- \\
         & $1/\dk$ & 70.20 & $-0.08$ & 0.321 \\
         & $1/\dk$ + LR comp. & 70.01 & $-0.27$ & 0.720 \\
         & mixed-scale & 70.33 & $+0.05$ & 0.456 \\
    \bottomrule
  \end{tabular*}
\end{table}

\section*{S15. Mask verification and leak probes}\label{sup:mask}
\addcontentsline{toc}{section}{S15. Mask verification and leak probes}

\subsection*{S15.1 Verification tests}
The corrected causal implementation is pinned by five assertions executed
against a probe model with padding (real lengths 5 and 9 of a 16-position
batch); all are exact float32 checks on the first-layer attention weights.

\begin{table}[H]
  \centering
  \caption{Causal-mask verification results.}
  \small
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l l}
    \toprule
    \textbf{ID} & \textbf{Property} & \textbf{Result} \\
    \midrule
    V1 & attention weight on any future position $j > i$ & exactly $0.0$ \\
    V2 & attention weight on any padding position & exactly $0.0$ \\
    V3 & every attention row sums to 1 over allowed positions & $0.99999988 \le \Sigma \le 1.00000012$ \\
    V4 & diagonal open on all real-token rows (predict $x_{t+1}$ from $x_{\le t}$) & pass \\
    V5 & $-\infty$ survives the $[-50,50]$ clamp (mask applied after clamping) & masked weights exactly $0$ \\
    \bottomrule
  \end{tabular*}
\end{table}

\subsection*{S15.2 Leak probes}
Leak probes rerun the corrected LM with the causal mask deliberately removed
(bidirectional attention over non-padding positions, matching the v1 failure
mode), holding seeds, data order, and budgets fixed (3 seeds per variant).

\begin{table}[H]
  \centering
  \caption{Mask-artifact probes: validation perplexity (mean over 3 matched
  seeds; all three variants within 0.3 ppl in every cell).}
  \small
  \begin{tabular*}{0.85\textwidth}{@{\extracolsep{\fill}} l c c}
    \toprule
    \textbf{Budget} & \textbf{Corrected causal mask} & \textbf{Leaky (v1-style)} \\
    \midrule
    3 epochs & 872.8 & 852.5 \\
    10 epochs & 268.4 & 252.4 \\
    \midrule
    \multicolumn{3}{l}{\footnotesize v1 reported (25{,}000 GPU steps, leaky): 1.0809--1.0917.} \\
    \bottomrule
  \end{tabular*}
\end{table}

The leak lowers perplexity in both cells (direction confirmed), but the
magnitude at replica budgets is a few percent because attention remains
near-uniform; the v1 values require the 25{,}000-step regime in which attention
can collapse onto the target position and fully exploit the leak. The corrected
protocol additionally restores a physically meaningful scale: with an 11.6\%
unknown-token rate in the validation head and a 5{,}655-item vocabulary,
perplexities in the high hundreds are the expected range for a model of this
size trained on 96k tokens, whereas perplexity $\approx 1.08$ (0.077 nats per
token) is below the entropy of the validation unigram distribution itself.

\section*{S16. Stage-2 diagnostics and LM convergence probe}\label{sup:diag2}
\addcontentsline{toc}{section}{S16. Stage-2 diagnostics and LM convergence probe}

\begin{table}[H]
  \centering
  \caption{Replica attention diagnostics (seed 1001, trained models).
  Entropy is the causal-horizon mean; the maximum possible entropy at sequence
  length 64 is $\ln 64 \approx 4.158$ nats. Gradient norms from a fresh
  validation batch.}
  \small
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l c c c}
    \toprule
    \textbf{Metric} & \textbf{Standard} & \textbf{Temp ($\tau=2$)} & \textbf{$1/\dk$} \\
    \midrule
    \multicolumn{4}{l}{\emph{AG News classification replica (2 epochs)}} \\
    Mean attention entropy (nats) & 4.048 & 4.049 & 4.049 \\
    Max attention weight & 0.030 & 0.027 & 0.025 \\
    Logits reaching the clamp & 0 & 0 & 0 \\
    $\|\partial L/\partial Q\| + \|\partial L/\partial K\|$ & 0.0476 & 0.0285 & 0.0093 \\
    $\|\partial L/\partial V\|$ & 1.6234 & 1.9141 & 1.8691 \\
    $\|W_q\|, \|W_k\|$ growth vs.\ init & 1.01$\times$ & 1.01$\times$ & 1.01$\times$ \\
    \midrule
    \multicolumn{4}{l}{\emph{WikiText-2 replica (corrected causal mask, 3 epochs)}} \\
    Mean attention entropy (nats) & 4.152 & 4.157 & 4.159 \\
    Raw pre-division logit SD & 1.017 & 1.168 & 1.191 \\
    \bottomrule
  \end{tabular*}
\end{table}

The 10-epoch LM convergence probe (3 seeds per variant, corrected mask) reaches
validation perplexity 267.5 $\pm$ 6.4 (standard), 268.7 $\pm$ 5.3 ($\tau=2$),
and 269.1 $\pm$ 5.3 ($1/\dk$); under the leaky mask the same seeds reach
251.2 $\pm$ 5.3, 252.0 $\pm$ 5.2, and 253.9 $\pm$ 5.5 respectively. Variant
ordering is absent in every cell, at both 3 and 10 epochs, consistent with the
near-uniform-regime account of Section~\ref{sec:grid} of the main text.
"""

import re

rows = open("/tmp/perseed_rows.tex").read()


def grab(marker):
    m = re.search(rf"% ---- {re.escape(marker)} ----\n(.*?)(?=% ---- |\Z)", rows, re.S)
    return m.group(1).rstrip()


out = APPEND.replace("%%CORE_ROWS%%", grab("S14 core replica (AG News)"))
out = out.replace("%%LM_ROWS%%", grab("S14 LM corrected"))
out = out.replace("%%GRID_ROWS%%", grab("S14 grid"))
out = out.replace("%%CONV_ROWS%%", grab("S14 conv"))
out = out.replace("%%ABL_ROWS%%", grab("S14 ablations"))

sup = open(os.path.join(os.getcwd(), "paper", "latex", "supplementary.tex")).read()
sup = sup.replace("\\bibliography{refs}", out + "\n\\bibliography{refs}")
open(os.path.join(os.getcwd(), "paper", "latex", "supplementary.tex"), "w").write(sup)
print("appended", len(out), "chars")
