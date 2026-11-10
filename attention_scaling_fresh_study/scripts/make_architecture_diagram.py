"""
Architecture diagram: Pre-LN Transformer encoder with the three attention
scaling variants and both task heads. Schematic (not data-driven).
Outputs figures/fig_architecture.png + .pdf.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({'font.size': 8.5, 'figure.dpi': 300, 'savefig.dpi': 300})

fig, ax = plt.subplots(figsize=(9.4, 7.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis('off')

C_IN   = '#dbe9f6'   # input blocks
C_BLK  = '#fde9d9'   # attention core
C_HEAD = '#e6e0ec'   # heads
C_EDGE = '#4472a8'
ARROW  = '#555555'

def box(x, y, w, h, label, fc, fs=8.3, lw=1.1, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                fc=fc, ec=C_EDGE, lw=lw))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')

def arrow(x1, y1, x2, y2, lw=1.2, color=ARROW, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=11, lw=lw, color=color, linestyle=ls))

ax.text(5, 8.32, 'Transformer encoder with attention logit-scale variants (study architecture)',
        ha='center', fontsize=10.5, fontweight='bold')

# ---------------- Input row ----------------
y0 = 7.5
box(0.4, y0, 2.0, 0.6, 'Token ids\n+ padding mask', C_IN)
box(2.9, y0, 2.0, 0.6, 'Token embedding\n(vocab → 128)', C_IN)
box(5.4, y0, 2.0, 0.6, 'Learned positional\nembedding (+)', C_IN)
arrow(2.4, y0+0.3, 2.9, y0+0.3); arrow(4.9, y0+0.3, 5.4, y0+0.3)
arrow(7.4, y0+0.3, 8.15, y0+0.3)
ax.text(8.3, y0+0.3, '→ layer 1', fontsize=8, va='center', color='#555555')

# ---------------- Attention scaling detail strip ----------------
sy = 6.15
ax.add_patch(FancyBboxPatch((0.4, sy), 9.2, 0.95, boxstyle='round,pad=0.05',
                            fc='#eef4fb', ec='#2f5b8f', lw=1.3))
ax.text(0.62, sy+0.72, 'Attention logit scaling — the manipulated factor (applied in every attention block, all heads):',
        fontsize=8.6, fontweight='bold', color='#2f5b8f', va='center')
ax.text(0.62, sy+0.42,
        r'logits = QK$^{\top}$/s    with    s $\in$ { $\sqrt{d_k}$ (standard),  $d_k$ (1/d$_k$),  $\tau\sqrt{d_k}$ (temp, $\tau \in \{0.5, 1.0, 1.5, 2.0\}$) }',
        fontsize=8.6, va='center')
ax.text(0.62, sy+0.15,
        'A = softmax(clamp(logits, ±50) + mask);  output = A·V      — bit-identical init & data order per seed; zero extra parameters',
        fontsize=7.8, va='center', color='#666666', style='italic')

# ---------------- Encoder layers ----------------
def encoder_layer(ytop, tag):
    H = 1.42
    ax.add_patch(Rectangle((0.4, ytop), 9.2, H, fc='#f7f7f7', ec='#bbbbbb', lw=1.0, ls='--'))
    ax.text(0.55, ytop + H - 0.14, tag, fontsize=8.5, color='#666666', va='center')
    yb = ytop + 0.30                      # block vertical center base
    box(0.62, yb, 1.15, 0.6, 'LayerNorm\n(pre)', C_IN, fs=7.6)
    box(2.05, yb - 0.07, 2.6, 0.74, 'Multi-head self-attention\n(4 heads, $d_k$ = 32)', C_BLK, fs=8.0)
    box(4.95, yb, 1.15, 0.6, 'LayerNorm\n(pre)', C_IN, fs=7.6)
    box(6.4, yb - 0.02, 2.95, 0.66, 'FFN  128 → 256\nGELU → 128', C_IN, fs=7.8)
    cx = yb + 0.3
    arrow(1.77, cx, 2.05, cx)
    arrow(4.65, cx, 4.95, cx)
    arrow(6.10, cx, 6.40, cx)
    # residual connectors
    ax.annotate('', xy=(2.05, yb - 0.16), xytext=(1.2, yb - 0.16),
                arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#999999',
                                connectionstyle='arc3,rad=-0.35', linestyle=(0, (2, 2))))
    ax.text(1.62, yb - 0.34, '+res', fontsize=6.6, color='#999999', ha='center')
    ax.annotate('', xy=(6.4, yb - 0.16), xytext=(5.5, yb - 0.16),
                arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#999999',
                                connectionstyle='arc3,rad=-0.35', linestyle=(0, (2, 2))))
    ax.text(5.95, yb - 0.34, '+res', fontsize=6.6, color='#999999', ha='center')
    return cx

l1 = encoder_layer(4.35, 'Encoder layer 1')
l2 = encoder_layer(2.45, 'Encoder layer 2')
# input -> layer1, strip -> attention blocks, layer1 -> layer2
arrow(5.0, y0, 5.0, 5.77 + 0.3)
arrow(5.0, 5.77 + 0.3, 5.0, 5.77 + 0.3)
arrow(3.35, sy, 3.35, 5.62, lw=1.0, color='#2f5b8f')
ax.text(3.45, 5.86, 'scaled logits in every block', fontsize=7.2, color='#2f5b8f')
arrow(5.0, 4.35, 5.0, 3.87)

# ---------------- Heads ----------------
hy = 1.15
box(2.05, hy, 2.2, 0.72, 'Masked mean\npooling', C_HEAD)
arrow(5.0, 2.45, 3.15, hy + 0.72)
box(4.75, hy, 2.0, 0.72, 'Linear\n→ num classes', C_HEAD)
arrow(4.25, hy + 0.36, 4.75, hy + 0.36)
box(7.35, hy, 2.05, 0.72, 'LM head (linear,\nweight-tied embedding)', C_HEAD)
arrow(5.0, 2.45, 8.35, hy + 0.78, lw=0.9, ls=(0, (3, 2)))
ax.text(3.15, hy - 0.24, 'classification: AG News / SST-2 / IMDB', fontsize=7.4, ha='center', color='#555555')
ax.text(8.35, hy - 0.24, 'language modelling: WikiText-2', fontsize=7.4, ha='center', color='#555555')

fig.tight_layout()
import os
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig_architecture.png', bbox_inches='tight')
fig.savefig('figures/fig_architecture.pdf', bbox_inches='tight')
print('saved figures/fig_architecture.png/.pdf')
