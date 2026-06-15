import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis('off')

def box(ax, x, y, w, h, lines, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.1', lw=lw, edgecolor='black', facecolor='white'))
    n = len(lines)
    step = h / (n + 1)
    for i, (txt, fs, fw) in enumerate(lines):
        ax.text(x + w/2, y + h - step*(i+1), txt,
                ha='center', va='center', fontsize=fs,
                fontweight=fw, multialignment='center', linespacing=1.25)

def arr(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color='black', lw=0.9))

def variant_label(ax, y, txt):
    ax.text(6.65, y, txt, ha='center', va='center', fontsize=8,
            color='#333333', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

ax.text(6.65, 8.65, 'MedGemma 27B  (Ollama, local inference)',
        ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.8))

box(ax, 0.3, 5.0, 3.0, 3.0, [
    ('Radiology report', 9, 'bold'),
    ('Findings', 8, 'normal'),
    ('Conclusion', 8, 'normal'),
    ('Combined', 8, 'normal'),
])
box(ax, 0.3, 2.2, 3.0, 2.4, [
    ('GP referral text', 9, 'bold'),
    ('Clinical background', 8, 'normal'),
    ('Clinical question', 8, 'normal'),
])

box(ax, 10.0, 6.7, 3.7, 1.4, [
    ('Radiological outcome', 8.5, 'bold'),
    ('findings  |  conclusion  |  combined', 7.5, 'normal'),
    ('abnormal  /  normal  /  unknown', 7.5, 'normal'),
])
box(ax, 10.0, 5.1, 3.7, 1.3, [
    ('KL grade  (0-4)', 8.5, 'bold'),
    ('combined report', 7.5, 'normal'),
    ('grade 0-4  /  unknown', 7.5, 'normal'),
])
box(ax, 10.0, 3.5, 3.7, 1.3, [
    ('NHG alignment', 8.5, 'bold'),
    ('GP referral text', 7.5, 'normal'),
    ('ja  /  nee  /  unknown', 7.5, 'normal'),
])
box(ax, 10.0, 2.2, 3.7, 1.0, [
    ('Trauma  /  fracture  /  prosthesis', 8, 'bold'),
    ('GP referral text', 7.5, 'normal'),
])

arr(ax, 3.3, 7.1, 10.0, 7.4);   variant_label(ax, 7.55, '3 prompt variants')
arr(ax, 3.3, 6.0, 10.0, 5.75);  variant_label(ax, 6.15, '3 prompt variants')
arr(ax, 3.3, 3.2, 10.0, 4.15);  variant_label(ax, 3.85, '3 prompt variants')
arr(ax, 3.3, 2.7, 10.0, 2.7);   variant_label(ax, 2.85, '1 prompt  (zero-shot)')

box(ax, 2.5, 0.15, 9.0, 1.6, [
    ('Evaluation against gold standard  (n = 384)', 9, 'bold'),
    ('Manual expert labels  .  2 independent raters  +  consensus meeting', 7.5, 'normal'),
    ('Accuracy  .  macro F1  .  Cohen kappa  .  bootstrap 95% CI', 7.5, 'normal'),
    ('Primary variant selected on label distribution match; tiebreak on macro F1', 7.5, 'normal'),
], lw=1.0)
arr(ax, 11.85, 2.2, 11.85, 1.75)

out = Path(__file__).parent / 'figures' / 'fig_llm_overview'
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out) + '.png', dpi=180, bbox_inches='tight', facecolor='white')
fig.savefig(str(out) + '.pdf', bbox_inches='tight', facecolor='white')
print(f'Saved: {out}.png  and  {out}.pdf')
