"""Physical-size publication style and reusable saved-value plot layouts."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import fitz
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

WIDTH = 6.35
TEXT, TITLE, PANEL = 10, 10.5, 13
BLUE, GREEN, ORANGE, PURPLE = '#17689A', '#168775', '#C6602A', '#8856A7'
GRAY = '#666666'
SEEDS = [(BLUE, 'o'), (ORANGE, 's'), (GREEN, '^')]
METHODS = {'PCA': (GREEN, '^'), 'UMAP': (ORANGE, 's'),
           'Diffusion': (PURPLE, 'D'), 'Slingshot': ('#242424', 'o'),
           'scRL': (BLUE, 'o'), 'Palantir': ('#242424', 'o')}


def configure():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': TEXT,
        'axes.labelsize': TEXT, 'axes.titlesize': TITLE,
        'xtick.labelsize': TEXT, 'ytick.labelsize': TEXT,
        'legend.fontsize': TEXT, 'axes.linewidth': .6,
        'axes.spines.right': False, 'axes.spines.top': False,
        'lines.linewidth': 1.6, 'lines.markersize': 4.6,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'hematopoietic-readouts', 'savefig.facecolor': 'white'})


def axes(fig, x, y, w, h):
    """Place an axes in inches; reserve title/legend space explicitly."""
    return fig.add_axes([x/WIDTH, y/fig.get_figheight(), w/WIDTH, h/fig.get_figheight()])


def panel(ax, letter, heading, offset=-22):
    ax.set_title(heading, pad=13, fontsize=TITLE, fontweight='normal')
    item = ax.annotate(letter.lower(), (0, 1), xycoords='axes fraction',
        xytext=(offset, 13), textcoords='offset points',
        fontsize=PANEL, fontweight='bold', va='bottom', annotation_clip=False)
    item._panel_letter = True


def correlation_axis(ax, axis='x'):
    if axis == 'x':
        ax.set_xlim(-1, 1); ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.axvline(0, color='#A5A5A5', lw=.7, ls='--', zorder=0)
        ax.set_xlabel('Spearman ρ')
    else:
        ax.set_ylim(-1, 1); ax.set_yticks([-1, 0, 1])
        ax.axhline(0, color='#A5A5A5', lw=.7, ls='--', zorder=0)
    ax.grid(axis=axis, color='#E7E7E7', lw=.6); ax.set_axisbelow(True)


def raw_fixed(ax, y, raw, fixed, raw_all, color, marker):
    """Connect estimates on the SAME eligible population, without end caps."""
    ax.plot([raw, fixed], [y, y], color=color, lw=1.1, zorder=1)
    ax.scatter(raw, y, marker=marker, s=35, facecolor='white', edgecolor=color, zorder=3)
    ax.scatter(fixed, y, marker=marker, s=35, facecolor=color, edgecolor=color, zorder=4)
    ax.scatter(raw_all, y+.09, marker='x', s=22, color=GRAY, lw=.8, zorder=5)


def matrix(ax, values, rows, columns, decimals=3):
    values = np.asarray(values, dtype=float)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('readable_signed',
        plt.get_cmap('RdBu_r')(np.linspace(.20, .80, 256)))
    cmap.set_bad('#EEEEEE')
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1,
              aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0, pad=6)
    for y in range(len(rows)):
        for x in range(len(columns)):
            v = values[y, x]
            ax.text(x, y, f'{v:.{decimals}f}' if np.isfinite(v) else 'invalid',
                    ha='center', va='center', fontsize=TEXT)
    for spine in ax.spines.values(): spine.set_visible(False)


def count_columns(fig, ax, labels, counts, coverage, x_name=.08, x_n=1.82, x_coverage=2.37):
    """Aligned metadata outside a shared numerical plotting region."""
    y_to_fig = ax.transData + fig.transFigure.inverted()
    for y, label, n, cov in zip(range(len(labels)), labels, counts, coverage):
        yy = y_to_fig.transform((0, y))[1]
        for x, text, ha in [(x_name, label, 'left'), (x_n, f'{int(n):,}', 'right'),
                             (x_coverage, f'{cov:.1%}', 'right')]:
            fig.text(x/WIDTH, yy, text, ha=ha, va='center', fontsize=TEXT)
    yy = ax.get_position().y1 + 8/(72*fig.get_figheight())
    for x, text, ha in [(x_name, 'Cohort / end', 'left'), (x_n, 'n', 'right'),
                         (x_coverage, 'Coverage', 'right')]:
        fig.text(x/WIDTH, yy, text, ha=ha, fontsize=TEXT)


def trajectory(ax, checkpoints, series, ylabel, correlation=False):
    for seed, values in enumerate(series):
        color, marker = SEEDS[seed]
        ax.plot(checkpoints, values, color=color, marker=marker, ms=3.3,
                lw=1.1, label=f'Seed {seed}')
    ax.axvline(5000, color=GRAY, lw=.8, ls=':')
    ax.set_xlim(0, 10000); ax.set_xticks([0, 5000, 10000], ['0', '5,000', '10,000'])
    ax.set_xlabel('Training episode'); ax.set_ylabel(ylabel)
    if correlation: correlation_axis(ax, 'y')
    else: ax.grid(axis='y', color='#E7E7E7', lw=.6)


def seed_legend(fig, y=.015):
    fig.legend(handles=[Line2D([], [], color=c, marker=m, label=f'Seed {i}')
        for i, (c, m) in enumerate(SEEDS)], loc='lower center',
        bbox_to_anchor=(.54, y), frameon=False, ncol=3, handlelength=1.3)


def export(fig, name, folder, qa, crop_vertical=False):
    """Check native-size text and export vectors; cropping never changes width."""
    text = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    for t in text:
        t.set_fontsize(max(TEXT, t.get_fontsize())); t.set_color('black')
        t.set_fontweight('bold' if getattr(t, '_panel_letter', False) else 'normal')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    boxes = [matplotlib.text.Text.get_window_extent(t, renderer) for t in text]
    outside = [t.get_text() for t,b in zip(text, boxes)
               if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1 > width+1 or b.y1 > height+1)]
    overlaps = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[i+1:], i+1):
            if min(a.x1,b.x1)-max(a.x0,b.x0)>.5 and min(a.y1,b.y1)-max(a.y0,b.y0)>.5:
                overlaps.append([text[i].get_text(), text[j].get_text()])
    record = dict(figure=name, width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
        width_mm=fig.get_figwidth()*25.4, height_mm=fig.get_figheight()*25.4,
        min_text_pt=min(t.get_fontsize() for t in text),
        font_sizes_pt=sorted({t.get_fontsize() for t in text}),
        out_of_canvas_text=outside, overlapping_text=overlaps,
        text_color='black', bold_text='panel letters only',
        subpanel_titles='centered, regular weight', crop_vertical_only=crop_vertical)
    qa.append(record)
    if outside or overlaps: raise ValueError((name, 'outside', outside, 'overlap', overlaps))
    bbox = None
    if crop_vertical:
        # Use the actual embedded-font bounds, avoiding backend-dependent legend
        # padding and descender metrics. Crop only vertically; never scale type.
        with tempfile.NamedTemporaryFile(suffix='.pdf') as scratch:
            fig.savefig(scratch.name, metadata={'CreationDate':None,'ModDate':None})
            with fitz.open(scratch.name) as document:
                spans = [span for block in document[0].get_text('dict')['blocks']
                         if 'lines' in block for line in block['lines']
                         for span in line['spans'] if 'Arial' in span['font']]
                top = min(span['bbox'][1] for span in spans)
                bottom = max(span['bbox'][3] for span in spans)
        native_height = fig.get_figheight()*72
        bbox = Bbox.from_extents(0, (native_height-bottom-2)/72,
                                WIDTH, (native_height-top+2)/72)
        record['export_height_inches'] = bbox.height
        record['visible_pdf_padding_pt'] = {'top':2, 'bottom':2}
    for ext in ['pdf', 'png', 'svg']:
        kwargs = dict(bbox_inches=bbox, pad_inches=0)
        if ext == 'pdf': kwargs['metadata'] = {'CreationDate':None, 'ModDate':None}
        if ext == 'svg': kwargs['metadata'] = {'Date':None}
        if ext == 'png': kwargs['dpi'] = 600
        fig.savefig(folder/(name+'.'+ext), **kwargs)
    plt.close(fig)
