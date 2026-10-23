#!/usr/bin/env python3
"""Redraw seven article figures exclusively from frozen, saved summary tables.

This script does not load raw cells, fit models, compute correlations, or rerun
the scientific pipeline. Only display operations (percent conversion, sums,
row counts, and the stated median of three saved condition summaries) occur.
Run from any directory with Python, matplotlib, numpy, and pandas installed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scientific-reports-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT
OUT = PROJECT / "figures"
EVIDENCE = PROJECT / "evidence"
FROZEN = Path("results/application_reanalysis_20260921")
MATCHED = Path("results/recovered_integration_20261001/native_analysis")
COHORTS = ["irradiation", "ucb", "dapp1", "sleep"]
TASKS = [("irradiation", "erythroid"), ("irradiation", "myeloid"),
         ("ucb", "megakaryocyte"), ("dapp1", "erythroid"),
         ("dapp1", "myeloid"), ("sleep", "myeloid")]
TASK_LABELS = ["IR Ery", "IR My", "TPO Mk", "Dapp1 Ery", "Dapp1 My", "Sleep My"]
COHORT_LABELS = {"irradiation": "Irradiation", "ucb": "TPO culture", "dapp1": "Dapp1", "sleep": "Sleep"}
ARMS = ["PCA", "LiVAE"]
ARM_COLORS = {"PCA": "#17689A", "LiVAE": "#8856A7"}
BLUE, ORANGE, GREEN = "#17689A", "#C6602A", "#168775"
INK, GRAY, LIGHT = "#000000", "#666666", "#CCCCCC"
PALE = "#F5F5F5"
WIDTH = 6.35
MIN_FONT = 10
LEDGER, SOURCES, QA = [], {}, []
CACHE = {}


def table(path):
    path = str(path)
    if path not in CACHE:
        packaged = PROJECT / "source_data" / path
        p = packaged if packaged.exists() else ROOT / path
        frame = pd.read_csv(p)
        frame.index = np.arange(1, len(frame) + 1)  # 1-based record; header excluded
        CACHE[path] = frame
        SOURCES[path] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "rows": len(frame), "columns": list(frame.columns)}
    return CACHE[path]


def rows(path, **selector):
    d = table(path)
    mask = pd.Series(True, index=d.index)
    for key, value in selector.items():
        mask &= d[key].eq(value)
    return d[mask]


def record(fig, panel, mark, path, selected, column, value, transform="identity", unit=""):
    LEDGER.append({"figure": fig, "panel": panel, "mark": mark,
                   "metric": column, "value": float(value), "unit": unit,
                   "source_file": str(path),
                   "source_rows_1based_excluding_header": ";".join(map(str, selected.index)),
                   "source_columns": column,
                   "source_selector": json.dumps(selected[[x for x in selected.columns if x in
                       ["cohort", "arm", "condition", "lineage", "population", "program",
                        "reward", "seed", "seed_a", "seed_b", "method"]]].to_dict("records"),
                       separators=(",", ":")),
                   "display_transform": transform})
    return float(value)


def one(fig, panel, mark, path, column, scale=1, unit="", **selector):
    selected = rows(path, **selector)
    assert len(selected) == 1, (path, selector, len(selected))
    value = float(selected[column].iloc[0]) * scale
    return record(fig, panel, mark, path, selected, column, value,
                  "identity" if scale == 1 else f"multiply by {scale}", unit)


def series(fig, panel, mark, path, column, scale=1, **selector):
    selected = rows(path, **selector)
    values = []
    for i, row in selected.iterrows():
        value = float(row[column]) * scale
        values.append(record(fig, panel, f"{mark}:row{i}", path, selected.loc[[i]], column,
                             value, "identity" if scale == 1 else f"multiply by {scale}"))
    return np.asarray(values)


def setup():
    for directory in [OUT, EVIDENCE]:
        directory.mkdir(parents=True, exist_ok=True)
    # Installed Arial is used explicitly; no silent publication-font substitution.
    arial = font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": MIN_FONT,
                         "axes.titlesize": 10.5, "axes.titleweight": "normal",
                         "axes.labelsize": MIN_FONT, "xtick.labelsize": MIN_FONT,
                         "ytick.labelsize": MIN_FONT, "legend.fontsize": MIN_FONT,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "lines.linewidth": 1.6,
                         "lines.markersize": 4.6, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "hematopoietic-readouts",
                         "savefig.facecolor": "white"})
    return arial


def title(ax, label, text):
    ax.set_title(text, loc="center", pad=13, fontweight="normal", color="black")
    # One physical offset aligns letters independently of the centered title.
    letter = ax.annotate(label.lower(), xy=(0, 1), xycoords="axes fraction",
                xytext=(-22, 13), textcoords="offset points", fontsize=13,
                fontweight="bold", color="black", ha="left", va="bottom",
                annotation_clip=False)
    letter._panel_letter = True


def clean(ax, axis="y"):
    ax.grid(axis=axis, color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def row_bands(ax, count):
    for y in range(count):
        if y % 2 == 0:
            ax.axhspan(y-.43, y+.43, facecolor=PALE, edgecolor="none", zorder=0)


def seed_range(ax, lower, upper, y, color):
    ax.plot([lower, upper], [y, y], color=color, lw=1.45, zorder=2)
    ax.vlines([lower, upper], y-.035, y+.035, color=color, lw=.8, zorder=2)


def save(fig, name):
    text_objects = [obj for obj in fig.findobj(matplotlib.text.Text)
                    if obj.get_visible() and obj.get_text()]
    for obj in text_objects:
        obj.set_fontsize(max(MIN_FONT, obj.get_fontsize()))
        obj.set_color("black")
        obj.set_fontweight("bold" if getattr(obj, "_panel_letter", False) else "normal")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check all visible text against the physical canvas (not a cropped export).
    w, h = fig.canvas.get_width_height()
    violations = []
    for obj in fig.findobj(matplotlib.text.Text):
        if obj.get_visible() and obj.get_text():
            box = obj.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                             box.x1 > w + 1 or box.y1 > h + 1):
                violations.append(obj.get_text())
    QA.append({"figure": name, "width_mm": fig.get_figwidth() * 25.4,
               "height_mm": fig.get_figheight() * 25.4,
               "width_inches": fig.get_figwidth(), "height_inches": fig.get_figheight(),
               "min_text_pt": min(obj.get_fontsize() for obj in text_objects),
               "font_sizes_pt": sorted(set(obj.get_fontsize() for obj in text_objects)),
               "out_of_canvas_text": violations,
               "text_color": "black", "bold_text": "panel letters only",
               "subpanel_titles": "centered, regular weight"})
    assert not violations, (name, violations)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def fig1():
    fig = plt.figure(figsize=(WIDTH, 5.2))
    ax = fig.add_axes([.075, .43, .90, .49]); ax.axis("off")
    title(ax, "a", "Analysis workflow")

    # Original line icons remain vector paths in PDF and editable SVG output.
    # They identify operations; they are not third-party method logos.
    def icon(kind, cx, cy):
        if kind == "observations":
            for dx, dy, radius in [(-.042, .012, .022), (.011, .037, .020), (.043, -.012, .024)]:
                ax.add_patch(Circle((cx+dx, cy+dy), radius, transform=ax.transAxes,
                                   facecolor="white", edgecolor="black", lw=.85))
            ax.plot([cx-.055, cx+.055], [cy-.055]*2, color="black", lw=.8)
            for dx, height in [(-.035,.026), (0,.045), (.035,.065)]:
                ax.add_patch(Rectangle((cx+dx-.009,cy-.055),.018,height,
                                      facecolor="#D9D9D9",edgecolor="black",lw=.55))
        elif kind == "frozen":
            for d in [-.045, -.015, .015, .045]:
                ax.plot([cx-.045,cx+.045],[cy+d]*2,color="#777777",lw=.65)
                ax.plot([cx+d]*2,[cy-.045,cy+.045],color="#777777",lw=.65)
            ax.plot([cx-.03,cx,cx+.03],[cy-.03,cy,cy+.03],color="black",lw=1.4,
                    marker="o",markersize=2.2)
        else:
            ax.plot([cx-.052,cx-.027,cx+.003],[cy-.035,cy+.03,cy-.005],
                    color="black",lw=.9,marker="o",markersize=2.2)
            ax.plot([cx+.018,cx+.043,cx+.065],[cy-.02,cy+.035,cy-.025],
                    color="black",lw=.9,marker="s",markersize=2.2)
            ax.annotate("",xy=(cx+.022,cy+.005),xytext=(cx-.005,cy+.005),
                        arrowprops={"arrowstyle":"->","color":"black","lw":.8})

    ax.add_patch(FancyBboxPatch((.13,.805),.74,.15,
                 boxstyle="round,pad=0.006,rounding_size=0.009",
                 facecolor="white",edgecolor="black",lw=.8))
    ax.text(.5,.906,"120,514 retained cells · four application cohorts",ha="center",va="center",fontsize=10.5)
    ax.text(.5,.846,"17 sampled times / conditions · 6 lineage tasks",ha="center",va="center")
    boxes=[(.005,"What is captured?","Composition / programs\nCommon-group contrasts\nRecorded sampling /\nculture days","observations"),
           (.35,"What do values add?","Frozen application\nPCA30 states\nFixed UMAP actions\nRewards / proximity","frozen"),
           (.695,"What varies by fit?","Paired interfaces / seeds\nPCA30 / LiVAE10\nEach to UMAP2\n132 downstream runs","paired")]
    for x, heading, body, kind in boxes:
        ax.add_patch(FancyBboxPatch((x,.08),.30,.595,
                     boxstyle="round,pad=0.005,rounding_size=0.009",
                     facecolor="white",edgecolor="#777777",lw=.7))
        ax.annotate("",xy=(x+.15,.68),xytext=(.5,.80),
                    arrowprops={"arrowstyle":"-|>","color":"black","linewidth":.8,
                                "shrinkA":2,"shrinkB":2})
        icon(kind,x+.15,.58)
        ax.text(x+.15,.47,heading,ha="center",va="center",fontsize=10.5)
        ax.text(x+.15,.385,body,ha="center",va="top",fontsize=MIN_FONT,linespacing=1.32)
    ax.text(.5,.018,"Separate technical reference: normal marrow · 6,011 cells",ha="center",va="center")
    ax.set_xlim(-.01,1.01);ax.set_ylim(-.015,1.0)

    a = fig.add_axes([.235, .10, .70, .24])
    title(a, "b", "Retained application cohorts")
    for item in a.texts:
        if getattr(item,"_panel_letter",False): item.set_position((-95,13))
    counts = []
    p = FROZEN / "geometry_reference/summary.csv"
    for c in COHORTS:
        lineage = next(l for cohort, l in TASKS if cohort == c)
        counts.append(one(1, "b", c, p, "n_cells", cohort=c, lineage=lineage, population="all"))
    a.barh([3,2,1,0],counts,height=.60,color=["#8856A7",BLUE,ORANGE,GREEN],zorder=3)
    a.set_yticks([3,2,1,0],["IR: 8 times","TPO: 5 times","Dapp1: WT / KO","Sleep: CT / SD"])
    a.set_xlim(0,47000);a.set_xticks([0,20000,40000],["0","20,000","40,000"])
    a.set_xlabel("Retained cells",labelpad=3)
    for y,count in zip([3,2,1,0],counts):
        a.text(count+650,y,f"{int(count):,}",va="center",fontsize=10)
    clean(a,"x")
    selected=rows(p,population="all")
    selected=selected.loc[[rows(p,cohort=c,lineage=next(l for cc,l in TASKS if cc==c),population="all").index[0] for c in COHORTS]]
    record(1,"a","total cells",p,selected,"n_cells",sum(counts),"sum across four distinct cohorts","cells")
    assert sum(counts)==120514
    retained_tasks=rows(p,population="all");retained_tasks=retained_tasks[retained_tasks.cohort.isin(COHORTS)]
    record(1,"a","lineage tasks",p,retained_tasks,"lineage",len(retained_tasks),"count unique cohort-lineage pairs","tasks")
    assert len(retained_tasks)==6
    condition_total=0
    for c in COHORTS:
        cp=FROZEN/c/"composition_counts.csv";d=table(cp)
        condition_total+=record(1,"a",f"{c} conditions",cp,d,"condition",len(d),"count condition rows","conditions")
    assert condition_total==17
    runs=0
    for filename,selector in [("associations_all_rewards.csv",{"population":"all"}),("focus_summary.csv",{})]:
        path=MATCHED/filename;d=rows(path,**selector);d=d[d.cohort.isin(COHORTS)]
        runs+=record(1,"a",filename,path,d,"seed",len(d),"count rows: one row per retained downstream run","runs")
    assert runs==132
    one(1,"a","marrow reference",p,"n_cells",cohort="marrow",lineage="erythroid",population="all")
    save(fig,"figure1_design")


def fig2():
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,5.2))
    fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.90,hspace=.62,wspace=.32)
    specs=[("dapp1",["wt","ko"],["WT","KO"],"cluster 6 (granulocytic)","Granulocytic cluster 6"),
           ("sleep",["CT","SD"],["CT","SD"],"Neu-like","Granulocytic-like groups")]
    for j,(c,cond,labels,column,desc) in enumerate(specs):
        ax=axes[0,j];p=FROZEN/c/"thesis_composition.csv"
        values=[one(2,chr(97+j),f"{c}:{x}",p,column,scale=100,**{"Unnamed: 0":x}) for x in cond]
        ax.bar([0,1],values,color=["#A8B5BE",[ORANGE,GREEN][j]],width=.54,zorder=3)
        ax.set_xticks([0,1],labels);ax.set_ylim(0,15 if j==0 else 55)
        if j == 0: ax.set_ylabel("Captured fraction (%)")
        for x,v in enumerate(values):ax.text(x,v+(.5 if j==0 else 1.3),f"{v:.2f}%",ha="center",fontsize=10,fontweight="bold")
        # Reuses the three cohort denominators already traced in Figure 1b.
        n=next(r['value'] for r in LEDGER if r['figure']==1 and r['panel']=='b' and r['mark']==c)
        numerator = "Dapp1 cluster 6" if c == "dapp1" else "Sleep granulocytic labels"
        title(ax,chr(97+j),f"{numerator}\nCohort n = {int(n):,}");clean(ax)

    programs=["neutrophil","cycling","interferon","inflammatory"]
    labels=["Granulocytic","Cell cycle","Interferon","Inflammatory"]
    for j,c in enumerate(["dapp1","sleep"]):
        ax=axes[1,j];p=FROZEN/c/"composition_standardized_effects.csv"
        for y,program in enumerate(programs):
            vals=[one(2,chr(99+j),f"{program}:{column}",p,column,program="program_"+program)
                  for column in ["unadjusted_delta","composition_standardized_delta"]]
            ax.plot(vals,[y,y],color=LIGHT,lw=2.2,zorder=1)
            ax.scatter(vals[0],y,facecolor="white",edgecolor=GRAY,linewidth=1.3,s=48,zorder=3)
            ax.scatter(vals[1],y,facecolor=BLUE,edgecolor="white",linewidth=.4,s=32,zorder=4)
            if y==0 or (j==0 and y==1):
                for v,dy,col in zip(vals,[-13,10],[GRAY,BLUE]):
                    ax.annotate(f"{v:+.3f}",(v,y),xytext=(0,dy),textcoords="offset points",ha="center",fontsize=MIN_FONT,color=col)
        ax.axvline(0,color="#9299A1",lw=.7,ls="--");ax.set_xlim(-.49,.23)
        ax.set_ylim(3.5,-1.1);ax.set_yticks(range(4),labels if j == 0 else [""]*4);ax.set_xticks([-.4,-.2,0,.2])
        row_bands(ax,4);ax.set_xlabel("Program difference")
        title(ax,chr(99+j),"KO − WT" if c=="dapp1" else "SD − CT");clean(ax,"x")
        ng=one(2,chr(99+j),"common groups",p,"matched_groups",program="program_neutrophil")
        cv=one(2,chr(99+j),"coverage",p,"fraction_cells_covered",program="program_neutrophil",scale=100)
        ax.set_title(ax.get_title()+f" ({int(ng)} groups)",loc="center",pad=13)
    fig.legend([Line2D([],[],marker="o",markerfacecolor="white",color=GRAY,ls=""),Line2D([],[],marker="o",color=BLUE,ls="")],
               ["Unadjusted","Fixed composition"],loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.55,.018))


    save(fig,"figure2_composition")


def fig3():
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,5.95));fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.92,wspace=.48,hspace=.88)
    days=[0,4,7,11,14];conditions=[f"D{x}" for x in days]
    p=FROZEN/"ucb/thesis_composition.csv";ax=axes[0,0]
    fractions=[]
    for col,label,color,marker in [("Prim-like","Primitive label",GREEN,"o"),("Mk-like","Mk label",BLUE,"s"),("Ery-like","Ery label",ORANGE,"^")]:
        values=[one(3,"a",f"{col}:{c}",p,col,scale=100,**{"Unnamed: 0":c}) for c in conditions]
        fractions.append(values)
        ax.plot(days,values,marker=marker,color=color,label=label.replace("Start-label subset", "Cluster 5"))
    # Account for every captured cell without changing the 378 original ledger rows.
    # The additional five arithmetic display values have their own source ledger.
    other=100-np.sum(np.asarray(fractions),axis=0)
    assert np.all(other>=0) and np.all(other<=100)
    ax.plot(days,other,color=GRAY,marker="D",ls="--",label="Other frozen labels")
    context=[]
    for i,condition in enumerate(conditions):
        selected=rows(p,**{"Unnamed: 0":condition})
        assert len(selected)==1
        context.append({"condition":condition,"primitive_fraction_pct":fractions[0][i],
            "megakaryocyte_fraction_pct":fractions[1][i],"erythroid_fraction_pct":fractions[2][i],
            "original_three_sum_pct":sum(fractions[j][i] for j in range(3)),
            "other_frozen_labels_pct":float(other[i]),"unit":"percent of captured cells",
            "source_file":str(p),"source_rows_1based_excluding_header":str(selected.index[0]),
            "source_columns":"Prim-like;Mk-like;Ery-like",
            "display_formula":"100 - 100 * (Prim-like + Mk-like + Ery-like)"})
    pd.DataFrame(context).to_csv(EVIDENCE/"figure3_composition_context.csv",index=False)
    context_metadata={"scope":"Five complementary display values for Figure 3a, separate from the unchanged original 378-row figure ledger.",
        "formula":"other_frozen_labels_pct = 100 - primitive_fraction_pct - megakaryocyte_fraction_pct - erythroid_fraction_pct",
        "source_formula":"100 - 100 * (Prim-like + Mk-like + Ery-like)",
        "source_file":str(p),"source_sha256":SOURCES[str(p)]["sha256"],
        "source_columns":["Prim-like","Mk-like","Ery-like"],
        "source_rows_1based_excluding_header":[int(rows(p,**{"Unnamed: 0":c}).index[0]) for c in conditions],
        "original_ledger":"figure_values.csv, figure 3 panel a, marks Prim-like:D0–D14; Mk-like:D0–D14; Ery-like:D0–D14",
        "derived_rows":len(context),"original_ledger_rows_unchanged":378,
        "interpretation":"Other frozen labels is the exact complement of the three plotted provisional labels; it is not a single validated cell type.",
        "scientific_policy":"Display arithmetic only; no new fits, relabelling, cell-level scoring or inferential analysis.",
        "context_csv_sha256":hashlib.sha256((EVIDENCE/"figure3_composition_context.csv").read_bytes()).hexdigest()}
    (EVIDENCE/"figure3_composition_context.json").write_text(json.dumps(context_metadata,indent=2)+"\n")
    ax.annotate(f"D0 other: {other[0]:.2f}%",xy=(0,other[0]),xytext=(3.7,91.5),fontsize=MIN_FONT,color=GRAY,
                arrowprops={"arrowstyle":"-","color":GRAY,"lw":.75},va="center")
    ax.set_ylabel("Captured fraction (%)");ax.set_ylim(-2,103);ax.set_yticks([0,25,50,75,100])
    title(ax,"a","Provisional label fractions")
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="center",ncol=4,frameon=False,bbox_to_anchor=(.55,.555),
               handlelength=1.6,columnspacing=.85,handletextpad=.45,fontsize=MIN_FONT)
    ax=axes[0,1]
    for file,label,color,marker in [("condition_program_means.csv","All cells",BLUE,"o"),("starting_population_programs.csv","Start-label subset",ORANGE,"s")]:
        p=FROZEN/"ucb"/file
        values=[one(3,"b",f"{label}:{c}",p,"program_megakaryocyte",condition=c) for c in conditions]
        ax.plot(days,values,marker=marker,color=color,label=label.replace("Start-label subset", "Cluster 5"))
    ax.axhline(0,color=LIGHT,lw=.8);ax.set_ylabel("Mean program score");ax.set_ylim(-1.2,.9)
    ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1),
              borderaxespad=.35,handlelength=1.25,labelspacing=.25)
    title(ax,"b","Megakaryocyte program")
    ax=axes[1,0];p=FROZEN/"ucb/condition_gene_means.csv"
    for gene,color,marker in [("PF4",BLUE,"o"),("PPBP",ORANGE,"s"),("ITGA2B",GREEN,"^")]:
        values=[one(3,"c",f"{gene}:{c}",p,gene,condition=c) for c in conditions]
        ax.plot(days,values,marker=marker,color=color,label=gene)
    ax.set_ylabel("Mean log1p(CP10k)");ax.set_ylim(-.1,3.65);ax.legend(frameon=False,loc="upper left")
    title(ax,"c","All-cell component genes")
    ax=axes[1,1];p=FROZEN/"ucb/state_values_by_condition.csv"
    for pop,label,color,marker in [("all","All cells",BLUE,"o"),("starting_population","Start-label subset",ORANGE,"s")]:
        for seed in range(3):
            values=[one(3,"d",f"{pop}:seed{seed}:{c}",p,"median_value_percentile",seed=seed,reward="neutral",population=pop,condition=c) for c in conditions]
            ax.plot(days,values,color=color,alpha=.22,lw=.8)
        med=[]
        for c in conditions:
            d=rows(p,reward="neutral",population=pop,condition=c)
            med.append(record(3,"d",f"{pop}:seed median:{c}",p,d,"median_value_percentile",d.median_value_percentile.median(),"median of three saved condition-level medians"))
        ax.plot(days,med,color=color,marker=marker,label=label.replace("Start-label subset", "Cluster 5"))
    denominators=[]
    for condition in conditions:
        denominators.append(one(3,"d",f"{condition} starting denominator",p,"n_cells",seed=0,reward="neutral",population="starting_population",condition=condition))
    ax.set_ylabel("Median value percentile");ax.set_ylim(0,1);ax.legend(frameon=False,loc="upper left",fontsize=MIN_FONT)

    ax.annotate("D0: 17 cells",xy=(0,.14),xytext=(3.3,.30),fontsize=MIN_FONT,color=ORANGE,
                arrowprops={"arrowstyle":"-","color":ORANGE,"lw":.75},va="center")
    title(ax,"d","No-time scRL value")
    for ax in axes.flat:
        ax.set_xticks(days);ax.set_xlim(-.5,14.5);ax.set_xlabel("Culture day");clean(ax)

    fig.text(.55,.12,"Panels b,d: each day's intersection with frozen cluster 5",ha="center",fontsize=MIN_FONT)
    for x,day,n in zip(np.linspace(.16,.94,5),days,denominators):
        fig.text(x,.067,f"D{day}: {int(n):,}",ha="center",fontsize=MIN_FONT)
    save(fig,"figure3_tpo")


def correlation_panel(ax, fig_no, panel, population, matched=False, show_y=True, details=False):
    p=MATCHED/"neutral_summary.csv" if matched else FROZEN/"geometry_reference/summary.csv"
    row_bands(ax,len(TASKS))
    annotations=[]
    for y,(cohort,lineage) in enumerate(TASKS):
        selector={"cohort":cohort,"lineage":lineage,"population":population}
        medians={}
        if matched:
            for arm,offset,marker in [("PCA",-.18,"s"),("LiVAE",.18,"o")]:
                q=dict(selector,arm=arm);color=ARM_COLORS[arm]
                v={k:one(fig_no,panel,f"{cohort}:{lineage}:{arm}:{k}",p,k,**q)
                   for k in ["value_min","value_median","value_max","geometry_rho"]}
                medians[arm]=v["value_median"]
                seed_range(ax,v["value_min"],v["value_max"],y+offset-.055,color)
                ax.scatter(v["value_median"],y+offset-.055,color=color,marker=marker,s=30,edgecolor="white",lw=.4,zorder=4)
                ax.scatter(v["geometry_rho"],y+offset+.065,facecolor="white",edgecolor=color,marker="D",s=24,lw=1.05,zorder=5)
        else:
            v={k:one(fig_no,panel,f"{cohort}:{lineage}:{k}",p,k,**selector) for k in ["scrl_min","scrl_median","scrl_max","umap","pca"]}
            seed_range(ax,v["scrl_min"],v["scrl_max"],y,BLUE)
            ax.scatter(v["scrl_median"],y,color=BLUE,s=31,edgecolor="white",lw=.4,zorder=4)
            ax.scatter(v["umap"],y-.16,color=ORANGE,marker="s",s=25,edgecolor="white",lw=.4,zorder=4)
            ax.scatter(v["pca"],y+.16,color=GREEN,marker="^",s=31,edgecolor="white",lw=.4,zorder=4)
        n=one(fig_no,panel,f"{cohort}:{lineage}:denominator",p,"n_cells",**dict(selector,**({"arm":"PCA"} if matched else {})))
        annotations.append((n,medians))
    ax.set_ylim(len(TASKS)-.45,-.55)
    labels=[f"{label} · {int(n):,}" for label,(n,_) in zip(TASK_LABELS,annotations)] if details else TASK_LABELS
    ax.set_yticks(range(len(TASKS)),labels if show_y else [""]*len(TASKS))
    if details:
        ax.text(1.315,1.14,"3-seed scRL median ρ",transform=ax.transAxes,
                ha="center",fontsize=MIN_FONT)
        for x,arm in [(1.18,"PCA"),(1.45,"LiVAE")]:
            ax.text(x,1.015,arm,transform=ax.transAxes,ha="center",fontsize=MIN_FONT)
            for y,(_,medians) in enumerate(annotations):
                ax.text(x,y,f"{medians[arm]:+.3f}",transform=ax.get_yaxis_transform(),ha="center",va="center",fontsize=MIN_FONT,color=ARM_COLORS[arm])
    ax.set_xlim(-.30,1.0);ax.set_xticks([0,.5,1]);ax.axvline(0,color="#9DA5AC",ls="--",lw=.7)
    ax.set_xlabel("Spearman ρ");clean(ax,"x")


def fig4():
    fig=plt.figure(figsize=(WIDTH,6.3));gs=fig.add_gridspec(2,6,left=.15,right=.98,bottom=.13,top=.85,hspace=.58,wspace=.64,height_ratios=[.85,1.4])
    p=FROZEN/"marrow/reward_timing.csv"
    for j,lineage in enumerate(["erythroid","myeloid"]):
        ax=fig.add_subplot(gs[0,j*3:(j+1)*3])
        for i,(reward,label,color) in enumerate([("decision","Decision",BLUE),("contribution","Contribution",ORANGE),("neutral","No-time",GREEN)]):
            vals=series(4,chr(97+j),reward,p,"top_decile_mean_pseudotime",lineage=lineage,reward=reward)
            for offset,value,marker in zip([-.09,0,.09],vals,["o","s","^"]):
                ax.scatter(i+offset,value,color=color,marker=marker,s=31,edgecolor="white",lw=.4,zorder=3)
        ax.set_xticks(range(3),["Decision","Contribution","No-time"]);ax.set_ylim(0,.85);ax.set_xlim(-.35,2.35)
        if j==0: ax.set_ylabel("High-value mean pseudotime")
        clean(ax)
        title(ax,chr(97+j),lineage.capitalize())
    for i,pop in enumerate(["all","nonterminal","starting"]):
        ax=fig.add_subplot(gs[1,i*2:(i+1)*2])
        correlation_panel(ax,4,chr(99+i),pop,show_y=i==0)
        title(ax,chr(99+i),["All cells","Nonterminal","Selected start"][i])
    handles=[Line2D([],[],color=BLUE,marker="o",label="scRL median / seed range"),
             Line2D([],[],ls="",marker="s",color=ORANGE,label="UMAP proximity"),
             Line2D([],[],ls="",marker="^",color=GREEN,label="PCA30 proximity")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.55,.014),columnspacing=1.0,handlelength=1.4,fontsize=MIN_FONT)

    fig.text(.565,.987,"Normal-marrow reward control",ha="center",va="top",fontsize=MIN_FONT)
    fig.legend(handles=[Line2D([],[],color=GRAY,marker=m,ls="",label=f"Seed {i}") for i,m in enumerate(["o","s","^"])],
               loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.60,.955),fontsize=MIN_FONT,columnspacing=1.4,handletextpad=.4)
    fig.text(.565,.558,"Application cohorts: no-time reward",ha="center",va="center",fontsize=MIN_FONT)



    save(fig,"figure4_controls")


def fig5():
    fig=plt.figure(figsize=(WIDTH,8.1))
    for i,pop in enumerate(["all","nonterminal","starting"]):
        ax=fig.add_axes([.245,.73-i*.29,.46,.215])
        correlation_panel(ax,5,chr(97+i),pop,matched=True,show_y=True,details=True)
        title(ax,chr(97+i),["All cells","Nonterminal","Selected start"][i])
        ax.set_xticks([-.25,0,.25,.5,.75,1],["−0.25","0","0.25","0.50","0.75","1.00"])
        ax.set_xlabel("Spearman ρ" if i==2 else "")
        if i < 2: ax.tick_params(labelbottom=False)
    handles=[Line2D([],[],color=BLUE,marker="s",label="PCA"),Line2D([],[],color=ARM_COLORS["LiVAE"],marker="o",label="Revised LiVAE"),
             Line2D([],[],color=GRAY,marker="D",markerfacecolor="white",ls="",label="Same-interface proximity")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.55,.015),columnspacing=1.1,handlelength=1.4)
    fig.text(.55,.069,"Filled marks: scRL medians; segments: seed ranges",ha="center",fontsize=MIN_FONT)


    save(fig,"figure5_interfaces")


def fig6():
    fig=plt.figure(figsize=(WIDTH,6.7));gs=fig.add_gridspec(2,2,left=.16,right=.95,bottom=.19,top=.90,hspace=.62,wspace=.28,height_ratios=[1,1.4])
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    p=MATCHED/"seed_stability.csv"
    markers=["o","s","^"]
    row_bands(a,len(COHORTS));row_bands(b,len(COHORTS));row_bands(c,len(TASKS))
    for y,cohort in enumerate(COHORTS):
        for arm,off in [("PCA",-.23),("LiVAE",.23)]:
            vals=series(6,"a",f"{cohort}:{arm}",p,"value",cohort=cohort,arm=arm,method="scFocus")
            a.plot([min(vals),max(vals)],[y+off]*2,color=ARM_COLORS[arm],alpha=.45,lw=.8)
            for k,v in enumerate(vals): a.scatter(v,y+off+(k-1)*.12,color=ARM_COLORS[arm],marker=markers[k],s=30,edgecolor="white",lw=.4,zorder=4)
            q=MATCHED/"focus_summary.csv"
            values=series(6,"b",f"{cohort}:{arm}",q,"median_entropy",cohort=cohort,arm=arm)
            for k,v in enumerate(values): b.scatter(v,y+off+(k-1)*.12,color=ARM_COLORS[arm],marker=markers[k],s=30,edgecolor="white",lw=.4,zorder=4)
    for ax in [a,b]:
        ax.set_yticks(range(len(COHORTS)),[COHORT_LABELS[x] for x in COHORTS]);ax.set_ylim(len(COHORTS)-.5,-.5);clean(ax,"x")
    a.set_xlim(0,1);a.set_xticks([0,.5,1]);a.set_xlabel("Assignment ARI");title(a,"a","scFocus seed agreement")
    b.set_xlim(.95,1);b.set_xticks([.95,.975,1],["0.950","0.975","1.000"]);b.set_xlabel("Median normalized entropy");title(b,"b","scFocus weight dispersion\nEntropy zoom: 0.95–1.00")
    b.set_yticklabels([])
    b.legend(handles=[Line2D([],[],color=GRAY,marker=m,ls="",label=f"Seed {i}")
                      for i,m in enumerate(markers)],loc="upper center",ncol=3,
             frameon=False,bbox_to_anchor=(.5,-.37),borderaxespad=0,
             handletextpad=.35,columnspacing=.6,handlelength=.8,fontsize=MIN_FONT)
    for y,(cohort,lineage) in enumerate(TASKS):
        for arm,off in [("PCA",-.23),("LiVAE",.23)]:
            values=series(6,"c",f"{cohort}:{lineage}:{arm}",p,"value",cohort=cohort,lineage=lineage,arm=arm,method="scRL",reward="neutral")
            c.plot([min(values),max(values)],[y+off]*2,color=ARM_COLORS[arm],alpha=.45,lw=.8)
            if arm=="LiVAE":
                c.text(.025,y,f"{min(values):.2f}–{max(values):.2f}",va="center",fontsize=MIN_FONT,color=ARM_COLORS[arm])
            for k,v in enumerate(values): c.scatter(v,y+off+(k-1)*.12,color=ARM_COLORS[arm],marker=markers[k],s=30,edgecolor="white",lw=.4,zorder=4)
    c.set_yticks(range(len(TASKS)),TASK_LABELS);c.set_ylim(len(TASKS)-.5,-.5);c.set_xlim(0,1)
    c.axvline(0,color="#9DA5AC",ls="--",lw=.7);c.set_xlabel("Between-seed value Spearman ρ");clean(c,"x")
    title(c,"c","No-time scRL seed agreement")
    c.text(.025,1.015,"LiVAE range",transform=c.transAxes,fontsize=MIN_FONT,color=ARM_COLORS["LiVAE"])
    fig.legend(handles=[Line2D([],[],color=ARM_COLORS[x],marker="o",ls="",label="Revised LiVAE" if x=="LiVAE" else x) for x in ARMS],
               loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.55,.077),fontsize=MIN_FONT)
    fig.legend(handles=[Line2D([],[],color=GRAY,marker=m,ls="",label=f"Pair {pair}")
                        for m,pair in zip(markers,["0–1","0–2","1–2"])],
               title="Seed pairs in a,c",title_fontsize=MIN_FONT,
               loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.55,.004),fontsize=MIN_FONT,columnspacing=1.1,handletextpad=.45)


    save(fig,"figure6_stability")


def fig7():
    """Irradiation displays saved aggregates, with denominators kept explicit."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 7.25))
    fig.subplots_adjust(left=.14, right=.98, bottom=.32, top=.92,
                        hspace=.72, wspace=.45)
    days = [0, 2, 5, 8, 11, 14, 21, 30]
    conditions = [f"d{day}" for day in days]
    composition = FROZEN / "irradiation/thesis_composition.csv"
    counts_path = FROZEN / "irradiation/composition_counts.csv"
    captured, hspc, erythroid = [], [], []
    for condition in conditions:
        d = rows(counts_path, condition=condition)
        assert len(d) == 1
        cell_columns = [x for x in d.columns if x != "condition"]
        captured.append(record(7, "ab", f"{condition}:captured n", counts_path, d,
                               ";".join(cell_columns), d[cell_columns].sum(axis=1).iloc[0],
                               "sum saved label counts within condition", "cells"))
        hspc.append(record(7, "ac", f"{condition}:HSPC n", counts_path, d,
                           "HSC;MPP2/3;MPP4", d[["HSC", "MPP2/3", "MPP4"]].sum(axis=1).iloc[0],
                           "sum saved HSC, MPP2/3 and MPP4 counts", "cells"))
        erythroid.append(one(7, "b", f"{condition}:Ery n", counts_path, "Ery",
                             condition=condition, unit="cells"))
    assert captured == [10386, 1297, 1344, 7984, 4129, 4235, 6682, 5195]
    assert hspc[2] == 66
    for ax, letter, column, heading, color, marker in [
            (axes[0, 0], "a", "HSPC", "Captured HSPC labels", BLUE, "o"),
            (axes[0, 1], "b", "Ery", "Captured erythroid label", ORANGE, "s")]:
        fractions = [one(7, letter, f"{column}:{condition}", composition, column,
                         scale=100, unit="percent", **{"Unnamed: 0": condition})
                     for condition in conditions]
        ax.plot(days, fractions, color=color, marker=marker)
        ax.set_ylabel("Captured fraction (%)")
        ax.set_ylim(0, 90 if column == "HSPC" else 75)
        title(ax, letter, heading)
    ax = axes[1, 0]
    programs = FROZEN / "irradiation/starting_population_programs.csv"
    for program, label, color, marker in [("program_neutrophil", "Granulocytic", BLUE, "o"),
                                          ("program_interferon", "Interferon", ORANGE, "s")]:
        values = [one(7, "c", f"{program}:{condition}", programs, program,
                      condition=condition) for condition in conditions]
        ax.plot(days, values, color=color, marker=marker, label=label)
        if program == "program_neutrophil":
            ax.annotate("Day 5: 66 HSPC", xy=(5, values[2]), xytext=(12, .47),
                        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .75},
                        fontsize=MIN_FONT, ha="left")
    ax.axhline(0, color=LIGHT, lw=.8)
    ax.set_ylabel("Mean program score")
    ax.legend(frameon=False, loc="upper right", handlelength=1.15,
              handletextpad=.4, borderaxespad=.2)
    title(ax, "c", "Programs within HSPC labels")
    ax = axes[1, 1]
    values_path = FROZEN / "irradiation/state_values_by_condition.csv"
    for lineage, label, color, marker in [("erythroid", "Erythroid", BLUE, "o"),
                                          ("myeloid", "Myeloid", ORANGE, "s")]:
        values = {name: [] for name in ["min", "median", "max"]}
        for condition in conditions:
            d = rows(values_path, condition=condition, lineage=lineage,
                     reward="neutral", population="starting_population")
            assert len(d) == 3 and sorted(d.seed.tolist()) == [0, 1, 2]
            for name in values:
                result = getattr(d.median_value_percentile, name)()
                values[name].append(record(7, "d", f"{lineage}:{condition}:{name}", values_path,
                                           d, "median_value_percentile", result,
                                           f"{name} of three saved condition-level medians"))
        ax.fill_between(days, values["min"], values["max"], color=color, alpha=.15, linewidth=0)
        ax.plot(days, values["median"], color=color, marker=marker, label=label)
    ax.set_ylabel("Median value percentile")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", handlelength=1.15,
              handletextpad=.4, borderaxespad=.2)
    title(ax, "d", "No-time HSPC scRL value")
    for ax in axes.flat:
        ax.set_xticks(days)
        ax.set_xlim(-1, 31)
        ax.set_xlabel("Day after irradiation")
        clean(ax)
    labels = fig.add_axes([.04, .025, .94, .19])
    labels.axis("off")
    count_rows = [["Day"] + [str(x) for x in days],
                  ["Captured n"] + [f"{int(x):,}" for x in captured],
                  ["HSPC n"] + [f"{int(x):,}" for x in hspc],
                  ["Ery n"] + [f"{int(x):,}" for x in erythroid]]
    count_table = labels.table(cellText=count_rows, cellLoc="center",
                              colWidths=[.18]+[.1025]*8, bbox=[0, 0, 1, 1])
    count_table.auto_set_font_size(False)
    count_table.set_fontsize(MIN_FONT)
    for (row, col), cell in count_table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor(PALE if row % 2 else "white")
        if col == 0:
            cell.get_text().set_ha("left")
    save(fig, "figure7_irradiation")


# Figure captions are maintained with the manuscript and evidence, not duplicated here.


def main():
    arial=setup()
    for function in [fig1,fig2,fig3,fig4,fig5,fig6,fig7]:
        function()
    pd.DataFrame(LEDGER).to_csv(EVIDENCE/"figure_values.csv",index=False)
    manifest={"scope":{"cohorts":COHORTS,"tasks":TASKS,"cells":120514,"conditions":17,"matched_runs":132,
                       "matched_scRL_runs":108,"matched_scFocus_runs":24,"upstream_fits":4},
              "row_convention":"CSV data records numbered from 1; header excluded; source line is record + 1",
              "sources":SOURCES,"figure_checks":QA,"font":arial,
              "analysis_policy":"Saved summary tables only. No experimental reruns, new cell-level calculations, statistical tests, or new correlations.",
              "export_policy":{"vector":"PDF with embedded TrueType fonts; editable SVG with live Arial text","raster":"PNG at 600 dpi","width_mm":WIDTH*25.4,"minimum_native_text_pt":MIN_FONT,"intended_print_width_inches":6.35},
              "display_aggregations":["Figure 1 sum of distinct cohort counts and row counts of retained conditions/runs.",
                  "Composition fractions multiplied by 100. Figure 3a Other frozen labels is the exact 100% complement of the original three labels, traced in a separate five-row display ledger.",
                  "Figure 3d median of three already saved condition-level value-percentile medians.",
                  "Irradiation: sum saved label counts for denominators; min, median and max of three saved condition-level value-percentile medians.",
                  "Figure 6 segments span existing three pairwise values; endpoints use min/max for display."],
              
              
              "derived_display_context":{"figure3_composition_context.csv":{"rows":5,"sha256":hashlib.sha256((EVIDENCE/"figure3_composition_context.csv").read_bytes()).hexdigest()},
                  "figure3_composition_context.json":{"sha256":hashlib.sha256((EVIDENCE/"figure3_composition_context.json").read_bytes()).hexdigest()},
                  "note":"The separate five-point TPO complement line is not duplicated in figure_values.csv."},
              "outputs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("figure[1-7]*"))}}
    (EVIDENCE/"figure_sources.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (EVIDENCE/"figure_qa.json").write_text(json.dumps({"figures":QA,"status":"pass"},indent=2)+"\n")
    # Captions are maintained with the manuscript; rendering does not overwrite them.
    print(json.dumps({"figures":7,"source_tables":len(SOURCES),"ledger_entries":len(LEDGER),"checks":QA},indent=2))


if __name__=="__main__":
    main()
