"""Recreate the six main-figure panels from the packaged v6 source tables.

This is a source-data reproduction, not a model refit. Outputs are written to
an explicit directory and do not overwrite the manuscript-matched figure PNGs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"Pf": "#b84b43", "Pb": "#4e7b55", "Pk": "#4c78a8", "Tg": "#8b6aa8"}
BLUE = "#315b7d"
LIGHT_BLUE = "#8db7d5"
ORANGE = "#d88432"
GRAY = "#82919b"
SOURCE_INPUTS = [
    "Fig2/F2a_performance.tsv", "Fig2/F2b_branch_benchmarks.tsv", "Fig2/F2c_calibration.tsv",
    "Fig2d_sensitivity/F2d_five_allocation_summary.tsv",
    "Fig3/F3a_LOSO.tsv", "Fig3/F3bc_full_stack_bootstrap.tsv", "Fig3/F3bc_shared_bootstrap.tsv",
    "Fig3/F3d_full_stack_metrics.tsv", "Fig3/F3d_shared_metrics.tsv",
    "Fig4_final_frozen_20261024/Fig4a_species_calls.tsv", "Fig4_final_frozen_20261024/Fig4b_triplet_funnel.tsv",
    "Fig4_final_frozen_20261024/Fig4c_directional_pattern_counts.tsv", "Fig4_final_frozen_20261024/Fig4d_recovery_histogram.tsv",
    "Fig4_final_frozen_20261024/Fig4e_orthomcl_support_summary.tsv",
    "Figure5_context_recomputed_20261023_final/Fig5_sequence_identity_matched_groups.tsv",
    "Figure5_context_recomputed_20261023_final/Fig5_sequence_identity_statistics.tsv",
    "Figure5_context_recomputed_20261023_final/Fig5_GO_enrichment_FDR05.tsv",
    "Figure5_context_recomputed_20261023_final/Fig5_stable_group_evidence_matrix.tsv",
    "Fig6/F6a_loso_performance.tsv", "Fig6/F6bc_target_enrichment.tsv", "Fig6/F6d_target_recovery.tsv",
]


def read(root: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(root / rel, sep="\t")


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="left")


def save(fig: plt.Figure, out: Path, name: str) -> dict:
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    p = out / f"{name}.png"
    return {"file": p.name, "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def fig1(root: Path, out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_axis_off()
    boxes = [
        (.02, .57, .21, .25, "95% redundancy\nreduction +\nstratified split", "#e9f2fb"),
        (.275, .57, .21, .25, "Raw 628 features\n9 routes\nfold-local dynamics", "#e6f4ea"),
        (.53, .57, .21, .25, "5-fold OOF\nLightGBM meta-model\ntraining-only threshold", "#fff2cc"),
        (.785, .57, .19, .25, "Frozen scores\nmain tests + atlas", "#fce4d6"),
        (.20, .14, .25, .22, "Four-species LOSO\nPf / Pb / Pk / Tg", "#e9f2fb"),
        (.56, .14, .25, .22, "Pk dual screen\nhuman-RBC condition", "#f4cccc"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                                   facecolor=color, edgecolor="#555", lw=1.1))
        ax.text(x+w/2, y+h/2, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=10, fontweight="bold")
    for a, b in [((.23,.695),(.275,.695)), ((.485,.695),(.53,.695)),
                 ((.74,.695),(.785,.695)), ((.38,.57),(.325,.36)), ((.64,.57),(.685,.36))]:
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops={"arrowstyle":"->", "lw":1.2, "color":"#444"})
    ax.text(.5, .94, "ProtoEss release architecture", transform=ax.transAxes,
            ha="center", fontsize=15, fontweight="bold")
    ax.text(.5, .025,
            "Current-run test labels are used only for final scoring; historical model development was not fully blind.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#8b0000")
    return save(fig, out, "Fig1")


def fig2(root: Path, out: Path) -> dict:
    a = read(root, "Fig2/F2a_performance.tsv")
    b = read(root, "Fig2/F2b_branch_benchmarks.tsv")
    c = read(root, "Fig2/F2c_calibration.tsv")
    d = read(root, "Fig2d_sensitivity/F2d_five_allocation_summary.tsv")
    short = ["Pf", "Pb", "Pk"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.3))
    ax=axs[0,0]; x=np.arange(3); width=.32
    ax.bar(x-width/2,a.AUROC,width,color="#333333",label="AUROC")
    ax.bar(x+width/2,a.AUPRC,width,color="#2e83b4",label="AUPRC")
    ax.errorbar(x-width/2,a.AUROC,
        yerr=[a.AUROC-a.AUROC_ci95_low,a.AUROC_ci95_high-a.AUROC],fmt="none",ecolor="#222",capsize=3)
    ax.errorbar(x+width/2,a.AUPRC,
        yerr=[a.AUPRC-a.AUPRC_ci95_low,a.AUPRC_ci95_high-a.AUPRC],fmt="none",ecolor="#2e83b4",capsize=3)
    ax.set_xticks(x,short); ax.set_ylim(.65,1); ax.set_ylabel("Held-out performance"); ax.legend(frameon=False)
    panel(ax,"a","Species-matched performance")

    ax=axs[0,1]; models=list(b.model.drop_duplicates()); colors=plt.cm.tab10(np.linspace(0,.8,len(models)))
    for j,model in enumerate(models):
        q=b[b.model.eq(model)].set_index("short_species").reindex(short)
        ax.scatter(np.arange(3)+(j-(len(models)-1)/2)*.07,q.AUROC,s=32,label=model,color=colors[j])
    ax.set_xticks(np.arange(3),short); ax.set_ylim(.5,1); ax.set_ylabel("AUROC"); ax.legend(frameon=False,fontsize=7)
    panel(ax,"b","Direct branch benchmarks")

    ax=axs[1,0]
    for s in short:
        q=c[(c.species.map({"Plasmodium_falciparum":"Pf","Plasmodium_berghei":"Pb","Plasmodium_knowlesi":"Pk"})==s)&c.kind.eq("bin")].sort_values("bin")
        ax.plot(q.mean_predicted_probability,q.observed_essential_fraction,"o-",lw=1.4,ms=3,label=s,color=COLORS[s])
    ax.plot([0,1],[0,1],"--",color="#777",lw=1); ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted probability",ylabel="Observed essential fraction")
    ax.legend(frameon=False); panel(ax,"c","Held-out calibration")

    ax=axs[1,1]; x=np.arange(3); width=.32
    ax.bar(x-width/2,d.nine_route_AUROC_mean,width,yerr=d.nine_route_AUROC_std,capsize=3,color="#3f667f",label="Nine-route model")
    ax.bar(x+width/2,d.signed_nn_AUROC_mean,width,yerr=d.signed_nn_AUROC_std,capsize=3,color="#d46a70",label="Signed NN baseline")
    ax.set_xticks(x,short); ax.set_ylim(.5,1); ax.set_ylabel("AUROC"); ax.legend(frameon=False,fontsize=8)
    ax.text(.5,-.18,"Mean ± sample SD across five re-allocations; sensitivity analysis, not test CI.",transform=ax.transAxes,ha="center",fontsize=7)
    panel(ax,"d","30% homology-cluster reallocation")
    fig.tight_layout()
    return save(fig,out,"Fig2")


def fig3(root: Path, out: Path) -> dict:
    loso=read(root,"Fig3/F3a_LOSO.tsv")
    full=read(root,"Fig3/F3d_full_stack_metrics.tsv")
    shared=read(root,"Fig3/F3d_shared_metrics.tsv")
    full_ci=read(root,"Fig3/F3bc_full_stack_bootstrap.tsv")
    shared_ci=read(root,"Fig3/F3bc_shared_bootstrap.tsv")
    source_map={"Plasmodium_falciparum":"Pf","Plasmodium_berghei":"Pb","Plasmodium_knowlesi":"Pk"}
    loso["short"]=loso.species.map({**source_map,"Toxoplasma_gondii":"Tg"})
    fig,axs=plt.subplots(2,2,figsize=(12,8.2))
    ax=axs[0,0]; q=loso.set_index("short").reindex(["Pf","Pb","Pk","Tg"]); x=np.arange(4); w=.34
    ax.bar(x-w/2,q.AUROC,w,yerr=[q.AUROC-q.AUROC_low,q.AUROC_high-q.AUROC],capsize=3,color=BLUE,label="AUROC")
    ax.bar(x+w/2,q.AUPRC,w,yerr=[q.AUPRC-q.AUPRC_low,q.AUPRC_high-q.AUPRC],capsize=3,color=LIGHT_BLUE,label="AUPRC")
    ax.set_xticks(x,q.index); ax.set_ylim(.5,1); ax.set_ylabel("Target-label ranking"); ax.legend(frameon=False)
    panel(ax,"a","Four-species LOSO")

    for ax,metric,label in [(axs[0,1],"AUROC","b"),(axs[1,0],"AUPRC","c")]:
        f=full_ci[full_ci.metric.eq(metric)].set_index("held_out_species").reindex(list(source_map))
        s=shared_ci[shared_ci.metric.eq(metric)].set_index("held_out_species").reindex(list(source_map))
        y=np.arange(3); ax.axvline(0,color="#777",ls="--",lw=1)
        for vals,off,color,title in [(f,-.08,BLUE,"Full stack"),(s,.08,ORANGE,"Shared molecular")]:
            xval=vals.delta; lo=vals.bootstrap_ci_low; hi=vals.bootstrap_ci_high
            ax.errorbar(xval,y+off,xerr=[xval-lo,hi-xval],fmt="o",capsize=3,color=color,label=title)
        ax.set_yticks(y,["Pf","Pb","Pk"]); ax.set_xlabel(f"Change after adding Tg ({metric})"); ax.legend(frameon=False,fontsize=8)
        panel(ax,label,f"Paired Tg contrast: {metric}")

    ax=axs[1,1]
    allrows=pd.concat([full.assign(series="Full stack"),shared.assign(series="Shared molecular")],ignore_index=True)
    labels=["Pf","Pb","Pk"]; x=np.arange(3); w=.23
    for j,(col,title,color) in enumerate([("two_malaria_sources","2 malaria", "#adc0d0"),("two_malaria_plus_Tg","+ Tg",BLUE),("two_malaria_plus_Tg_shared_features","Shared + Tg",ORANGE)]):
        vals=[]
        for species in ["Plasmodium_falciparum","Plasmodium_berghei","Plasmodium_knowlesi"]:
            if col=="two_malaria_plus_Tg_shared_features":
                row=shared[shared.held_out_species.eq(species)&shared.model.eq(col)]
            else: row=full[full.held_out_species.eq(species)&full.model.eq(col)]
            vals.append(float(row.AUROC.iloc[0]))
        ax.bar(x+(j-1)*w,vals,w,label=title,color=color)
    ax.set_xticks(x,labels); ax.set_ylim(.65,1); ax.set_ylabel("AUROC"); ax.legend(frameon=False,fontsize=8)
    panel(ax,"d","Source-composition comparison")
    fig.tight_layout()
    return save(fig,out,"Fig3")


def fig4(root: Path, out: Path) -> dict:
    base="Fig4_final_frozen_20261024/"
    calls=read(root,base+"Fig4a_species_calls.tsv")
    funnel=read(root,base+"Fig4b_triplet_funnel.tsv")
    patterns=read(root,base+"Fig4c_directional_pattern_counts.tsv")
    hist=read(root,base+"Fig4d_recovery_histogram.tsv")
    orth=read(root,base+"Fig4e_orthomcl_support_summary.tsv")
    fig=plt.figure(figsize=(13,8.5)); gs=fig.add_gridspec(2,3,width_ratios=[1,1.2,1.1],height_ratios=[1,1.1])
    ax=fig.add_subplot(gs[0,0]); x=np.arange(len(calls)); ax.bar(x,calls.predicted_essential,color=[COLORS[s] for s in calls.species_short])
    for i,r in calls.reset_index(drop=True).iterrows(): ax.text(i,r.predicted_essential+55,f"{r.predicted_essential:,}/{r.predicted_essential+r.predicted_nonessential:,}",ha="center",fontsize=8)
    ax.set_xticks(x,calls.species_short); ax.set_ylabel("Predicted-essential genes"); panel(ax,"a","Matched-species atlas")
    ax=fig.add_subplot(gs[0,1]); q=funnel.iloc[::-1]; ax.barh(q.stage,q.orthogroups,color=[GRAY,"#4d7e58","#667f8b",ORANGE,ORANGE,BLUE][:len(q)])
    for i,v in enumerate(q.orthogroups): ax.text(v+35,i,f"{v:,}",va="center",fontsize=8)
    ax.set_xlabel("Orthogroups"); panel(ax,"b","Triplet filtering")
    ax=fig.add_subplot(gs[0,2]); y=np.arange(len(patterns)); ax.barh(y,patterns.candidate_groups-patterns.stable_groups,color=BLUE,label="Candidates")
    ax.barh(y,patterns.stable_groups,left=patterns.candidate_groups-patterns.stable_groups,color=ORANGE,label="Refit-stable")
    ax.set_yticks(y,patterns.new_pattern); ax.set_xlabel("Orthogroups"); ax.legend(frameon=False,fontsize=7); panel(ax,"c","Directional patterns")
    ax=fig.add_subplot(gs[1,:2]); centers=(hist.bin_left+hist.bin_right)/2; width=hist.bin_right-hist.bin_left
    ax.bar(centers,hist.orthogroups,width=width,align="center",color=LIGHT_BLUE,edgecolor="white")
    ax.axvline(.8,color="#c44949",ls="--",label="Prespecified 80% criterion"); ax.set_xlabel("Reference-pattern recovery frequency across 100 paired refits"); ax.set_ylabel("Candidate orthogroups"); ax.legend(frameon=False)
    panel(ax,"d","Recovery across paired refits")
    ax=fig.add_subplot(gs[1,2]); y=np.arange(len(orth)); unsupported=orth.groups-orth.shared_orthomcl_support
    ax.barh(y,orth.shared_orthomcl_support,color=BLUE,label="Shared OrthoMCL"); ax.barh(y,unsupported,left=orth.shared_orthomcl_support,color="#c6cdd2",label="No support")
    ax.set_yticks(y,orth.cohort); ax.set_xlabel("Groups"); ax.legend(frameon=False,fontsize=7); panel(ax,"e","Independent orthology context")
    fig.tight_layout()
    return save(fig,out,"Fig4")


def fig5(root: Path, out: Path) -> dict:
    base="Figure5_context_recomputed_20261023_final/"
    identity=read(root,base+"Fig5_sequence_identity_matched_groups.tsv")
    stats=read(root,base+"Fig5_sequence_identity_statistics.tsv").iloc[0]
    go=read(root,base+"Fig5_GO_enrichment_FDR05.tsv")
    evidence=read(root,base+"Fig5_stable_group_evidence_matrix.tsv")
    orth=read(root,"Fig4_final_frozen_20261024/Fig4e_orthomcl_support_summary.tsv")
    fig,axs=plt.subplots(2,2,figsize=(12,8.4))
    ax=axs[0,0]; cohort=orth[orth.cohort.eq("Refit-stable groups")].iloc[0]
    stable=int(cohort.groups); support=int(cohort.shared_orthomcl_support)
    ax.barh([0],[support],color=BLUE); ax.barh([0],[stable-support],left=[support],color="#c6cdd2")
    ax.text(support/2,0,f"{support}/{stable}",ha="center",va="center",color="white",fontweight="bold")
    ax.set_xlim(0,stable*1.1); ax.set_yticks([0],["Refit-stable groups"]); ax.set_xlabel("Groups"); panel(ax,"a","Independent orthology support")
    ax=axs[0,1]
    stable_values=identity[identity.cohort.eq("Refit-stable directional")].mean_pairwise_global_identity
    control=identity[identity.cohort.str.contains("length",case=False)].mean_pairwise_global_identity
    ax.violinplot([control,stable_values],showmedians=True,showextrema=False)
    ax.set_xticks([1,2],["Length-matched\nE/E/E","Refit-stable\ndirectional"]); ax.set_ylabel("Mean pairwise global identity")
    ax.text(.5,.97,f"Median {stats.control_median_identity:.3f} vs {stats.stable_median_identity:.3f}; MW P={stats.mann_whitney_p_two_sided:.2g}",transform=ax.transAxes,ha="center",va="top",fontsize=8)
    panel(ax,"b","Sequence identity")
    ax=axs[1,0]; go=go.sort_values("fold_enrichment")
    y=np.arange(len(go)); ax.hlines(y,0,go.fold_enrichment,color="#c8cdd1",lw=2); ax.scatter(go.fold_enrichment,y,c="#7656a7",s=45)
    ax.set_yticks(y,go.GO_name); ax.set_xlabel("Fold enrichment"); panel(ax,"c","GO enrichment")
    ax=axs[1,1]
    evidence=evidence.sort_values(["pattern","recovery_frequency"],kind="stable")
    arr=np.vstack([evidence.Pf_call.eq("E"),evidence.Pb_call.eq("E"),evidence.Pk_call.eq("E"),evidence.shared_orthomcl_group_support.astype(bool),evidence.recovery_frequency]).astype(float)
    im=ax.imshow(arr,aspect="auto",interpolation="nearest",cmap="Blues",vmin=0,vmax=1)
    ax.set_yticks(range(5),["Pf call","Pb call","Pk call","OrthoMCL","Recovery"]); ax.set_xticks([]); ax.set_xlabel("184 groups ordered by call pattern and recovery")
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03); panel(ax,"d","Stable-group evidence")
    fig.tight_layout()
    return save(fig,out,"Fig5")


def fig6(root: Path, out: Path) -> dict:
    perf=read(root,"Fig6/F6a_loso_performance.tsv")
    enrich=read(root,"Fig6/F6bc_target_enrichment.tsv")
    recovery=read(root,"Fig6/F6d_target_recovery.tsv")
    fig,axs=plt.subplots(2,2,figsize=(12,8.2))
    ax=axs[0,0]; cols=["AUROC","AUPRC","MCC","sensitivity","specificity"]
    mat=perf.set_index("species_short").reindex(["Pf","Pb","Pk","Tg"])[cols]
    im=ax.imshow(mat.values,aspect="auto",cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(len(cols)),cols,rotation=25,ha="right"); ax.set_yticks(range(4),[f"{s} (n={int(n):,})" for s,n in zip(mat.index,perf.set_index("species_short").reindex(mat.index).held_labelled_genes)])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,f"{mat.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if mat.iloc[i,j]>.55 else "black")
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04); panel(ax,"a","LOSO performance on labelled held-out genes")
    ax=axs[0,1]
    states=["Observed essentiality","LOSO predicted essentiality"]
    groups=[]
    for _,r in enrich.drop_duplicates(["species_short","target_set"]).iterrows():
        s=enrich[(enrich.species_short.eq(r.species_short))&(enrich.target_set.eq(r.target_set))]
        # Plot only the four registered target cohorts used in the figure.
        if r.species_short=="Pf" and "direct" in r.target_set.lower(): label="Pf direct"
        elif r.species_short=="Pf": label="Pf supported"
        elif r.species_short=="Pb": label="Pb supported"
        else: label="Pk supported"
        groups.append((label,s))
    ypos=np.arange(len(groups))[::-1]
    for y,(label,s) in zip(ypos,groups):
        for state,off,color,marker in [(states[0],-.1,"#222222","o"),(states[1],.1,"#2a83a6","s")]:
            r=s[s.essentiality_state.eq(state)].iloc[0]
            ax.errorbar(r.haldane_odds_ratio,y+off,xerr=[[r.haldane_odds_ratio-r.or_ci95_low],[r.or_ci95_high-r.haldane_odds_ratio]],fmt=marker,color=color,capsize=2)
    ax.axvline(1,color="#777",ls="--"); ax.set_xscale("log"); ax.set_yticks(ypos,[x[0] for x in groups]); ax.set_xlabel("Known-target enrichment odds ratio (95% CI)")
    panel(ax,"b","Known targets in observed and predicted sets")
    ax=axs[1,0]; y=np.arange(len(groups))[::-1]
    for yy,(label,s) in zip(y,groups):
        for state,off,color,marker in [(states[0],-.07,"#222222","o"),(states[1],.07,"#2a83a6","s")]:
            r=s[s.essentiality_state.eq(state)].iloc[0]
            ax.scatter(r.target_essential_fraction,yy+off,color=color,marker=marker,s=32)
    ax.set_xlim(0,1.05); ax.set_yticks(y,[x[0] for x in groups]); ax.set_xlabel("Fraction of known targets classified essential")
    panel(ax,"c","Target-member concordance")
    ax=axs[1,1]
    for s,color in [("Pf",BLUE),("Pb",ORANGE),("Pk","#7656a7")]:
        q=recovery[recovery.species_short.eq(s)].sort_values("top_fraction")
        ax.plot(q.top_fraction,q.target_recall,"o-",label=f"{s} (n={int(q.mapped_target_genes.iloc[0])})",color=color)
    ax.plot([.1,1],[.1,1],"--",color="#888",lw=1,label="Uniform ranking")
    ax.set(xlim=(.08,1.02),ylim=(0,1.05),xlabel="Top fraction of frozen LOSO ranking",ylabel="Cumulative known-target recovery"); ax.legend(frameon=False,fontsize=8)
    panel(ax,"d","Threshold-free target recovery")
    fig.tight_layout()
    return save(fig,out,"Fig6")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--release-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); root=args.release_dir/"source_data"; out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    outputs=[fig1(root,out),fig2(root,out),fig3(root,out),fig4(root,out),fig5(root,out),fig6(root,out)]
    inputs=[]
    for rel in SOURCE_INPUTS:
        raw=(root/rel).read_bytes()
        inputs.append({"path":rel,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
    report={"release_id":"v20261024_fivefold_nine_route_candidate_v7","reproduction_scope":"source-data figure regeneration; no model fitting or manuscript replacement","source_data_root":"source_data/","runtime":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__},"input_files":inputs,"figures":outputs}
    (out/"figure_reproduction_manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
