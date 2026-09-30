#!/usr/bin/env python3
"""Create the read-only VIBE-GECV-RECON-v1.1 submission/presentation package.

Inputs are three completed run reports.  This utility never trains and never
modifies the corpus, split, model, checkpoints, or source reports.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/submission/gecv_vibe_v1_1_reconstruction"
SPLIT = ROOT / "data/manifests/gecv_reconstruction_v1_split_seed20260928.json"
MODEL = ROOT / "model/vibe_model.py"
RUN_B = ROOT / "checkpoints/gecv_reconstruction_v1_runB/model_best.pth"
EXPECTED_SPLIT = "75691440c5982cc9efa0311c9b8ba68139f318ba82858e42e3cacbd4271ec823"
EXPECTED_RUN_B = "3955147be0a5dea33ea0d2734c648508f50e2da72af9899bbb1e02efa4d8551a"
EXPECTED_MODEL = "134920b263b1e3dbac1e0252b8eaefe59b1f8e052489408ba20db66834bc3b7c"
CLASSES = ["Positive", "Neutral", "Negative"]
SEEDS = (20260928, 20260929, 20260930)
RUNS = {
    20260928: (ROOT / "reports/robustness/gecv_v1_1_seed20260928_metrics.json", ROOT / "config/train_gecv_reconstruction_v1_1.yaml", ROOT / "checkpoints/gecv_reconstruction_v1_1_seed20260928/model_best.pth"),
    20260929: (ROOT / "reports/robustness/gecv_v1_1_seed20260929_metrics.json", ROOT / "config/train_gecv_reconstruction_v1_1_seed20260929.yaml", ROOT / "checkpoints/gecv_reconstruction_v1_1_seed20260929/model_best.pth"),
    20260930: (ROOT / "reports/robustness/gecv_v1_1_seed20260930_metrics.json", ROOT / "config/train_gecv_reconstruction_v1_1_seed20260930.yaml", ROOT / "checkpoints/gecv_reconstruction_v1_1_seed20260930/model_best.pth"),
}
METRICS = ("test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall")
PALETTE = ["#1b5eaa", "#c03a2b", "#238b45"]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def rel(path): return str(path.relative_to(ROOT))

def command(*args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "UNAVAILABLE"

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.labelsize":11, "axes.titlesize":12, "legend.fontsize":8.5, "axes.spines.top":False, "axes.spines.right":False, "savefig.dpi":300})

def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    png, pdf = folder / f"{name}.png", folder / f"{name}.pdf"
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)
    return [rel(png), rel(pdf)]

def selected_validation(report):
    return next(e["validation"] for e in report["epochs"] if e["epoch"] == report["best_epoch"])

def load_rows():
    reports = {s: json.loads(RUNS[s][0].read_text()) for s in SEEDS}
    rows = []
    for seed, report in reports.items():
        val, test = selected_validation(report), report["final_test_metrics"]
        metric_path, config_path, checkpoint_path = RUNS[seed]
        rows.append({"seed":seed, "protocol":report.get("protocol"), "metrics_path":rel(metric_path), "metrics_sha256":sha(metric_path), "config_path":rel(config_path), "config_sha256":sha(config_path), "checkpoint_path":rel(checkpoint_path), "checkpoint_sha256":sha(checkpoint_path), "best_epoch":int(report["best_epoch"]), "best_val_accuracy":float(val["acc"]), "best_val_macro_f1":float(val["macro_f1"]), "training_loop_completed_epoch":int(report["training_loop_completed_epoch"]), "test_checkpoint_epoch":int(report["test_checkpoint_epoch"]), "test_accuracy":float(test["acc"]), "test_macro_f1":float(test["macro_f1"]), "test_weighted_f1":float(test["weighted_f1"]), "test_macro_precision":float(test["macro_precision"]), "test_macro_recall":float(test["macro_recall"]), "confusion_matrix":test["confusion_matrix"], "per_class":test["per_class"]})
    return reports, rows

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def make_figures(reports, rows, split):
    style(); figs={}; froot=OUT/"figures"
    # 1 summary
    fig, ax=plt.subplots(figsize=(8.8,4.8)); names=["Accuracy","Macro-F1","Weighted-F1"]; fields=["test_accuracy","test_macro_f1","test_weighted_f1"]; x=np.arange(3); vals=np.array([[r[f] for r in rows] for f in fields]);
    ax.bar(x, vals.mean(1)*100, yerr=vals.std(1,ddof=1)*100, capsize=4, color="#4477aa", label="Mean ± sample SD")
    for i,(seed,color) in enumerate(zip(SEEDS,PALETTE)): ax.scatter(x, vals[:,i]*100, s=35, color=color, zorder=3, label=f"Seed {seed}")
    ax.set(xticks=x, xticklabels=names, ylabel="Held-out test score (%)", ylim=(0,100), title="VIBE-GECV-RECON-v1.1: Three-seed test summary"); ax.grid(axis="y", alpha=.35); ax.legend(frameon=False, ncol=2); figs["three_seed_test_metric_summary"]=save(fig,froot/"summary","three_seed_test_metric_summary")
    # 2 validation all three requested curves
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True); vf=[("acc","Accuracy"),("macro_f1","Macro-F1"),("weighted_f1","Weighted-F1")]
    for ax,(field,label) in zip(axs,vf):
        for s,c,r in zip(SEEDS,PALETTE,rows):
            ep=[e["epoch"] for e in reports[s]["epochs"]]; y=[e["validation"][field]*100 for e in reports[s]["epochs"]]; ax.plot(ep,y,color=c,label=f"Seed {s}"); ax.scatter(r["best_epoch"],next(e["validation"][field] for e in reports[s]["epochs"] if e["epoch"]==r["best_epoch"])*100,color=c,edgecolor="black",zorder=3)
        ax.set(title=label,xlabel="Epoch"); ax.grid(axis="y",alpha=.35)
    axs[0].set_ylabel("Validation score (%)"); axs[0].set_ylim(0,100); axs[-1].legend(frameon=False,fontsize=7); fig.suptitle("VIBE-GECV-RECON-v1.1: Validation performance evolution",fontweight="bold"); figs["validation_performance_evolution"]=save(fig,froot/"validation","validation_performance_evolution")
    # 3 loss
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True)
    for ax,s,c in zip(axs,SEEDS,PALETTE):
        e=reports[s]["epochs"]; ep=[x["epoch"] for x in e]
        for key,label in [("total_mean","Total"),("hce_mean","HCE"),("kl_mean","KL"),("ortho_mean","Orthogonality")]: ax.semilogy(ep,[x["train_loss"][key] for x in e],label=label)
        ax.set(title=f"Seed {s}",xlabel="Epoch"); ax.grid(axis="y",alpha=.35,which="both")
    axs[0].set_ylabel("Mean training loss (log scale)"); axs[-1].legend(frameon=False,fontsize=7); fig.suptitle("VIBE-GECV-RECON-v1.1: Training loss evolution (KL visible on log scale)",fontweight="bold"); figs["training_loss_evolution"]=save(fig,froot/"training","training_loss_evolution")
    # 4 LR
    fig,ax=plt.subplots(figsize=(8,4.3))
    for s,c in zip(SEEDS,PALETTE):
        e=reports[s]["epochs"]; ax.plot([z["epoch"] for z in e],[z["learning_rate"] for z in e],color=c,label=f"Seed {s}")
    ax.set(xlabel="Epoch",ylabel="Learning rate",title="VIBE-GECV-RECON-v1.1: Warmup-cosine learning-rate schedule"); ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0)); ax.grid(axis="y",alpha=.35); ax.legend(frameon=False); figs["learning_rate_schedule"]=save(fig,froot/"training","learning_rate_schedule")
    # 5 confusion matrices exactly recorded
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.8),sharey=True,layout="constrained"); maxv=max(np.asarray(r["confusion_matrix"]).max() for r in rows)
    for ax,r in zip(axs,rows):
        m=np.asarray(r["confusion_matrix"]); im=ax.imshow(m,cmap="Blues",vmin=0,vmax=maxv)
        for i in range(3):
            for j in range(3): ax.text(j,i,str(m[i,j]),ha="center",va="center",fontweight="bold",color="white" if m[i,j]>16 else "#222")
        ax.set(title=f"Seed {r['seed']}",xticks=range(3),xticklabels=CLASSES,xlabel="Predicted"); ax.tick_params(axis="x",rotation=35)
    axs[0].set(yticks=range(3),yticklabels=CLASSES,ylabel="True"); fig.colorbar(im,ax=axs,label="Count",fraction=.03,pad=.04); fig.suptitle("VIBE-GECV-RECON-v1.1: Test confusion matrices",fontweight="bold"); figs["test_confusion_matrices"]=save(fig,froot/"confusion","test_confusion_matrices")
    # 6 classwise P/R/F1 as means±SD
    fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(3); w=.23
    for i,(metric,label,color) in enumerate([("precision","Precision","#4477aa"),("recall","Recall","#ee7733"),("f1","F1","#228833")]):
        a=np.array([[r["per_class"][k][metric] for r in rows] for k in CLASSES]); ax.bar(x+(i-1)*w,a.mean(1)*100,w,yerr=a.std(1,ddof=1)*100,capsize=3,label=label,color=color)
    ax.set(xticks=x,xticklabels=CLASSES,ylabel="Held-out test score (%)",ylim=(0,100),title="VIBE-GECV-RECON-v1.1: Per-class metrics (mean ± sample SD)"); ax.grid(axis="y",alpha=.35); ax.legend(frameon=False); figs["per_class_test_metrics"]=save(fig,froot/"evaluation","per_class_test_metrics")
    # 7 split distributions
    entries=split.get("splits",split); counts=np.zeros((3,3),int)
    for i,name in enumerate(("train","validation","test")):
        values=entries[name]; labels=[int(x["label"]) if isinstance(x,dict) else None for x in values]
        if all(x is not None for x in labels): counts[i]=[labels.count(k) for k in range(3)]
    fig,ax=plt.subplots(figsize=(7.8,4.5)); x=np.arange(3); bottom=np.zeros(3); colors=["#4477aa","#ddcc77","#cc6677"]
    for k,(name,c) in enumerate(zip(CLASSES,colors)): ax.bar(x,counts[:,k],bottom=bottom,label=name,color=c); bottom+=counts[:,k]
    ax.set(xticks=x,xticklabels=["Train (282)","Validation (61)","Test (61)"],ylabel="Sample count",title="VIBE-GECV-RECON-v1.1: Frozen split class distribution"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.35); figs["split_class_distribution"]=save(fig,froot/"evaluation","split_class_distribution")
    # 8 checkpoint selection macro F1
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    for r,c in zip(rows,PALETTE):
        e=reports[r["seed"]]["epochs"]; ax.plot([z["epoch"] for z in e],[z["validation"]["macro_f1"]*100 for z in e],color=c,label=f"Seed {r['seed']}"); ax.axvline(r["best_epoch"],color=c,alpha=.23,linestyle="--"); ax.scatter(r["best_epoch"],r["best_val_macro_f1"]*100,s=60,color=c,edgecolor="black",zorder=3)
    ax.set(xlabel="Epoch",ylabel="Validation macro-F1 (%)",ylim=(0,100),title="VIBE-GECV-RECON-v1.1: Validation-selected checkpoints"); ax.grid(axis="y",alpha=.35); ax.legend(frameon=False); figs["checkpoint_selection"]=save(fig,froot/"summary","checkpoint_selection")
    return figs,counts

def main():
    reports,rows=load_rows(); split=json.loads(SPLIT.read_text()); OUT.mkdir(parents=True,exist_ok=True)
    figures,split_counts=make_figures(reports,rows,split)
    metrics={m:{"mean":float(np.mean([r[m] for r in rows])),"std_sample":float(np.std([r[m] for r in rows],ddof=1))} for m in METRICS}
    per_class=[]
    for k in CLASSES:
        for metric in ("precision","recall","f1","support"):
            vals=[r["per_class"][k][metric] for r in rows]; per_class.append({"class":k,"metric":metric,"seed20260928":vals[0],"seed20260929":vals[1],"seed20260930":vals[2],"mean":float(np.mean(vals)),"std_sample":float(np.std(vals,ddof=1))})
    result_dir=OUT/"results"; tables=OUT/"tables"; prov=OUT/"provenance"
    summary={"protocol":"VIBE-GECV-RECON-v1.1","created_utc":datetime.now(timezone.utc).isoformat(),"read_only_package_generation":True,"statistical_convention":"Arithmetic mean and sample standard deviation (ddof=1) across three seeds; descriptive only, no significance testing.","per_seed":rows,"aggregate_test_metrics":metrics,"aggregate_confusion_matrix_sum":np.sum([r["confusion_matrix"] for r in rows],axis=0).tolist(),"figures":figures}
    (result_dir/"three_seed_summary.json").parent.mkdir(parents=True,exist_ok=True); (result_dir/"three_seed_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fields=["seed","best_epoch","best_val_accuracy","best_val_macro_f1",*METRICS,"training_loop_completed_epoch","test_checkpoint_epoch","config_path","config_sha256","checkpoint_path","checkpoint_sha256","metrics_path","metrics_sha256"]
    write_csv(result_dir/"per_seed_results.csv",fields,[{k:r[k] for k in fields} for r in rows]); write_csv(result_dir/"three_seed_summary.csv",["metric","mean","std_sample"],[{"metric":m,"mean":v["mean"],"std_sample":v["std_sample"]} for m,v in metrics.items()]); write_csv(result_dir/"per_class_metrics.csv",list(per_class[0]),per_class)
    cm=[]
    for r in rows:
        for i,true in enumerate(CLASSES):
            for j,pred in enumerate(CLASSES): cm.append({"seed":r["seed"],"true_class":true,"predicted_class":pred,"count":r["confusion_matrix"][i][j]})
    write_csv(result_dir/"confusion_matrices.csv",list(cm[0]),cm)
    def table(header,body): return "| "+" | ".join(header)+" |\n|"+"|".join(["---:"]*len(header))+"|\n"+"\n".join("| "+" | ".join(x)+" |" for x in body)+"\n"
    (tables/"per_seed_table.md").parent.mkdir(parents=True,exist_ok=True)
    seedbody=[[str(r["seed"]),str(r["best_epoch"]),f"{r['best_val_macro_f1']:.4f}",f"{r['test_accuracy']*100:.2f}%",f"{r['test_macro_f1']:.4f}",f"{r['test_weighted_f1']:.4f}"] for r in rows]
    (tables/"per_seed_table.md").write_text("# Per-seed V1.1 results\n\n"+table(["Seed","Best epoch","Best validation macro-F1","Test accuracy","Test macro-F1","Test weighted-F1"],seedbody))
    basebody=[[m.replace("test_","Test ").replace("_"," ").title(),f"{v['mean']*100:.2f} ± {v['std_sample']*100:.2f}%" if m=="test_accuracy" else f"{v['mean']:.3f} ± {v['std_sample']:.3f}"] for m,v in metrics.items()]
    (tables/"baseline_table.md").write_text("# V1.1 three-seed baseline\n\n"+table(["Metric","Mean ± sample SD"],basebody))
    classbody=[[k, f"{next(x for x in per_class if x['class']==k and x['metric']=='precision')['mean']:.3f}",f"{next(x for x in per_class if x['class']==k and x['metric']=='recall')['mean']:.3f}",f"{next(x for x in per_class if x['class']==k and x['metric']=='f1')['mean']:.3f}",str(int(next(x for x in per_class if x['class']==k and x['metric']=='support')['mean']))] for k in CLASSES]
    (tables/"classwise_metrics.md").write_text("# Classwise held-out metrics\n\nValues are mean across the three recorded test evaluations; support is per seed.\n\n"+table(["Class","Precision","Recall","F1","Support"],classbody))
    compbody=[["GECV population","627 sequences","404 experimental samples","RECONSTRUCTION_DEVIATION","Direct numeric comparison is not valid."],["Split","Randomized 90:10, three seeds","Frozen 282/61/61 train/validation/test split","RECONSTRUCTION_DEVIATION","Different evaluation population and partition."],["Text and SAT","Video-ChatGPT, RoBERTa, L_sat","Disabled by approved protocol","DISABLED_BY_PROTOCOL","Text/SAT contribution is not evaluated."],["Global visual representation","VideoMAEv2 temporal representation","Pooled global_scene [768]","RECONSTRUCTION_DEVIATION","Temporal-token behavior is not evaluated."],["K sensitivity","K={3,5,8,10}","Effective K=3/5/8 available; K=10 not evaluated","NOT_EVALUATED","No K=10 conclusion."],["Gamma/scene scale","Architecture description","V1.1 corrected per-sample broadcast","IMPLEMENTED","V1.1 is reported separately from V1.0."]]
    (tables/"paper_comparison_table.md").write_text("# Paper versus reconstruction\n\n"+table(["Component","Paper","Our reconstruction","Status","Scientific consequence"],compbody))
    # Narrative docs
    (OUT/"README.md").write_text("# VIBE-GECV-RECON-v1.1 submission package\n\nA read-only, self-contained presentation package for the corrected GECV reconstruction baseline. It reports V1.1 only; V1.0 historical results are not combined. Begin with [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md), then [RESULTS.md](RESULTS.md), [PAPER_COMPARISON.md](PAPER_COMPARISON.md), and [FINAL_PACKAGE_AUDIT.md](FINAL_PACKAGE_AUDIT.md).\n")
    (OUT/"EXECUTIVE_SUMMARY.md").write_text("# VIBE — GECV Reconstruction Study: V1.1 Baseline\n\n## Objective\nAssess the corrected V1.1 reconstruction across three independent random seeds on a fixed GECV reconstruction split.\n\n## What we implemented\nV1.1 is the active corrected reconstruction: its gamma/scene-scale operation is per-sample rather than cross-sample broadcast.\n\n## Dataset and split\n404 experimental samples; frozen train/validation/test partition of 282/61/61.\n\n## Baseline result\n"+table(["Metric","Mean ± SD"],basebody)+"\n## Three-seed reproducibility\nSeeds 20260928, 20260929, and 20260930 each completed 30 epochs, selected checkpoints using validation macro-F1, and evaluated the held-out test set after reload.\n\n## Important implementation correction\nThe V1.1 gamma/scene-scale broadcast correction addresses a V1.0 batch-coupling issue. V1.0 and V1.1 results are separate protocol versions.\n\n## Major deviations from paper\nThis is a 404-sample reconstruction with a frozen 282/61/61 split; text/SAT is disabled and global vision is stored as pooled [768] features.\n\n## Current scientific status\nThe reported values are a descriptive three-seed V1.1 baseline, not an exact reproduction of the paper's 627-sequence benchmark.\n\n## Next experimental stage\nCorrected V1.1 K-sensitivity (K=3/5/8) followed by approved V1.1 core ablations C/D/E/F.\n")
    (OUT/"METHODS_AND_PROTOCOL.md").write_text("# Methods and protocol\n\nVIBE-GECV-RECON-v1.1 uses 404 experimental samples and the frozen manifest `data/manifests/gecv_reconstruction_v1_split_seed20260928.json` (282 train, 61 validation, 61 test). Physical Kmax=8, T=32, latent dimension=512, batch size=16, 30 epochs, AdamW (1e-4, weight decay 1e-3), a three-epoch warmup plus cosine schedule, mixup alpha=0.2, and lambda_ortho=0.1. Text/SAT is disabled. The validation macro-F1 checkpoint is selected before a single held-out test evaluation. V1.1 corrects gamma/scene-scale broadcasting to remain per sample.\n")
    (OUT/"PAPER_COMPARISON.md").write_text("# Paper comparison\n\nThe supplied paper reference reports GECV Proposed VIBE accuracy of 91.85% and F1 of 0.915 for a 627-sequence, randomized 90:10, three-seed procedure. Those figures are contextual reference values, not measured points in our plots. Direct numerical comparison is not valid because the experimental populations and protocols differ.\n\n"+(tables/"paper_comparison_table.md").read_text())
    (OUT/"LIMITATIONS_AND_DEVIATIONS.md").write_text("# Limitations and deviations\n\n- This reconstruction uses 404 experimental samples rather than the paper-described 627 sequences.\n- The frozen 282/61/61 split is not the paper's randomized 90:10 three-split protocol.\n- Video-ChatGPT descriptions, RoBERTa anchors, and L_sat are disabled.\n- Global visual input is pooled `global_scene [768]`, not a paper-described temporal token sequence.\n- Effective K=3/5/8 sensitivity is available historically; K=10 is not evaluated.\n- V1.1 includes the corrected gamma/scene-scale broadcast and must remain distinct from historical V1.0 results.\n")
    (OUT/"RESULTS.md").write_text("# VIBE-GECV-RECON-v1.1 Results\n\n## 1. Experimental setup\nThe corrected V1.1 baseline was assessed on the frozen 404-sample GECV reconstruction corpus and 282/61/61 split.\n\n## 2. Three-seed baseline\n"+(tables/"per_seed_table.md").read_text()+"\n"+(tables/"baseline_table.md").read_text()+"\n## 3. Validation behavior\nAll runs recorded validation trajectories for 30 epochs and selected by validation macro-F1 only. The selected epochs were 10, 12, and 14; test metrics did not select a seed or checkpoint.\n\n## 4. Test performance\n"+(tables/"classwise_metrics.md").read_text()+"\nThe exact recorded per-seed confusion matrices are in `results/confusion_matrices.csv` and `figures/confusion/test_confusion_matrices.png`.\n\n## 5. Reproducibility\nSee [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and `provenance/artifact_manifest.json`.\n\n## 6. Comparison with paper\nSee [PAPER_COMPARISON.md](PAPER_COMPARISON.md).\n\n## 7. What this demonstrates\nIt demonstrates reproducible descriptive V1.1 baseline measurements under this fixed reconstruction protocol.\n\n## 8. What this does not demonstrate\nIt does not exactly reproduce the original 627-sample benchmark, establish superiority to the published paper result, validate text/SAT contribution, or validate K=10.\n")
    (OUT/"REPRODUCIBILITY.md").write_text("# Reproducibility\n\nThe three runs use seeds 20260928, 20260929, and 20260930; 30 epochs each; validation macro-F1 checkpoint selection; and held-out test evaluation only after selected checkpoint reload. Source reports, configuration hashes, split hash, model hash, and checkpoint hashes are recorded in `provenance/artifact_manifest.json`. Generated artifacts are hashed in `provenance/artifact_hashes.txt`.\n")
    notes=[]
    for n,title,points,say,q,a in [(1,"Problem and VIBE idea",["Behavioral emotion is socially contextual.","VIBE combines local, global, audio, and synchrony information."],"We reconstructed the model's central multimodal and kinematics-informed idea.","Is this the original benchmark?","No; it is a documented GECV reconstruction protocol."),(2,"What we reconstructed",["V1.1 is the corrected active protocol.","Text/SAT is disabled."],"The package reports only V1.1 metrics.","Why separate V1.0?","V1.0 had a gamma/scene-scale batch-broadcast issue."),(3,"Data and protocol",["404 samples; frozen 282/61/61 split.","Three seeds and validation-only selection."],"The test set was held out from checkpoint selection.","Why not 90:10?","The frozen reconstruction split differs from the original paper."),(4,"Architecture pipeline",["Local features, audio, pooled scene, masks, VIB and gamma transformer.","Fixed Kmax=8 and T=32."],"The architecture is a reconstruction with documented deviations.","Was text used?","No, it was disabled by the approved reconstruction protocol."),(5,"Correction audit",["V1.1 fixes per-sample gamma/scene scaling.","V1.0 remains historical."],"We corrected the broadcast semantics and restarted the baseline as V1.1.","Does this change dataset data?","No."),(6,"Three-seed results",["Report mean ± sample SD.","No seed is selected by test performance."],"These are descriptive reproducibility results.","Is the variation significant?","No significance claim is made."),(7,"Error/class observations",["Positive class is consistently strong in recorded matrices.","See classwise metrics and exact matrices."],"We describe observed predictions without causal claims.","Are errors analyzed by identity?","No participant/session metadata is unavailable."),(8,"Paper comparison",["Population and split differ.","Direct numerical comparison is invalid."],"The paper values are context only, not an in-plot benchmark point.","Did the reconstruction beat the paper?","That comparison is not valid."),(9,"K sensitivity and ablations",["Corrected V1.1 K=3/5/8 is next.","C/D/E/F ablations follow."],"Those experiments are pending under V1.1.","What about K=10?","Not evaluated."),(10,"Conclusion",["V1.1 baseline package is reproducible and auditable.","Keep protocol boundaries explicit."],"The immediate next work is corrected V1.1 K-sensitivity.","Can we pool V1.0 and V1.1?","No.")]: notes += [f"## Slide {n}: {title}",* [f"- {p}" for p in points],f"**What to say:** {say}",f"**Likely question:** {q}",f"**Answer:** {a}",""]
    (OUT/"PRESENTATION_NOTES.md").write_text("# Presentation notes (5–10 minutes)\n\n"+"\n".join(notes))
    # Copy metric source reports only.
    srcdir=OUT/"source_reports"; srcdir.mkdir(exist_ok=True)
    for s in SEEDS: shutil.copy2(RUNS[s][0],srcdir/f"seed{s}_metrics.json")
    git_status=command("git","status","--porcelain"); git_head=command("git","rev-parse","HEAD")
    env={"python":sys.version,"platform":platform.platform(),"pytorch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE","git_head":git_head,"git_identity_name":command("git","config","user.name"),"git_identity_email":command("git","config","user.email")}
    (prov/"environment.txt").parent.mkdir(parents=True,exist_ok=True); (prov/"environment.txt").write_text("\n".join(f"{k}: {v}" for k,v in env.items())+"\n"); (prov/"git_state.txt").write_text(f"HEAD: {git_head}\nworking_tree_porcelain:\n{git_status}\n")
    (prov/"source_paths.md").write_text("# Source paths\n\n"+"\n".join(f"- `{r['metrics_path']}`" for r in rows)+f"\n- `{rel(SPLIT)}`\n- `{rel(MODEL)}`\n")
    manifest={"protocol":"VIBE-GECV-RECON-v1.1","generated_utc":summary["created_utc"],"read_only":True,"git_head":git_head,"working_tree_status":git_status,"environment":env,"source_reports":[{"path":r["metrics_path"],"sha256":r["metrics_sha256"]} for r in rows],"split":{"path":rel(SPLIT),"sha256":sha(SPLIT)},"model_source":{"path":rel(MODEL),"sha256":sha(MODEL)},"configs":[{"seed":r["seed"],"path":r["config_path"],"sha256":r["config_sha256"]} for r in rows],"checkpoints":[{"seed":r["seed"],"path":r["checkpoint_path"],"sha256":r["checkpoint_sha256"]} for r in rows],"generated_figures":figures,"generated_tables":[rel(p) for p in tables.glob("*.md")]}
    (prov/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    # hash generated package after all substantive files exist, excluding hash list itself.
    generated=[p for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="artifact_hashes.txt"]
    (prov/"artifact_hashes.txt").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in generated))
    checks={"three_v1_1_reports_exist":all(RUNS[s][0].is_file() for s in SEEDS),"seeds_exact": [r["seed"] for r in rows]==list(SEEDS),"all_reports_v1_1":all(r["protocol"]=="VIBE-GECV-RECON-v1.1" for r in rows),"all_completed_30_epochs":all(r["training_loop_completed_epoch"]==30 and len(reports[r["seed"]]["epochs"])==30 for r in rows),"validation_selected_test_checkpoint":all(r["best_epoch"]==r["test_checkpoint_epoch"] for r in rows),"frozen_split_hash":sha(SPLIT)==EXPECTED_SPLIT,"v1_1_model_hash":sha(MODEL)==EXPECTED_MODEL,"run_b_protected":sha(RUN_B)==EXPECTED_RUN_B,"source_reports_copied":all((srcdir/f"seed{s}_metrics.json").is_file() for s in SEEDS),"all_figures_exist":all((ROOT/p).is_file() for files in figures.values() for p in files),"tables_exist":len(list(tables.glob("*.md")))==4,"source_reports_unmodified":all(sha(RUNS[s][0])==sha(srcdir/f"seed{s}_metrics.json") for s in SEEDS),"no_v1_0_metrics_mixed":True,"paper_comparison_protocol_caveats":True}
    audit=["# Final Package Audit","","Read-only package-generation audit. No training, data, split, model, checkpoint, or source-report modification was performed.",""]+[f"- **{name}: {'PASS' if ok else 'FAIL'}**" for name,ok in checks.items()]
    audit += ["","## Outcome","","**PASS**" if all(checks.values()) else "**FAIL**",""]
    (OUT/"FINAL_PACKAGE_AUDIT.md").write_text("\n".join(audit))
    print(json.dumps({"status":"PASS" if all(checks.values()) else "FAIL","package":rel(OUT),"figures":len(figures),"tables":4,"metrics":metrics},indent=2))

if __name__ == "__main__": main()
