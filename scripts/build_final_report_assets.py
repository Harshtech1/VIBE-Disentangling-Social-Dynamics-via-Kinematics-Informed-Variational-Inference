#!/usr/bin/env python3
"""Build report-local, non-scientific assets from verified project artifacts."""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'reports/final_report/assets'; F=A/'figures'; C=A/'code'
def save(fig,name):
 F.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(F/f'{name}.png',dpi=300,bbox_inches='tight');fig.savefig(F/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def boxes(name,title,items,colors=None,vertical=False):
 fig,ax=plt.subplots(figsize=(11,4.3 if not vertical else 8));ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',fontsize=14)
 n=len(items)
 for i,item in enumerate(items):
  x=.05+(i%(3 if vertical else n))*(.9/(3 if vertical else n)); y=.75-(i//3)*.28 if vertical else .48; w=.9/(3 if vertical else n)-.04; h=.16 if vertical else .22
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor=(colors or ['#dbeafe'])[i%len(colors or ['#dbeafe'])],edgecolor='#334155'))
  ax.text(x+w/2,y+h/2,item,ha='center',va='center',fontsize=9,wrap=True)
  if (not vertical and i<n-1):ax.annotate('',(x+w+.02,y+h/2),(x+w+.002,y+h/2),arrowprops={'arrowstyle':'->'})
 ax.set(xlim=(0,1),ylim=(0,1));save(fig,name)
def bar(name,title,labels,values,colors,note=''):
 fig,ax=plt.subplots(figsize=(7.8,4.5));bars=ax.bar(labels,values,color=colors,edgecolor='#334155');ax.set(title=title,ylabel='Count');ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
 for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+max(values)*.02,str(v),ha='center',fontweight='bold')
 if note:ax.text(.5,-.2,note,transform=ax.transAxes,ha='center',fontsize=8);save(fig,name)
def main():
 F.mkdir(parents=True,exist_ok=True);C.mkdir(parents=True,exist_ok=True)
 copies={'split_class_distribution':'reports/submission/gecv_vibe_final_reconstruction_package/figures/evaluation/split_class_distribution.png','three_seed_test_metric_summary':'reports/submission/gecv_vibe_final_reconstruction_package/figures/summary/three_seed_test_metric_summary.png','validation_performance_evolution':'reports/submission/gecv_vibe_final_reconstruction_package/figures/validation/validation_performance_evolution.png','test_confusion_matrices':'reports/submission/gecv_vibe_final_reconstruction_package/figures/confusion/test_confusion_matrices.png','per_class_test_metrics':'reports/submission/gecv_vibe_final_reconstruction_package/figures/evaluation/per_class_test_metrics.png','training_loss_evolution':'reports/submission/gecv_vibe_final_reconstruction_package/figures/training/training_loss_evolution.png','learning_rate_schedule':'reports/submission/gecv_vibe_final_reconstruction_package/figures/training/learning_rate_schedule.png','historical_v1_0_k':'reports/submission/gecv_vibe_final_reconstruction_package/figures/k_sensitivity/historical_v1_0_effective_k_sensitivity.png'}
 for n,p in copies.items():shutil.copy2(ROOT/p,F/f'{n}.png')
 for src in ['dataloading/vibe_dataset.py','dataloading/reconstruction_split.py','model/vibe_model.py','model/vib.py','model/context_gate.py','model/gamma_transformer.py','model/learned_attenttion.py','losses/loss_function.py','trainer/train_model.py','tests/gamma_scene_scale_invariance_cpu.py','scripts/build_gecv_final_reconstruction_package.py']:
  shutil.copy2(ROOT/src,C/src.replace('/','__'))
 bar('raw_inventory','Raw readable MP4 inventory',['Positive / H','Neutral / N','Negative / S'],[201,107,100],['#4c78a8','#72b7b2','#e45756'],'408 readable videos; this is not the 404-record experimental corpus.')
 bar('processing_outcome','Reconstruction processing outcomes',['Pass','Resume / skip','H116 tracking failed'],[365,42,1],['#59a14f','#edc949','#e15759'],'H116: no pedestrian detections were obtained.')
 bar('final_dataset','Final experimental HDF5 records',['Positive','Neutral','Negative'],[199,106,99],['#4c78a8','#72b7b2','#e45756'],'Total: 404 experimental records.')
 boxes('project_overview','Project overview (conceptual, not an empirical result)',['Original VIBE paper','Repository inspection','Data reconstruction','Feature extraction','Dataset audit','Historical V1.0','Implementation audit','Gamma correction','V1.1 baseline','3-seed package'],vertical=True,colors=['#dbeafe','#dcfce7','#fef3c7'])
 boxes('feature_pipeline','Feature extraction pipeline (conceptual)',['Raw video','6 Hz decode','YOLOv8l + ByteTrack','K=8 agents + centroids','DINOv2 local','HuBERT audio','VideoMAEv2 global','T=32 + masks','Synchrony / gamma','HDF5'],vertical=True,colors=['#dbeafe','#fef3c7','#dcfce7'])
 boxes('hdf5_schema','HDF5 schema: model tensors and provenance (conceptual)',['MODEL: local [8,32,768]\naudio [32,768]\nglobal [768]','MASKS: agent [8]\nlocal time [8,32]\naudio time [32]','SYNCHRONY: sync [8,8]\npair mask [8,8]\ngamma scalar','PROVENANCE: frames [32]\ntimestamps [32]\nselected IDs variable'],colors=['#dbeafe','#dcfce7','#fef3c7','#f3e8ff'])
 boxes('vibe_architecture','Current VIBE reconstruction architecture (conceptual)',['Local visual + masks','Affective / environment VIB','Context gate','Audio cross-modal attention','Gamma-gated transformer\n+ AdaLN','Temporal pooling','Gamma fusion','3-class classifier','TEXT/SAT: disabled'],vertical=True,colors=['#dbeafe','#dcfce7','#fef3c7'])
 boxes('synchrony_gamma','Synchrony to gamma (conceptual)',['Person centroid trajectories','Velocity vectors','Pairwise cosine similarity','Active valid pairs','Mean synchrony = gamma'],colors=['#dbeafe','#dcfce7'])
 boxes('gamma_bug','Gamma/scene-scale correction (implementation-level reconstruction issue)',['V1.0\ngamma [B,1,1]\nscale [B,1]\n→ [B,B,1]\nbatch coupling','V1.1\ngamma [B,1,1]\nscale [B,1,1]\n→ [B,1,1]\nsample-wise','Probe\nbefore: 9.23e-05 companion\n5.25e-06 order\nafter: both 0.0'],colors=['#fee2e2','#dcfce7','#dbeafe'])
 boxes('vib_loss','VIB and loss composition (conceptual)',['Input features','mu + log variance','Affective latent\nEnvironmental latent','Classification / HCE','Orthogonality','KL: affect branch only','SAT: disabled','Total loss'],vertical=True,colors=['#dbeafe','#dcfce7','#fef3c7'])
 boxes('provenance_chain','Reproducibility chain (conceptual)',['Source corpus hash','Frozen split','Config + code SHA','Checkpoint SHA','Selected test result','Final report + ZIP'],colors=['#dbeafe','#dcfce7'])
 boxes('report_layers','How to read this report (conceptual)',['1. Published paper proposal','2. Available repository/data','3. Reconstruction and measurements','4. Pending and out-of-scope work'],colors=['#dbeafe','#dcfce7','#fef3c7','#fee2e2'])
 boxes('timeline','Evidence-based project timeline (conceptual)',['Paper study','Repository/data audit','Extraction + HDF5','Frozen split','Historical V1.0','Forensic audit','V1.1 correction','3-seed V1.1','Final package'],vertical=True,colors=['#dbeafe','#dcfce7','#fef3c7'])
 print(f'figures={len(list(F.glob("*.png")))} code_snapshots={len(list(C.glob("*.py")))}')
if __name__=='__main__':main()
