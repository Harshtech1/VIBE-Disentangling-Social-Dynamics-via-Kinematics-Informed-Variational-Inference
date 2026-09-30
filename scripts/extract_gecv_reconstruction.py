#!/usr/bin/env python3
"""Resumable VIBE-GECV-RECON-v1.0 corpus extractor. Never launches automatically."""

from __future__ import annotations
import argparse, importlib.util, json, os, shutil, subprocess, tarfile, tempfile, time
from pathlib import Path
from typing import Any
import cv2, h5py, numpy as np, soundfile as sf, torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel, Dinov2Model, HubertModel, Wav2Vec2FeatureExtractor
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from data_preprocessing.reconstruction_contract import map_local_tracks, select_k8_duration_confidence_id, shared_interval_synchrony, shared_time_grid, bin_pool
from data_preprocessing.backbone_compat import load_hubert_base

K,T,D=8,32,768
LABELS={"positive":0,"neutral":1,"negative":2}
ARCHIVES={"positive":"happy.tar.xz","neutral":"neutral.tar.xz","negative":"negative.tar.xz"}
MODEL_IDS={"yolo":"yolov8l.pt","dino":"facebook/dinov2-base","hubert":"facebook/hubert-base-ls960","global":"OpenGVLab/VideoMAEv2-Base"}
FAILURE_PREFIXES=("TRACKING_FAILED","DINO_FAILED","AUDIO_DECODE_FAILED","HUBERT_FAILED","GLOBAL_FAILED","TEMPORAL_ALIGNMENT_FAILED","INVALID_MASK","INVALID_SHAPE","INVALID_LABEL","NONFINITE_FEATURE","WRITE_FAILED")

def utc(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def safe(v):
    if isinstance(v,Path): return str(v)
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,np.ndarray): return v.tolist()
    raise TypeError(type(v).__name__)

def ids_for(cls):
    if cls=="positive": return [f"videoH{i:03d}" for i in range(1,202)]
    if cls=="neutral": return [f"videoN{i:03d}" for i in range(1,109) if i!=10]
    return [f"videoS{i:03d}" for i in range(1,101)]

def parse_id(value):
    value=value.removeprefix("video")
    if value[0].upper()=="H": return "positive",f"videoH{int(value[1:]):03d}"
    if value[0].upper()=="N": return "neutral",f"videoN{int(value[1:]):03d}"
    if value[0].upper()=="S": return "negative",f"videoS{int(value[1:]):03d}"
    raise ValueError(f"Unknown video ID {value}")

def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,default=safe)+"\n"); os.replace(tmp,path)

def valid_primitive(path: Path) -> bool:
    """Cheap, read-only resume validation; rejects partial/foreign HDF5 files."""
    required = {
        "local_features": (K,T,D), "local_time_mask": (K,T), "agent_mask": (K,),
        "audio_features": (T,D), "audio_time_mask": (T,), "global_scene": (D,),
        "physics_sync": (K,K), "gamma": (), "selected_agent_ids": (None,),
        "reference_timestamps": (T,), "reference_source_frames": (T,),
    }
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("protocol") != "VIBE-GECV-RECON-v1.0" or bool(handle.attrs.get("paper_faithful", True)):
                return False
            if int(handle.attrs["label"]) not in (0, 1, 2): return False
            for name, shape in required.items():
                if name not in handle: return False
                if shape != (None,) and tuple(handle[name].shape) != shape: return False
            return bool(np.asarray(handle["agent_mask"]).any()) and bool(np.asarray(handle["audio_time_mask"]).any())
    except (OSError, KeyError, ValueError):
        return False

def failure_category(message: str) -> str:
    return next((prefix for prefix in FAILURE_PREFIXES if prefix in message), "UNCLASSIFIED_FAILURE")

class Engine:
    def __init__(self,args):
        self.device=args.device; self.det_batch=args.detector_batch_size; self.dino_batch=args.dino_batch_size
        weights=args.yolo_weights
        if not weights.exists(): raise RuntimeError(f"YOLO weights missing: {weights}")
        self.yolo=YOLO(str(weights)).to(self.device)
        self.dino_processor=AutoImageProcessor.from_pretrained(MODEL_IDS["dino"])
        self.dino=Dinov2Model.from_pretrained(MODEL_IDS["dino"]).to(self.device).eval()
        self.hubert_processor=Wav2Vec2FeatureExtractor.from_pretrained(MODEL_IDS["hubert"])
        self.hubert, self.hubert_load_meta=load_hubert_base(MODEL_IDS["hubert"], self.device)
        self.global_processor=AutoImageProcessor.from_pretrained(MODEL_IDS["global"])
        cfg=AutoConfig.from_pretrained(MODEL_IDS["global"],trust_remote_code=True)
        self.global_model=AutoModel.from_pretrained(MODEL_IDS["global"],config=cfg,trust_remote_code=True).to(self.device).eval()

    def decode(self,path):
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened(): raise RuntimeError("DECODE_FAILED: OpenCV cannot open video")
        fps=float(cap.get(cv2.CAP_PROP_FPS)); fps=30. if fps<=0 else fps; step=max(1,round(fps/6))
        frames=[]; n=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if n%step==0: frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            n+=1
        cap.release()
        if not frames: raise RuntimeError("DECODE_FAILED: no sampled frames")
        return np.asarray(frames),{"fps":fps,"frame_count":n,"duration_seconds":(n-1)/fps,"sampled_frame_count":len(frames),"step":step}

    def tracks(self,frames):
        raw={}; conf={}
        results=self.yolo.track(source=list(frames),persist=True,tracker="bytetrack.yaml",classes=[0],conf=.5,verbose=False,device=self.device,stream=True,batch=self.det_batch)
        for index,r in enumerate(results):
            if r.boxes is None or r.boxes.id is None: continue
            for box,pid,score in zip(r.boxes.xywh.cpu().numpy(),r.boxes.id.cpu().numpy().astype(int),r.boxes.conf.cpu().numpy()):
                raw.setdefault(int(pid),[]).append((index,np.asarray([box[0]-box[2]/2,box[1]-box[3]/2,box[2],box[3]],np.float32)))
                conf.setdefault(int(pid),[]).append(float(score))
        if not raw: raise RuntimeError("TRACKING_FAILED: no ByteTrack persons")
        from scipy.ndimage import gaussian_filter1d
        smooth={}
        for pid,points in raw.items():
            if len(points)<5: smooth[pid]=points; continue
            boxes=np.asarray([p[1] for p in points]); filtered=np.zeros_like(boxes)
            for d in range(4): filtered[:,d]=gaussian_filter1d(boxes[:,d],sigma=5/3)
            smooth[pid]=[(p[0],filtered[i]) for i,p in enumerate(points)]
        return smooth,conf

    def local(self,frames,tracks,selected,sampled_times,target):
        crops=[]; owners=[]; h,w=frames.shape[1:3]
        for pid in selected:
            for fi,box in tracks[pid]:
                x,y,bw,bh=map(int,box); x=max(0,min(x,w-1)); y=max(0,min(y,h-1)); bw=max(1,min(bw,w-x)); bh=max(1,min(bh,h-y))
                crop=frames[fi,y:y+bh,x:x+bw]
                if crop.size: crops.append(crop); owners.append((pid,fi))
        if not crops: raise RuntimeError("DINO_FAILED: no valid crops")
        out=[]
        with torch.no_grad():
            for start in range(0,len(crops),self.dino_batch):
                inputs=self.dino_processor(crops[start:start+self.dino_batch],return_tensors="pt").to(self.device)
                out.append(self.dino(**inputs).last_hidden_state[:,0,:].cpu().numpy())
        out=np.concatenate(out).astype(np.float32); features={}; times={}
        for pid in selected:
            indices=[i for i,(owner,_) in enumerate(owners) if owner==pid]
            features[pid]=out[indices]; times[pid]=np.asarray([sampled_times[owners[i][1]] for i in indices])
        local,mask,agent=map_local_tracks(features,times,selected,target)
        return local,mask,agent,{"native_dino_shape":list(out.shape)}

    def audio(self,video,scratch,duration,target):
        wav=scratch/(video.stem+".wav")
        cmd=["ffmpeg","-y","-v","error","-i",str(video),"-ac","1","-ar","16000","-c:a","pcm_s16le",str(wav)]
        r=subprocess.run(cmd,text=True,capture_output=True)
        if r.returncode: raise RuntimeError("AUDIO_DECODE_FAILED: "+r.stderr.strip())
        y,sr=sf.read(str(wav),dtype="float32")
        if sr!=16000 or y.ndim!=1 or len(y)<1600: raise RuntimeError("AUDIO_DECODE_FAILED: invalid 16-kHz mono PCM")
        with torch.no_grad():
            inputs=self.hubert_processor(y,sampling_rate=16000,return_tensors="pt",padding=True)
            native=self.hubert(inputs.input_values.to(self.device)).last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
        times=(np.arange(len(native))+.5)*(duration/len(native)); feat,mask=bin_pool(native,times,target)
        if not mask.any(): raise RuntimeError("HUBERT_FAILED: no valid temporal bins")
        return feat,mask,{"native_hubert_shape":list(native.shape),"pcm_samples":int(len(y)),"hubert_batch_size":1,**self.hubert_load_meta}

    def global_scene(self,frames):
        if len(frames)<16: raise RuntimeError("GLOBAL_FAILED: fewer than 16 sampled frames")
        ids=np.linspace(0,len(frames)-1,16).astype(int); inputs=self.global_processor([frames[i] for i in ids],return_tensors="pt")
        inputs["pixel_values"]=inputs["pixel_values"].permute(0,2,1,3,4).to(self.device)
        with torch.no_grad(): out=self.global_model(**inputs)
        tensor=out if isinstance(out,torch.Tensor) else (out.last_hidden_state if hasattr(out,"last_hidden_state") else out[0])
        if tensor.shape[-1]!=D: raise RuntimeError(f"GLOBAL_FAILED: last dimension {tensor.shape[-1]}")
        scene=tensor.reshape(tensor.shape[0],-1,D).mean(1).squeeze(0).cpu().numpy().astype(np.float32)
        return scene,{"raw_output_shape":list(tensor.shape),"pooling":"mean over non-feature axes; scene calibration only"}

def validate(local,lmask,agent,audio,amask,scene,sync,label):
    if local.shape!=(K,T,D) or lmask.shape!=(K,T) or agent.shape!=(K,): raise RuntimeError("INVALID_SHAPE: local/mask")
    if audio.shape!=(T,D) or amask.shape!=(T,) or scene.shape!=(D,) or sync.shape!=(K,K): raise RuntimeError("INVALID_SHAPE: audio/global/synchrony")
    if label not in (0,1,2): raise RuntimeError("INVALID_LABEL")
    if not agent.any() or not amask.any() or not np.isfinite(local[lmask]).all() or not np.isfinite(audio[amask]).all() or not np.isfinite(scene).all() or not np.isfinite(sync).all(): raise RuntimeError("NONFINITE_FEATURE or INVALID_MASK")

def process_member(engine, member, cls, vid, tf, scratch, output, manifest, manifest_path):
    """Process exactly one archive member; errors are isolated in its manifest record."""
    record=manifest["records"].get(vid,{"source_id":vid,"class":cls,"label":LABELS[cls],"source_archive":str(tf.name),"status":"PENDING"})
    out=output/f"{vid}_recon_v1.h5"
    record.update(status="RUNNING",started_utc=utc()); manifest["records"][vid]=record; atomic_json(manifest_path,manifest)
    video=scratch/(vid+".mp4")
    try:
        source=tf.extractfile(member)
        if source is None: raise RuntimeError("WRITE_FAILED: archive member cannot be read")
        with video.open("wb") as dest:
            shutil.copyfileobj(source,dest,length=1024*1024)
        frames,meta=engine.decode(video); grid=shared_time_grid(meta["frame_count"],meta["fps"])
        tracks,conf=engine.tracks(frames); selection=select_k8_duration_confidence_id(tracks,conf); selected=selection["selected_track_ids"]
        sampled=np.arange(len(frames))*meta["step"]/meta["fps"]
        local,lmask,agent,dmeta=engine.local(frames,tracks,selected,sampled,grid["timestamps"])
        audio,amask,ameta=engine.audio(video,scratch,float(grid["duration_seconds"]),grid["timestamps"])
        scene,gmeta=engine.global_scene(frames); sync,pairs,gamma=shared_interval_synchrony(tracks,selected,tuple(frames.shape[1:3]))
        validate(local,lmask,agent,audio,amask,scene,sync,LABELS[cls])
        tmp=out.with_suffix(".tmp")
        with h5py.File(tmp,"w") as h:
            payload={
                "local_features":local,"local_time_mask":lmask,"agent_mask":agent,
                "audio_features":audio,"audio_time_mask":amask,"global_scene":scene,
                "physics_sync":sync,"gamma":np.asarray(gamma,dtype=np.float32),
                "reference_timestamps":grid["timestamps"],"reference_source_frames":grid["source_frames"],
                "selected_agent_ids":np.asarray(selected,dtype=np.int32),"synchrony_pair_mask":pairs,
            }
            for name,value in payload.items(): h.create_dataset(name,data=value)
            h.attrs.update({"label":LABELS[cls],"protocol":"VIBE-GECV-RECON-v1.0","paper_faithful":False,
                            "text_status":"OFF","l_sat_status":"DISABLED_ZERO","validity":"PASS",
                            "metadata_json":json.dumps({**meta,**dmeta,**ameta,**gmeta,"selection":selection,
                            "track_count":len(tracks),"synchrony":"shared physical consecutive intervals",
                            "models":MODEL_IDS,"software":{"torch":torch.__version__}})})
        os.replace(tmp,out)
        if not valid_primitive(out): raise RuntimeError("WRITE_FAILED: final HDF5 did not pass resume validation")
        record.update(status="PASS",finished_utc=utc(),output=str(out),gamma=gamma,selected_agent_ids=selected,
                      track_count=len(tracks),metadata={**meta,**dmeta,**ameta,**gmeta,"selection":selection})
    except Exception as exc:
        record.update(status="FAIL",finished_utc=utc(),failure=str(exc),failure_category=failure_category(str(exc)))
    finally:
        video.unlink(missing_ok=True); (scratch/(vid+".wav")).unlink(missing_ok=True)
        manifest["records"][vid]=record; atomic_json(manifest_path,manifest)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--transfer-root",type=Path,default=Path.home()/"VIBE_TRANSFER_FROM_FRIEND"); p.add_argument("--output",type=Path,default=REPO/"data/gecv_reconstruction_v1"); p.add_argument("--ids",nargs="*"); p.add_argument("--class",dest="classes",choices=["positive","neutral","negative","all"],default="all"); p.add_argument("--limit",type=int); p.add_argument("--resume",action="store_true"); p.add_argument("--workers",type=int,default=2); p.add_argument("--detector-batch-size",type=int,default=16); p.add_argument("--dino-batch-size",type=int,default=128); p.add_argument("--hubert-batch-size",type=int,default=1); p.add_argument("--global-batch-size",type=int,default=1); p.add_argument("--device",default="cuda"); p.add_argument("--yolo-weights",type=Path,default=None); args=p.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"): raise SystemExit("CUDA required; run from the H200 terminal")
    if importlib.util.find_spec("lap") is None: raise SystemExit("lap is required for ByteTrack; install it before extraction")
    if shutil.which("ffmpeg") is None: raise SystemExit("ffmpeg is required for deterministic PCM audio decoding")
    candidates=(REPO/"yolov8l.pt",args.transfer_root/"code_reference/yolov8l.pt")
    args.yolo_weights=args.yolo_weights or next((x for x in candidates if x.exists()),None)
    if args.yolo_weights is None: raise SystemExit("YOLO weights were not found; pass --yolo-weights PATH")
    wanted=[parse_id(x) for x in args.ids] if args.ids else [(c,v) for c in ARCHIVES if args.classes in ("all",c) for v in ids_for(c)]
    if args.limit: wanted=wanted[:args.limit]
    manifest_path=REPO/"data/manifests/gecv_reconstruction_extraction_manifest.json"; manifest=json.loads(manifest_path.read_text()) if args.resume and manifest_path.exists() else {"protocol":"VIBE-GECV-RECON-v1.0","paper_faithful":False,"records":{}}
    for cls in {cls for cls,_ in wanted}:
        archive=args.transfer_root/"raw"/ARCHIVES[cls]
        if not archive.is_file(): raise SystemExit(f"Source archive missing: {archive}")
    manifest["run_config"]={"workers":args.workers,"detector_batch_size":args.detector_batch_size,"dino_batch_size":args.dino_batch_size,"hubert_batch_size":args.hubert_batch_size,"global_batch_size":args.global_batch_size,"device":args.device,"updated_utc":utc()}
    engine=Engine(args); scratch=Path(tempfile.mkdtemp(prefix="gecv_recon_")); args.output.mkdir(parents=True,exist_ok=True)
    try:
      wanted_by_class={cls:[vid for item_cls,vid in wanted if item_cls==cls] for cls in ARCHIVES}
      for cls, ids in wanted_by_class.items():
        if not ids: continue
        archive=args.transfer_root/"raw"/ARCHIVES[cls]
        with tarfile.open(archive,"r:xz") as tf:
          members={Path(member.name).stem:member for member in tf.getmembers() if member.isfile() and member.name.lower().endswith(".mp4")}
          for vid in ids:
            out=args.output/f"{vid}_recon_v1.h5"; previous=manifest["records"].get(vid,{})
            if args.resume and previous.get("status") in ("PASS","SKIP") and valid_primitive(out):
              previous.update(status="SKIP",output=str(out),skipped_utc=utc()); manifest["records"][vid]=previous; atomic_json(manifest_path,manifest); continue
            member=members.get(vid)
            if member is None:
              manifest["records"][vid]={"source_id":vid,"class":cls,"label":LABELS[cls],"status":"FAIL","failure":"WRITE_FAILED: source member absent","failure_category":"WRITE_FAILED","finished_utc":utc()}; atomic_json(manifest_path,manifest); continue
            process_member(engine,member,cls,vid,tf,scratch,args.output,manifest,manifest_path)
    finally: shutil.rmtree(scratch,ignore_errors=True)
    counts={s:sum(r.get("status")==s for r in manifest["records"].values()) for s in ("PENDING","RUNNING","PASS","FAIL","SKIP")}
    categories={}
    for record in manifest["records"].values():
        if record.get("status")=="FAIL": categories[record.get("failure_category","UNCLASSIFIED_FAILURE")]=categories.get(record.get("failure_category","UNCLASSIFIED_FAILURE"),0)+1
    manifest["summary"]={"updated_utc":utc(),**counts,"failure_categories":categories}; atomic_json(manifest_path,manifest); print(json.dumps(manifest["summary"],indent=2))
if __name__=="__main__": main()
