"""Training, validation, checkpoint selection, and provenance for VIBE-GECV.

Lifecycle: frozen split -> training batches -> validation each epoch -> select
the best validation macro-F1 checkpoint -> reload it -> one held-out test
evaluation -> structured provenance. Test metrics never select a checkpoint.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
import yaml
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import numpy as np

from model.vibe_model import VIBE_Transformer
from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn, create_group_split
from dataloading.reconstruction_split import load_split_files
from trainer.training_utils import CANONICAL_CLASS_NAMES, CANONICAL_LABEL_ORDER, WarmupCosineScheduler, compute_evaluation_metrics
from losses.loss_function import VIBE_Loss

class VIBETrainer:
    """Own one reproducible VIBE-GECV experiment from configuration to report."""
    def __init__(self, cfg, run_report_path=None):
        """Build loaders, model, loss, optimizer, schedule, and provenance state."""
        self.cfg = cfg
        self.seed = int(cfg.get('seed', 20260928))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup Directories
        self.log_dir = Path(cfg['log_dir'])
        self.ckpt_dir = Path(cfg['checkpoint_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Structured run logging is provenance-only.  It records observed
        # training/validation/test facts and does not alter model, data, split,
        # optimizer, scheduler, or checkpoint-selection behavior.
        self.run_report_path = Path(run_report_path) if run_report_path else self.log_dir / "run_metrics.json"
        self.run_report_path.parent.mkdir(parents=True, exist_ok=True)
        split_path = Path(cfg['split_manifest']) if cfg.get('split_manifest') else None
        split_sha256 = self._sha256(split_path) if split_path and split_path.is_file() else None
        source_corpus_hash = None
        if split_path and split_path.is_file():
            split_manifest = json.loads(split_path.read_text())
            source_corpus_hash = split_manifest.get('source_corpus_hash')
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        self.run_record = {
            'protocol': cfg.get('protocol', 'VIBE-GECV-RECON-v1.0'),
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'config': cfg,
            'epochs': [],
            'checkpoint_selection_metric': cfg.get('best_metric', 'macro_f1'),
            'storage_max_k': cfg.get('max_k', 8),
            'effective_k': cfg.get('effective_k', cfg.get('max_k', 8)),
            'provenance': {
                'config_path': cfg.get('_scientific_config_path'),
                'config_sha256': cfg.get('_scientific_config_sha256'),
                'split_manifest_path': str(split_path) if split_path else None,
                'split_sha256': split_sha256,
                'source_corpus_hash': source_corpus_hash,
                'gpu_name': gpu_name,
            },
        }
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.viz = None
        if cfg.get('enable_visualization', False):
            from trainer.vizualization import VIBE_DeepVisualizer
            self.viz = VIBE_DeepVisualizer(str(self.log_dir))
        
        print("Initializing Datasets...")
        
        # A frozen reconstruction manifest is the only source of truth for the
        # flat VIBE-GECV-RECON-v1.0 corpus.  Legacy directory splitting remains
        # available only for non-reconstruction configurations.
        if cfg.get('split_manifest'):
            split_files = load_split_files(Path(cfg['split_manifest']), Path(cfg['data_root']))
            train_files, val_files, test_files = split_files['train'], split_files['validation'], split_files['test']
        else:
            train_files, val_files = create_group_split(
                cfg['data_root'],
                val_ratio=cfg.get('val_ratio', 0.1)
            )
            test_source = Path(cfg['data_root']) / 'val'
            test_files = sorted(list(test_source.glob("*_vibe.h5")))
        
        # Extract dataset params
        max_k = cfg.get('max_k', 8)
        max_t = cfg.get('max_t', 32)
        num_workers = cfg.get('num_workers', 0)
        
        dataset_kwargs = {
            'max_k': max_k,
            'max_t': max_t,
            'schema_mode': cfg.get('dataset_schema', 'auto'),
            'reconstruction_gamma_source': cfg.get('reconstruction_gamma_source', 'stored'),
            'effective_k': cfg.get('effective_k'),
        }
        self.train_loader = DataLoader(
            VIBEDataset(train_files, **dataset_kwargs),
            batch_size=cfg['batch_size'], shuffle=True,
            collate_fn=vibe_collate_fn, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(self.seed)
        )
        self.val_loader = DataLoader(
            VIBEDataset(val_files, **dataset_kwargs),
            batch_size=cfg['batch_size'], shuffle=False, 
            collate_fn=vibe_collate_fn, num_workers=num_workers, pin_memory=True
        )
        
        # The manifest test membership is isolated from train/validation and is
        # only evaluated after best-validation checkpoint selection.
        self.test_loader = DataLoader(
            VIBEDataset(test_files, **dataset_kwargs),
            batch_size=cfg['batch_size'], shuffle=False,
            collate_fn=vibe_collate_fn, num_workers=num_workers
        )
        print(f"  > Test Set (Held Out): {len(test_files)} clips loaded.")

        # --- C. Init Model (V5.3 Global-Aware) ---
        print(f"Initializing VIBE (Latent: {cfg['latent_dim']})...")
        self.model = VIBE_Transformer(
            input_dim=cfg.get('input_dim', 768), 
            latent_dim=cfg['latent_dim'],
            num_classes=cfg.get('num_classes', 3),
            attention_heads=cfg.get('attention_heads', 8),
            transformer_depth=cfg.get('transformer_depth', 2),
            dropout=cfg.get('dropout'),
            drop_path=cfg.get('drop_path', 0.1),
            enable_context_injection=cfg.get('enable_context_injection', True),
            gamma_conditioning=cfg.get('gamma_conditioning', 'stored'),
            enable_cross_modal_attention=cfg.get('enable_cross_modal_attention', True),
            audio_input_mode=cfg.get('audio_input_mode', 'observed'),
            global_scene_conditioning=cfg.get('global_scene_conditioning', 'observed'),
        ).to(self.device)
        
        # --- D. Init Loss (V5.5 Improved) ---
        self.criterion = VIBE_Loss(
            lambda_sat=cfg.get('lambda_sat', 0.5),
            lambda_ortho=cfg.get('lambda_ortho', 0.1), # New Ortho Weight
            lambda_kl=cfg.get('lambda_kl', 1.0),
            enable_sat=cfg.get('enable_sat', True),
        ).to(self.device)
        self.reconstruction_mode = cfg.get('reconstruction_mode', False)
        
        # --- E. Optimizer & Scheduler ---
        if cfg.get('optimizer', 'adamw').lower() != 'adamw':
            raise ValueError("Only AdamW is implemented for this reconstruction configuration")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.01)
        )
        scheduler_name = cfg.get('scheduler', 'reduce_on_plateau')
        self.scheduler_name = scheduler_name
        if scheduler_name == 'warmup_cosine':
            self.scheduler = WarmupCosineScheduler(
                self.optimizer, base_lr=cfg['lr'], total_epochs=cfg['epochs'],
                warmup_epochs=cfg.get('warmup_epochs', 3), eta_min=cfg.get('eta_min', 0.0)
            )
        elif cfg.get('reconstruction_mode', False):
            raise ValueError("VIBE-GECV-RECON-v1.0 requires scheduler=warmup_cosine")
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=cfg.get('scheduler_patience', 3),
                factor=cfg.get('scheduler_factor', 0.1)
            )
        self.best_metric_name = cfg.get('best_metric', 'macro_f1')

    def run(self, epochs):
        """Train all epochs, select by validation macro-F1, then evaluate test.

        The test set is deliberately deferred until ``model_best.pth`` has
        been selected and reloaded. This protects held-out test data from
        influencing either training or checkpoint selection.
        """
        print(f"Starting VIBE Training on {self.device}")
        print(f"Checkpoints will be saved to: {self.ckpt_dir}")
        best_metric = float('-inf')
        best_epoch = None
        best_checkpoint_path = None
        
        mixup_alpha = self.cfg.get('mixup_alpha', 0.2)
        grad_clip = self.cfg.get('grad_clip', 1.0)
        save_threshold = self.cfg.get('save_threshold', 0.65)
        
        for epoch in range(1, epochs + 1):
            if self.scheduler_name == 'warmup_cosine':
                current_lr = self.scheduler.set_epoch(epoch)
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            self.model.train()
            total_loss = 0
            total_hce = 0
            total_kl = 0
            total_sat = 0
            total_ortho = 0
            batch_count = 0
            pbar = tqdm(self.train_loader, desc=f"Ep {epoch}")
            
            for batch in pbar:
                if batch is None: continue
                
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # Model Forward
                out = self.model.forward_with_gamma(
                    local_p=batch['local_p'], 
                    global_v=batch['global_v'],
                    env_feat=batch['env_feat'], 
                    audio_seq=batch['audio_seq'],
                    gamma=batch['gamma'],
                    mixup_alpha=mixup_alpha,
                    targets=batch['labels'],
                    agent_mask=batch.get('agent_mask'),
                    local_time_mask=batch.get('local_time_mask'),
                    audio_time_mask=batch.get('audio_time_mask'),
                    global_time_mask=batch.get('global_time_mask'),
                    reconstruction_mode=self.reconstruction_mode,
                )
                
                # Loss Calculation (Unpacking Ortho Loss now)
                loss, l_hce, l_kl, l_sat, l_ortho = self.criterion(
                    out, batch['labels'], batch
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_hce += l_hce.item()
                total_kl += l_kl.item()
                total_sat += l_sat.item()
                total_ortho += l_ortho.item()
                batch_count += 1
                
                # Monitor Disentanglement (Ortho) in realtime
                pbar.set_postfix({
                    'L': f"{loss.item():.3f}", 
                    'HCE': f"{l_hce.item():.3f}",
                    'KL': f"{l_kl.item():.3f}",
                    'Ort': f"{l_ortho.item():.3f}"
                })
            
            metrics = self.validate(epoch, self.val_loader, mode='Val')
            if self.scheduler_name != 'warmup_cosine':
                self.scheduler.step(metrics['acc'])
            
            if metrics['acc'] > save_threshold:
                ckpt_name = f"model_ep{epoch}_acc{metrics['acc']:.4f}_f1{metrics['f1']:.4f}.pth"
                torch.save(self.model.state_dict(), self.ckpt_dir / ckpt_name)
                print(f"   >>> Model Saved (Acc > {save_threshold:.0%}): {ckpt_name}")
            
            if self.best_metric_name not in metrics:
                raise KeyError(f"Best checkpoint metric is unavailable: {self.best_metric_name}")
            if metrics[self.best_metric_name] > best_metric:
                best_metric = metrics[self.best_metric_name]
                best_epoch = epoch
                best_checkpoint_path = self.ckpt_dir / "model_best.pth"
                torch.save(self.model.state_dict(), best_checkpoint_path)
                print(f"   >>> New Best Model Saved ({self.best_metric_name}: {best_metric:.4f})")

            # Persist the completed epoch before writing checkpoint_last. This
            # is structured provenance for future runs, not a training change.
            epoch_record = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'train_batches': batch_count,
                'train_loss': {
                    'total_mean': total_loss / max(batch_count, 1),
                    'hce_mean': total_hce / max(batch_count, 1),
                    'kl_mean': total_kl / max(batch_count, 1),
                    'sat_mean': total_sat / max(batch_count, 1),
                    'ortho_mean': total_ortho / max(batch_count, 1),
                },
                'validation': metrics,
                'best_epoch_so_far': best_epoch,
                'best_metric_so_far': best_metric,
                'best_checkpoint_path': str(best_checkpoint_path) if best_checkpoint_path else None,
                'best_checkpoint_sha256': self._sha256(best_checkpoint_path) if best_checkpoint_path else None,
            }
            self.run_record['epochs'].append(epoch_record)
            self._write_run_record()

            # ``checkpoint_last`` records the post-selection best metric and the
            # exact scheduler state used for this completed epoch.
            checkpoint = {
                'epoch': epoch, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'best_metric': best_metric,
                'best_metric_name': self.best_metric_name, 'scheduler': self.scheduler.state_dict(),
                'learning_rate': current_lr, 'config': self.cfg
            }
            torch.save(checkpoint, self.ckpt_dir / "checkpoint_last.pth")

        # Final test is deliberately after all best-validation checkpoint
        # decisions.  Test metrics are never used to select a checkpoint.
        print("\n\n" + "="*50)
        print("  FINAL EVALUATION ON HELD-OUT TEST SET")
        print("="*50)
        best_ckpt = torch.load(self.ckpt_dir / "model_best.pth")
        self.model.load_state_dict(best_ckpt)
        test_metrics = self.validate(epochs, self.test_loader, mode='Test')
        self.run_record.update({
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'training_loop_completed_epoch': epochs,
            'test_checkpoint_epoch': best_epoch,
            'test_checkpoint_path': str(self.ckpt_dir / "model_best.pth"),
            'test_checkpoint_sha256': self._sha256(self.ckpt_dir / "model_best.pth"),
            'best_epoch': best_epoch,
            'best_validation_metric': best_metric,
            'final_test_metrics': test_metrics,
        })
        self._write_run_record()
        self.writer.close()

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_run_record(self):
        self.run_report_path.write_text(json.dumps(self.run_record, indent=2, default=str) + '\n')

    @torch.no_grad()
    def validate(self, epoch, loader, mode='Val'):
        """Return canonical metrics for validation or post-selection test data.

        The displayed reporting epoch is not necessarily the checkpoint epoch;
        provenance stores the selected checkpoint epoch separately.
        """
        self.model.eval()
        preds, targets = [], []
        
        # Storage for Visualization
        rationale_list, text_list = [], []
        h_final_list = []
        z_aff_list, z_env_list = [], []
        gamma_list = []
        
        for batch in tqdm(loader, desc=f"Validating {mode}", leave=False):
            if batch is None: continue
            
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            out = self.model.forward_with_gamma(
                batch['local_p'], batch['global_v'],
                batch['env_feat'], batch['audio_seq'],
                batch['gamma'],
                agent_mask=batch.get('agent_mask'),
                local_time_mask=batch.get('local_time_mask'),
                audio_time_mask=batch.get('audio_time_mask'),
                global_time_mask=batch.get('global_time_mask'),
                reconstruction_mode=self.reconstruction_mode,
            )
            
            preds.extend(torch.argmax(out['logits'], dim=1).cpu().numpy())
            targets.extend(batch['labels'].cpu().numpy())
            
            # Subsample for Viz
            if len(h_final_list) * self.cfg['batch_size'] < 800:
                rationale_list.append(out['rationale'].cpu())
                if batch.get('text_anch') is not None:
                    text_list.append(batch['text_anch'].cpu())
                h_final_list.append(out['h_final'].cpu())
                
                # Reshape VIB outputs
                B, K, T, _ = batch['local_p'].shape
                z_aff_flat = out['vib_out']['z_aff']
                z_env_flat = out['vib_out']['z_env']
                
                z_aff_reshaped = z_aff_flat.view(B, K, T, -1)
                z_env_reshaped = z_env_flat.view(B, K, T, -1)
                
                z_aff_list.append(z_aff_reshaped.mean(dim=(1, 2)).cpu())
                z_env_list.append(z_env_reshaped.mean(dim=(1, 2)).cpu())
                gamma_list.append(batch['gamma'].cpu())

        if len(targets) == 0:
            return {'acc': 0.0, 'f1': 0.0, 'weighted_f1': 0.0, 'macro_f1': 0.0,
                    'macro_precision': 0.0, 'macro_recall': 0.0, 'per_class': {},
                    'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    'class_order': ['Positive', 'Neutral', 'Negative']}
        
        metrics = compute_evaluation_metrics(targets, preds)
        print("\n" + "="*50)
        print(f"  {mode} Report | Epoch {epoch}")
        print("="*50)
        print(classification_report(targets, preds, labels=CANONICAL_LABEL_ORDER, target_names=CANONICAL_CLASS_NAMES, zero_division=0))
        print(f"  > Accuracy             : {metrics['acc']:.4f}")
        print(f"  > F1 Score (Weighted)  : {metrics['weighted_f1']:.4f}")
        print(f"  > F1 Score (Macro)     : {metrics['macro_f1']:.4f}")
        print(f"  > Macro Precision      : {metrics['macro_precision']:.4f}")
        print(f"  > Macro Recall         : {metrics['macro_recall']:.4f}")
        print(f"  > Confusion Matrix     : {metrics['confusion_matrix']}")
        print("="*50 + "\n")
        self.writer.add_scalar(f'{mode}/Accuracy', metrics['acc'], epoch)
        self.writer.add_scalar(f'{mode}/Weighted_F1', metrics['weighted_f1'], epoch)
        self.writer.add_scalar(f'{mode}/Macro_F1', metrics['macro_f1'], epoch)
        self.writer.add_scalar(f'{mode}/Macro_Precision', metrics['macro_precision'], epoch)
        self.writer.add_scalar(f'{mode}/Macro_Recall', metrics['macro_recall'], epoch)
        
        # --- VISUALIZATION BLOCK ---
        if self.viz is not None and mode == 'Val' and len(rationale_list) > 0 and len(text_list) > 0:
            gammas = torch.cat(gamma_list).numpy()
            targets_viz = np.array(targets)[:len(gammas)]
            z_aff = torch.cat(z_aff_list).numpy()
            z_env = torch.cat(z_env_list).numpy()
            texts = torch.cat(text_list).numpy()
            rationales = torch.cat(rationale_list).numpy()
            
            # Metrics
            ebs = F.cosine_similarity(torch.tensor(z_aff), torch.tensor(z_env)).abs().mean().item()
            rc = F.cosine_similarity(torch.tensor(rationales), torch.tensor(texts)).mean().item()
            gamma_mean = gammas.mean().item()
            
            self.viz.update_history(epoch, ebs, rc, acc, gamma_mean)
            
            if epoch % 2 == 0 or epoch == 1:
                self.viz.plot_disentanglement_proof(z_aff, z_env, targets_viz, epoch)
                self.viz.plot_gamma_dynamics(gammas, targets_viz, epoch)
                self.viz.plot_3d_manifold(torch.cat(h_final_list).numpy(), targets_viz, epoch)
                self.viz.plot_cot_alignment(rationales, texts, epoch)
                self.viz.plot_confusion_matrix(np.array(targets), np.array(preds), epoch)
                self.viz.plot_metric_trends()

            self.writer.add_scalar(f'{mode}/EBS', ebs, epoch)
        
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VIBE Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--log-dir", type=str, default=None, help="Optional output-only override; does not change hyperparameters.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional output-only override; does not change hyperparameters.")
    parser.add_argument("--run-report", type=str, default=None, help="Optional structured JSON provenance output path.")
    args = parser.parse_args()

    # Load YAML Configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    config_path = Path(args.config)
    config['_scientific_config_path'] = str(config_path)
    config['_scientific_config_sha256'] = VIBETrainer._sha256(config_path)

    if args.log_dir is not None:
        config['log_dir'] = args.log_dir
    if args.checkpoint_dir is not None:
        config['checkpoint_dir'] = args.checkpoint_dir

    trainer = VIBETrainer(config, run_report_path=args.run_report)
    trainer.run(epochs=config['epochs'])
