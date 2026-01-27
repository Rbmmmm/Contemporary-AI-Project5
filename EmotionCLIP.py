import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os

from models import Adapter


# ============================================================
# SimCLR Projection Head
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# NT-Xent Loss
# ============================================================
def nt_xent_loss(image_z, text_z, temperature=0.07):
    image_z = F.normalize(image_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)

    sim = image_z @ text_z.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)

    loss_i2t = F.cross_entropy(sim, labels)
    loss_t2i = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


# ============================================================
# EmotionCLIP (Checkpoint-safe)
# ============================================================
class EmotionCLIP(pl.LightningModule):
    """
    关键点：
    1) class_prototypes 是 register_buffer → 会随 checkpoint 保存/加载
    2) prototype 在每个 train epoch end 用训练集全量重算并写入 buffer
       → 保证“最优 ckpt 的 prototype”和“最优 ckpt 的参数”一致
    3) test 时绝不依赖运行期 list 缓存
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ----------------------------
        # Metrics
        # ----------------------------
        self.acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.num_classes
        )
        self.f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=cfg.num_classes,
            average="macro",
        )

        # ----------------------------
        # Feature mapping
        # ----------------------------
        self.image_map = nn.Linear(cfg.image_dim, cfg.map_dim)
        self.text_map = nn.Linear(cfg.text_dim, cfg.map_dim)

        self.img_adapter = Adapter(cfg.map_dim)
        self.txt_adapter = Adapter(cfg.map_dim)

        self.image_proj = ProjectionHead(cfg.map_dim, cfg.map_dim)
        self.text_proj = ProjectionHead(cfg.map_dim, cfg.map_dim)

        self.ce_loss = nn.CrossEntropyLoss()

        # ----------------------------
        # Prototype buffers (saved in checkpoint)
        # ----------------------------
        self.register_buffer(
            "class_prototypes",
            torch.zeros(cfg.num_classes, cfg.map_dim),
        )
        self.register_buffer(
            "_prototype_initialized",
            torch.tensor(False),
        )

        # ----------------------------
        # History (optional)
        # ----------------------------
        self.train_loss_history = []
        self.val_acc_history = []
        self.val_f1_history = []
        self._epoch_train_losses = []

    # ========================================================
    # Prototype builder (text / image / multimodal)
    # ========================================================
    def _build_prototypes(self, v_feats, t_feats, labels):
        """
        v_feats: (N, D) image embeddings
        t_feats: (N, D) text embeddings
        labels : (N,)
        return : (C, D) normalized prototypes
        """
        prototypes = []
        for k in range(self.cfg.num_classes):
            mask = labels == k
            if not mask.any():
                proto = torch.zeros(v_feats.size(1), device=v_feats.device)
            else:
                if self.cfg.prototype_type == "text":
                    proto = t_feats[mask].mean(dim=0)
                elif self.cfg.prototype_type == "image":
                    proto = v_feats[mask].mean(dim=0)
                elif self.cfg.prototype_type == "multimodal":
                    fused = F.normalize(v_feats[mask] + t_feats[mask], dim=-1)
                    proto = fused.mean(dim=0)
                else:
                    raise ValueError(f"Unknown prototype_type: {self.cfg.prototype_type}")
            prototypes.append(proto)

        return F.normalize(torch.stack(prototypes, dim=0), dim=-1)

    # ========================================================
    # Encode helpers (no side effects)
    # ========================================================
    def _encode(self, image_features, text_features):
        v = self.image_map(image_features)
        t = self.text_map(text_features)

        v = self.cfg.ratio * self.img_adapter(v) + (1 - self.cfg.ratio) * v
        t = self.cfg.ratio * self.txt_adapter(t) + (1 - self.cfg.ratio) * t

        v_z = self.image_proj(v)
        t_z = self.text_proj(t)
        return v_z, t_z

    # ========================================================
    # Forward
    # ========================================================
    def forward(self, image_features, text_features, labels=None):
        v_z, t_z = self._encode(image_features, text_features)

        # contrastive
        loss_con = nt_xent_loss(v_z, t_z, temperature=self.cfg.temperature)

        # training/val
        if labels is not None:
            # batch-level prototypes for training objective (fast, stable)
            batch_proto = self._build_prototypes(v_z, t_z, labels)

            v_cls = F.normalize(v_z, dim=-1)
            logits = v_cls @ batch_proto.T
            loss_cls = self.ce_loss(logits, labels)

            loss = loss_cls
            if self.cfg.use_contrastive:
                loss = loss + self.cfg.contrastive_weight * loss_con

            preds = logits.argmax(dim=1)
            return {"logits": logits, "preds": preds, "loss": loss}

        # test/inference: must use checkpoint prototypes
        if not bool(self._prototype_initialized.item()):
            raise RuntimeError(
                "class_prototypes not initialized in checkpoint. "
                "This means prototypes were not computed/saved during training. "
                "Please train with the checkpoint-safe EmotionCLIP and save ckpt again."
            )

        v_cls = F.normalize(v_z, dim=-1)
        logits = v_cls @ F.normalize(self.class_prototypes, dim=-1).T
        preds = logits.argmax(dim=1)
        return {"logits": logits, "preds": preds}

    # ========================================================
    # Training
    # ========================================================
    def training_step(self, batch, batch_idx):
        out = self(batch["image_features"], batch["text_features"], batch["labels"])
        loss = out["loss"]
        preds = out["preds"]

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", self.acc(preds, batch["labels"]), on_epoch=True)

        self._epoch_train_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        # record train loss history (optional)
        if self._epoch_train_losses:
            avg_loss = torch.stack(self._epoch_train_losses).mean().item()
            self.train_loss_history.append(avg_loss)
            self._epoch_train_losses.clear()

        # 🔥 checkpoint-safe: recompute prototypes from FULL training set for current weights
        self._recompute_and_store_train_prototypes()

    # ========================================================
    # Recompute prototypes from train dataloader (FULL pass)
    # ========================================================
    @torch.no_grad()
    def _recompute_and_store_train_prototypes(self):
        """
        使用训练集全量重算 prototype，并写入 register_buffer。
        这会随 checkpoint 保存，保证 best ckpt 的 prototype 与其权重一致。
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        train_loader = trainer.train_dataloader
        if train_loader is None:
            return

        was_training = self.training
        self.eval()

        device = self.device
        all_v = []
        all_t = []
        all_y = []

        for batch in train_loader:
            # batch 已由 collator 放到 GPU（你的 collator 返回的是 cuda tensor）
            v_z, t_z = self._encode(batch["image_features"], batch["text_features"])
            all_v.append(v_z.float().detach())
            all_t.append(t_z.float().detach())
            all_y.append(batch["labels"].detach())

        v_feats = torch.cat(all_v, dim=0).to(device)
        t_feats = torch.cat(all_t, dim=0).to(device)
        labels = torch.cat(all_y, dim=0).to(device)

        prototypes = self._build_prototypes(v_feats, t_feats, labels)
        self.class_prototypes.copy_(prototypes)
        self._prototype_initialized.fill_(True)

        if was_training:
            self.train()

    # ========================================================
    # Validation
    # ========================================================
    def validation_step(self, batch, batch_idx):
        out = self(batch["image_features"], batch["text_features"], batch["labels"])
        self.log("val/loss", out["loss"], on_epoch=True, prog_bar=True)
        self.log("val/acc", self.acc(out["preds"], batch["labels"]), on_epoch=True, prog_bar=True)
        self.log("val/f1", self.f1(out["preds"], batch["labels"]), on_epoch=True)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val/acc" in metrics:
            self.val_acc_history.append(metrics["val/acc"].item())
        if "val/f1" in metrics:
            self.val_f1_history.append(metrics["val/f1"].item())

    # ========================================================
    # Optimizer
    # ========================================================
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    # ========================================================
    # Test
    # ========================================================
    def on_test_start(self):
        self.test_results = []

    def test_step(self, batch, batch_idx):
        out = self(batch["image_features"], batch["text_features"], labels=None)
        for guid, pred in zip(batch["guids"], out["preds"].tolist()):
            self.test_results.append((guid, pred))

    def on_test_end(self):
        os.makedirs(self.cfg.test_dir, exist_ok=True)
        output_path = os.path.join(self.cfg.test_dir, "test_with_label.txt")
        with open(output_path, "w") as f:
            for guid, pred in self.test_results:
                f.write(f"{guid},{self.cfg.class_names[pred]}\n")
        print(f">>> test_with_label.txt saved to: {output_path}")


# ============================================================
def create_model(cfg):
    return EmotionCLIP(cfg)
