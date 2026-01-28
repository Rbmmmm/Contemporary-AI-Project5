import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datasets import Custom_Collator, load_dataset
from configs import cfg

from utils import (
    save_metrics,
    plot_training_curves,
    evaluate_on_validation,
)

torch.use_deterministic_algorithms(False)


# ============================================================
# Main
# ============================================================
def main(cfg, exp_name=None):

    # ----------------------------
    # Reproducibility
    # ----------------------------
    seed_everything(cfg.seed, workers=True)

    # ========================================================
    # TEST ONLY MODE (checkpoint-safe)
    # ========================================================
    if cfg.test_only:
        print("=" * 60)
        print(">>> TEST ONLY MODE")
        print(f">>> Loading model from: {cfg.model_path}")
        print(f">>> Test output dir  : {cfg.test_dir}")
        print(f">>> Model type       : {cfg.model_type}")
        print("=" * 60)

        os.makedirs(cfg.test_dir, exist_ok=True)

        # ⚠️ 为稳定性，test-only 强制 num_workers = 0
        collator = Custom_Collator(cfg)
        test_loader = DataLoader(
            load_dataset(cfg, split="test"),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )


        from EmotionCLIP import EmotionCLIP
        model = EmotionCLIP.load_from_checkpoint(
            cfg.model_path, cfg=cfg
        )

        # ⚠️ test 必须用新的 Trainer
        test_trainer = Trainer(
            accelerator="gpu",
            devices=cfg.gpus,
            logger=False,
        )

        test_trainer.test(model, dataloaders=test_loader)
        print(">>> TEST ONLY FINISHED")
        return

    # ========================================================
    # TRAIN + EVAL MODE
    # ========================================================
    if exp_name is None:
        now = datetime.now().strftime("%b%d_%H-%M-%S")
        exp_name = f"{now}_{cfg.model_type}_lr-{cfg.lr}_bs-{cfg.batch_size}"

    exp_dir = os.path.join("results", exp_name)
    cfg.test_dir = exp_dir

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    split_dir = os.path.join(exp_dir, "splits")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    print("=" * 60)
    print(f"Experiment  : {exp_name}")
    print(f"Model type  : {cfg.model_type}")
    print(f"Output dir  : {exp_dir}")
    print("=" * 60)

    # ----------------------------
    # Train / Val split
    # ----------------------------
    with open(cfg.train_file, "r") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    train_lines, val_lines = train_test_split(
        all_lines,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )

    train_split_file = os.path.join(split_dir, "train.txt")
    val_split_file = os.path.join(split_dir, "val.txt")

    with open(train_split_file, "w") as f:
        f.write("\n".join(train_lines))
    with open(val_split_file, "w") as f:
        f.write("\n".join(val_lines))

    cfg.train_file = train_split_file
    cfg.val_file = val_split_file

    # ----------------------------
    # Dataset & Dataloader
    # ⚠️ 强烈建议 num_workers = 0（CLIP 在 collator 里跑 GPU）
    # ----------------------------
    collator = Custom_Collator(cfg)

    train_loader = DataLoader(
        load_dataset(cfg, "train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    val_loader = DataLoader(
        load_dataset(cfg, "val"),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    test_loader = DataLoader(
        load_dataset(cfg, "test"),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # ----------------------------
    # Model
    # ----------------------------

    from EmotionCLIP import create_model
    model = create_model(cfg)

    logger = CSVLogger(save_dir=exp_dir, name="", version="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best_acc={val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # ----------------------------
    # Train
    # ----------------------------
    print(">>> Training")
    trainer.fit(model, train_loader, val_loader)

    # ----------------------------
    # Save metrics & curves
    # ----------------------------
    print(">>> Saving metrics and plots")

    metrics = {
        "train_loss": model.train_loss_history,
        "val_acc": model.val_acc_history,
        "val_f1": model.val_f1_history,
    }

    save_metrics(exp_dir, metrics)

    plot_training_curves(
        exp_dir,
        train_loss=model.train_loss_history,
        val_acc=model.val_acc_history,
        val_f1=model.val_f1_history,
    )

    # ----------------------------
    # Reload BEST checkpoint (with prototypes)
    # ----------------------------
    best_ckpt = checkpoint_callback.best_model_path
    print(f">>> Best checkpoint: {best_ckpt}")

    from EmotionCLIP import EmotionCLIP
    best_model = EmotionCLIP.load_from_checkpoint(
            best_ckpt, cfg=cfg
    )

    # ----------------------------
    # Validation analysis
    # ----------------------------
    print(">>> Running validation analysis with best model")

    evaluate_on_validation(
        model=best_model,
        val_loader=val_loader,
        device=cfg.device,
        class_names=cfg.class_names,
        save_dir=exp_dir,
    )

    # ----------------------------
    # Test (NEW Trainer, checkpoint-safe)
    # ----------------------------
    print(">>> Testing")

    test_trainer = Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
        logger=False,
    )
    test_trainer.test(best_model, dataloaders=test_loader)

    print("=" * 60)
    print(f"Experiment finished: {exp_name}")
    print("=" * 60)


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Multimodal Emotion Classification Experiment"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Custom experiment name",
    )

    args = parser.parse_args()
    main(cfg, args.exp_name)
