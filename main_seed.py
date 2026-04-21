import numpy as np
import torch
import random
from sklearn import preprocessing
from model.base import Base
from utils.mlp_data.seed_feature import SEEDFeatureDataset
from utils.common_utils import _load_config, build_dataset, CustomDataset, setup_seed, load_seed_data, setup_device
from deeplearning.trainDG import Trainer

def build_training_components(cfg):
    # 模型
    dataset_name = cfg.get("dataset_name")
    name_key = str(dataset_name).lower()
    if name_key in {"seed3", "seed"}:
        num_of_class = cfg.get("seed3_num_of_class")
    elif name_key in {"seed4", "seediv", "seed-iv", "seed_iv"}:
        num_of_class = cfg.get("seed4_num_of_class")
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    base_params = {
        "num_of_class": num_of_class,
    }
    params = {
        "transfer_loss_type": cfg.get("transfer_loss_type"),
        "max_iter": cfg.get("max_iter"),
    }

    combined_params = {**base_params, **params}
    device, device_idx = setup_device(cfg)
    model = Base(**combined_params).to(device)

    # 优化器
    params = model.get_parameters()
    optimizer = torch.optim.RMSprop(
        params,
        lr=float(cfg.get("lr")),
        weight_decay=float(cfg.get("weight_decay")),
    )

    # 学习率scheduler
    if cfg.get("lr_scheduler"):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1.0 + cfg.get("lr_gamma") * float(x)) ** (-cfg.get("lr_decay")),
        )
    else:
        scheduler = None
    # 训练器
    trainer_params = {
        "lr_scheduler": scheduler,
        "k_per_trial": cfg.get("k_per_trial"),
        "n_epochs": cfg.get("n_epochs"),
        "transfer_loss_weight": cfg.get("transfer_loss_weight"),
        "early_stop": cfg.get("early_stop"),
        "tmp_saved_path": cfg.get("tmp_saved_path"),
        "log_interval": cfg.get("log_interval"),
        "dataset_name": cfg.get("dataset_name"),
        "transfer_loss_type": cfg.get("transfer_loss_type"),
        "seed": cfg.get("seed")
    }

    return model, optimizer, trainer_params


def _normalize_session(session):
    if session is None:
        return None
    if isinstance(session, str) and session.isdigit():
        return int(session)
    return session


def train(target, source_lists, cfg, source_ds, target_ds):
    model, optimizer, trainer_params = build_training_components(cfg)
    device = next(model.parameters()).device

    steps_per_epoch = cfg.get("steps_per_epoch", 1)
    session = _normalize_session(cfg.get("session"))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        k_per_trial=trainer_params["k_per_trial"],
        target_id=target,
        source_ids=source_lists,
        session=session,
        steps_per_epoch=steps_per_epoch,
        transfer_loss_weight=trainer_params["transfer_loss_weight"],
        lr_scheduler=trainer_params["lr_scheduler"],
        n_epochs=trainer_params["n_epochs"],
        early_stop=trainer_params["early_stop"],
        log_interval=trainer_params["log_interval"],
        tmp_saved_path=trainer_params["tmp_saved_path"],
        dataset_name=trainer_params["dataset_name"],
        transfer_loss_type=trainer_params["transfer_loss_type"],
        seed=trainer_params["seed"],
    )

    return trainer.train(
        source_ds=source_ds,
        target_ds=target_ds,
        debug_one_epoch=cfg.get("debug_one_epoch", False),
    )

def main(target, source_lists, cfg):
    # 1) 固定随机种子
    setup_seed(cfg.get("seed"))

    # 2) 准备数据（注意：target 不要写死 1）
    source_ds, target_ds = load_seed_data(cfg, target=target, source_lists=source_lists)

    print(f"\n[main] target={target} | #sources={len(source_lists)} | sources={source_lists}")

    best_acc = train(
        target=target,
        source_lists=source_lists,
        cfg=cfg,
        source_ds=source_ds,
        target_ds=target_ds,
    )

    return best_acc


if __name__ == "__main__":
    # 直接读取配置并加载数据
    cfg = _load_config(config_path = "config_seed.yaml")

    # 用来测试不同的迁移损失函数
    best_acc_mat = []
    # 准备数据
    sub_num = cfg.get("sub_num")
    for target in range(1, sub_num + 1):
    # for target in range(1, 2):
        source_lists = [i for i in range(1, sub_num + 1) if i != target]
        # 用 source_ids 做这一轮训练
        best_acc = main(target, source_lists, cfg)
        best_acc_mat.append(best_acc)
        print(f"target: {target}, best_acc: {best_acc}")

    mean = np.mean(best_acc_mat)
    std = np.std(best_acc_mat)

    for target, best_acc in enumerate(best_acc_mat):
        print(f"target: {target + 1}, best_acc: {best_acc:.6f}")
    print(f"all_best_acc: {mean:.4f} ± {std:.4f}")
