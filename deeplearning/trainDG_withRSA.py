
import os
import numpy as np
import torch

from utils.common_utils import build_aligned_batches, build_group_cache


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        k_per_trial,
        target_id,
        source_ids,
        session=None,
        steps_per_epoch=1,
        transfer_loss_weight=1.0,
        rdm_loss_weight=1.0,
        lr_scheduler=None,
        n_epochs=1,
        early_stop=0,
        log_interval=1,
        tmp_saved_path=None,
        dataset_name=None,
        transfer_loss_type=None,
        rng=None,
        seed=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.k_per_trial = k_per_trial
        self.target_id = target_id
        self.source_ids = source_ids
        self.session = session
        self.steps_per_epoch = steps_per_epoch
        self.transfer_loss_weight = transfer_loss_weight
        self.rdm_loss_weight = rdm_loss_weight
        self.lr_scheduler = lr_scheduler
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.log_interval = log_interval
        self.tmp_saved_path = tmp_saved_path
        self.dataset_name = dataset_name
        self.transfer_loss_type = transfer_loss_type
        if rng is None and seed is not None:
            rng = np.random.default_rng(seed)
        self.rng = rng

    @staticmethod
    def _labels_to_index(labels):
        if labels.ndim > 1 and labels.size(1) > 1:
            return torch.argmax(labels, dim=1)
        return labels

    def train_one_epoch(self, source_ds, target_ds, source_cache=None, target_cache=None):
        self.model.train()
        _ = target_ds  # DG setting: target data is not used during training.

        loss_clf = AverageMeter()
        loss_transfer = AverageMeter()
        loss_rdm = AverageMeter()
        correct = 0
        total = 0

        for _ in range(self.steps_per_epoch):
            total_cls = 0.0
            total_transfer = 0.0
            total_rdm = 0.0
            num_pairs = 0

            self.optimizer.zero_grad()
            if self.source_ids:
                for idx, src_id in enumerate(self.source_ids):
                    tgt_id = self.source_ids[(idx + 1) % len(self.source_ids)]
                    source_batches, pair_batch, _ = build_aligned_batches(
                        source_ds=source_ds,
                        target_ds=source_ds,
                        k_per_trial=self.k_per_trial,
                        target_id=tgt_id,
                        source_ids=[src_id],
                        session=self.session,
                        rng=self.rng,
                        source_cache=source_cache,
                        target_cache=source_cache,
                    )

                    tgt_data, _ = pair_batch
                    tgt_data = tgt_data.to(self.device)

                    for _, (src_data, src_label) in source_batches.items():
                        src_data = src_data.to(self.device)
                        src_label = self._labels_to_index(src_label).to(self.device)

                        cls_loss, transfer_loss, rdm_loss = self.model(
                            src_data, tgt_data, src_label
                        )
                        total_cls = total_cls + cls_loss
                        total_transfer = total_transfer + transfer_loss
                        total_rdm = total_rdm + rdm_loss
                        num_pairs += 1
                        if hasattr(self.model, "feature_extractor") and hasattr(self.model, "classifier"):
                            with torch.no_grad():
                                feature = self.model.feature_extractor(src_data)
                                logits = self.model.classifier(feature)
                                preds = torch.argmax(logits, dim=1)
                                correct += (preds == src_label).sum().item()
                                total += src_label.numel()

            if num_pairs > 0:
                total_cls = total_cls / num_pairs
                total_transfer = total_transfer / num_pairs
                total_rdm = total_rdm / num_pairs

            loss = (
                total_cls
                + self.transfer_loss_weight * total_transfer
                + self.rdm_loss_weight * total_rdm
            )
            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            loss_clf.update(total_cls.item())
            loss_transfer.update(total_transfer.item())
            loss_rdm.update(total_rdm.item())

        train_acc = correct / max(1, total) if total else 0.0
        return loss_clf.avg, loss_transfer.avg, loss_rdm.avg, train_acc


    def _get_log_path(self):
        if not self.tmp_saved_path:
            return None
        dataset_name = "" if self.dataset_name is None else str(self.dataset_name)
        session = "" if self.session is None else str(self.session)
        target = "" if self.target_id is None else str(self.target_id)
        log_dir = os.path.join(
            self.tmp_saved_path,
            dataset_name,
            f"{session}_{target}",
        )
        os.makedirs(log_dir, exist_ok=True)
        loss_type = "" if self.transfer_loss_type is None else str(self.transfer_loss_type)
        return os.path.join(log_dir, loss_type + "train_log.txt")

    def _log_epoch(self, epoch, loss_clf, loss_transfer, loss_rdm, acc, train_acc):
        log_path = self._get_log_path()
        if not log_path:
            return
        line = (
            f"[epoch {epoch:04d}] "
            f"loss_clf={loss_clf:.6f} "
            f"loss_transfer={loss_transfer:.6f} "
            f"loss_rdm={loss_rdm:.6f} "
            f"train_acc={train_acc:.4f} "
            f"acc={acc:.4f}\n"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def evaluate(self, dataset):
        self.model.eval()
        with torch.no_grad():
            x = dataset.data().to(next(self.model.parameters()).device)
            y = self._labels_to_index(dataset.label()).to(x.device)
            preds = torch.from_numpy(self.model.predict(x)).to(y.device)
            correct = (preds == y).sum().item()
        return correct / max(1, y.numel())

    def train(self, source_ds, target_ds, debug_one_epoch=False):
        best_acc = 0.0
        best_epoch = -1
        bad_epochs = 0
        source_cache = build_group_cache(source_ds)
        target_cache = build_group_cache(target_ds)

        if debug_one_epoch:
            loss_clf, loss_transfer, loss_rdm, train_acc = self.train_one_epoch(
                source_ds=source_ds,
                target_ds=target_ds,
                source_cache=source_cache,
                target_cache=target_cache,
            )
            acc = self.evaluate(target_ds)
            print(
                f"[debug_one_epoch] loss_clf={loss_clf:.6f} "
                f"loss_transfer={loss_transfer:.6f} "
                f"loss_rdm={loss_rdm:.6f} "
                f"train_acc={train_acc:.4f} "
                f"acc={acc:.4f}"
            )
            return acc

        for epoch in range(1, self.n_epochs + 1):
            loss_clf, loss_transfer, loss_rdm, train_acc = self.train_one_epoch(
                source_ds=source_ds,
                target_ds=target_ds,
                source_cache=source_cache,
                target_cache=target_cache,
            )

            # 做每一个epoch结束时的操作， 暂时只针对daan操作
            if hasattr(self.model, "epoch_based_processing"):
                self.model.epoch_based_processing(self.steps_per_epoch)

            acc = self.evaluate(target_ds)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            if self.log_interval and epoch % self.log_interval == 0:
                print(
                    f"[epoch {epoch:04d}] "
                    f"loss_clf={loss_clf:.6f} "
                    f"loss_transfer={loss_transfer:.6f} "
                    f"loss_rdm={loss_rdm:.6f} "
                    f"train_acc={train_acc:.4f} "
                    f"acc={acc:.4f}"
                )
                self._log_epoch(
                    epoch=epoch,
                    loss_clf=loss_clf,
                    loss_transfer=loss_transfer,
                    loss_rdm=loss_rdm,
                    train_acc=train_acc,
                    acc=acc,
                )

            if self.early_stop and bad_epochs >= self.early_stop:
                print(f"[early_stop] no improvement for {self.early_stop} epochs.")
                break

        print(f"[best] epoch={best_epoch} acc={best_acc:.4f}")
        return best_acc
