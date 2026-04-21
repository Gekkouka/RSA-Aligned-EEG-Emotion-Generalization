import numpy as np
import torch


def _labels_to_ids(labels):
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    if labels_np.ndim > 1:
        if labels_np.shape[1] > 1:
            return labels_np.argmax(axis=1)
        return labels_np.reshape(-1).astype(int)
    return labels_np.astype(int)


def _subject_mask(groups, subject_id, session=None):
    if isinstance(groups, torch.Tensor):
        g_np = groups.detach().cpu().numpy()
    else:
        g_np = np.asarray(groups)

    if session is None:
        return g_np[:, 1] == subject_id
    return (g_np[:, 0] == session) & (g_np[:, 1] == subject_id)


def build_label_batches(
    source_ds,
    target_ds,
    k_per_class,
    target_id,
    source_ids,
    num_of_class,
    session=None,
    rng=None,
    replace=False,
):
    """
    Sample k_per_class per label from each subject independently (no alignment).
    Returns source_batches, target_batch, and indices_by_subject.
    """
    if rng is None:
        rng = np.random.default_rng()

    def sample_subject(ds, subject_id):
        data = ds.data()
        label = ds.label()
        group = ds.group()

        mask = _subject_mask(group, subject_id, session=session)
        if isinstance(data, torch.Tensor):
            idx_pool = torch.nonzero(torch.as_tensor(mask), as_tuple=False).squeeze(1)
            data_pool = data.index_select(0, idx_pool)
            label_pool = label.index_select(0, idx_pool)
            label_ids = _labels_to_ids(label_pool)
        else:
            idx_pool = np.where(mask)[0]
            data_pool = np.asarray(data)[idx_pool]
            label_pool = np.asarray(label)[idx_pool]
            label_ids = _labels_to_ids(label_pool)

        picked_indices = {}
        picked_data = []
        picked_label = []

        for class_id in range(int(num_of_class)):
            class_mask = label_ids == class_id
            class_pool = idx_pool[class_mask]
            if class_pool.size < k_per_class and not replace:
                raise ValueError(
                    f"subject {subject_id} class {class_id} has only {class_pool.size} samples, "
                    f"need {k_per_class} (replace={replace})."
                )
            picked = rng.choice(class_pool, size=int(k_per_class), replace=replace)
            picked_indices[class_id] = picked.astype(np.int64)

            if isinstance(data, torch.Tensor):
                idx_t = torch.as_tensor(picked, dtype=torch.long, device=data.device)
                picked_data.append(data.index_select(0, idx_t))
                picked_label.append(label.index_select(0, idx_t))
            else:
                picked_data.append(np.asarray(data)[picked])
                picked_label.append(np.asarray(label)[picked])

        if isinstance(data, torch.Tensor):
            data_out = torch.cat(picked_data, dim=0)
            label_out = torch.cat(picked_label, dim=0)
        else:
            data_out = np.concatenate(picked_data, axis=0)
            label_out = np.concatenate(picked_label, axis=0)

        return (data_out, label_out), picked_indices

    target_batch, target_idx = sample_subject(target_ds, target_id)

    source_batches = {}
    source_idx = {}
    for sid in source_ids:
        batch, picked = sample_subject(source_ds, sid)
        source_batches[sid] = batch
        source_idx[sid] = picked

    indices_by_subject = {"target": target_idx, "source": source_idx}
    return source_batches, target_batch, indices_by_subject
