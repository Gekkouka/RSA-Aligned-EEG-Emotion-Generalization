import numpy as np
import torch
import os
import random
import yaml
from sklearn import preprocessing
from torch.utils.data import TensorDataset

from utils.mlp_data.deap import DEAPDataset
from utils.mlp_data.seed_feature import SEEDFeatureDataset
from utils.mlp_data.seediv_feature import SEEDIVFeatureDataset


def _load_config(config_path: str = "config_seed_rsaPlus.yaml") -> dict:
    # 从 YAML 配置文件读取参数
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_device(cfg, key: str = "device", default: str = "auto", verbose: bool = True):
    """
    解析配置中的 device 字段，返回 torch.device。
    支持：'auto' / 'cuda' / 'gpu' / 'cpu' / 'cuda:1' 等。
    """
    dev = cfg.get(key, default)

    if dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev in ("cuda", "gpu"):
        device = torch.device("cuda")
    else:
        device = torch.device(dev)  # 支持 'cpu'、'cuda:1' 等

    idx = None
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()

    if verbose:
        print(f"[device] cfg.{key} = {dev!r} -> selected = {device}")
        if device.type == "cuda":
            print(f"[cuda] current_device = {idx}")
            print(f"[cuda] name = {torch.cuda.get_device_name(idx)}")
            print(f"[cuda] visible_devices = {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

    return device, idx

def build_dataset(cfg):
    # 根据 dataset_name 选择对应 Dataset，并全参数传递
    dataset_name = cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("dataset_name is missing in config.")

    name_key = str(dataset_name).lower()
    feature = cfg.get("feature")
    channels = cfg.get("channels")
    subjects = cfg.get("subjects")
    session = cfg.get("session")

    if name_key in {"seed3", "seed"}:
        root_path = cfg.get("seed3_path")
        dataset = SEEDFeatureDataset(
            root_path=root_path,
            feature=feature,
            channels=channels,
            subjects=subjects,
            session=session,
        )
    elif name_key in {"seed4", "seediv", "seed-iv", "seed_iv"}:
        root_path = cfg.get("seed4_path")
        dataset = SEEDIVFeatureDataset(
            root_path=root_path,
            feature=feature,
            channels=channels,
            subjects=subjects,
            session=session,
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    return dataset


def setup_seed(seed):  ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_seed_data(cfg, target, source_lists=None):
    dataset = build_dataset(cfg)
    EEG, Label, Group = dataset.data()
    EEG = EEG.reshape(-1, 310)
    tGroup = Group[:, 2] - 1  # 影片的group
    sGroup = Group[:, 1]

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(sGroup):
        EEG[sGroup == i] = min_max_scaler.fit_transform(EEG[sGroup == i])

    dataset_name = cfg.get("dataset_name")
    name_key = str(dataset_name).lower()
    if name_key in {"seed3", "seed"}:
        num_of_class = 3
        Label += 1
    elif name_key in {"seed4", "seediv", "seed-iv", "seed_iv"}:
        num_of_class = 4
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    one_hot_mat = np.eye(num_of_class, dtype="float32")[Label]

    # 源域数据，如果只有几个受试者
    if source_lists is None:
        raise ValueError("source_lists 不能为 None，请传入源域被试列表，例如 [2,3,4,5]")
    source_lists = np.array(source_lists)

    # 获得目标域数据
    target_mask = sGroup == target
    target_features = torch.from_numpy(EEG[target_mask]).type(torch.Tensor)
    target_labels = torch.from_numpy(one_hot_mat[target_mask])
    target_group = torch.from_numpy(Group[target_mask])
    torch_dataset_target = CustomDataset(target_features, target_labels, target_group)

    # 获得源域数据
    source_mask = np.in1d(sGroup, source_lists)
    source_features = torch.from_numpy(EEG[source_mask]).type(torch.Tensor)
    source_labels = torch.from_numpy(one_hot_mat[source_mask])
    source_group = torch.from_numpy(Group[source_mask])
    torch_dataset_source = CustomDataset(source_features, source_labels, source_group)

    return torch_dataset_source, torch_dataset_target


def _normalize_selected_label(selected):
    LABEL_KEYS = ("v", "a", "d", "l")
    LABEL_INDEX = {"v": 0, "a": 1, "d": 2, "l": 3}

    if isinstance(selected, (list, tuple)):
        if len(selected) != 1:
            raise ValueError("Only one label key is supported.")
        token = str(selected[0]).strip().lower()
    else:
        token = str(selected).strip().lower()

    if token not in LABEL_INDEX:
        raise ValueError(f"Unknown label key: {token}. Use {LABEL_KEYS}.")
    return token


def discretize_labels_to_onehot(labels, selected, threshold=5):
    LABEL_KEYS = ("v", "a", "d", "l")
    LABEL_INDEX = {"v": 0, "a": 1, "d": 2, "l": 3}

    labels = np.asarray(labels)
    sel_key = _normalize_selected_label(selected)
    sel_idx = LABEL_INDEX[sel_key]
    sel_vals = labels[:, sel_idx]
    cls = (sel_vals >= float(threshold)).astype(int)
    return np.eye(2, dtype=np.int64)[cls]

def load_deap_data(
    root,
    feature_name="de",
    window_sec=1,
    step_sec=1,
    selected_label="v",
    threshold=5,
    add_session=True,
):
    ds = DEAPDataset(
        root_path=root,
        feature_name=feature_name,
        window_sec=window_sec,
        step_sec=step_sec,
    )
    dataset = ds.get_dataset()

    data = dataset["data"]
    labels = discretize_labels_to_onehot(
        dataset["labels"], selected_label, threshold=threshold
    )
    groups = dataset["groups"]
    if add_session:
        ones = np.ones((groups.shape[0], 1), dtype=groups.dtype)
        groups = np.column_stack((ones, groups))

    return data, labels, groups

def build_aligned_batches(
        source_ds,
        target_ds,
        k_per_trial,
        target_id,
        source_ids,
        session=None,
        rng=None,
        source_cache=None,
        target_cache=None,
):
    if rng is None:
        rng = np.random.default_rng()
    if target_cache is None:
        target_cache = build_group_cache(target_ds)
    if source_cache is None:
        source_cache = build_group_cache(source_ds)

    if session is not None:
        trials = target_cache["trials_by_subject_session"].get((session, target_id), [])
    else:
        trials = target_cache["trials_by_subject"].get(target_id, [])
    trial_to_win = {}
    for trial_id in trials:
        if session is not None:
            win_ids = target_cache["wins_by_trial_session"].get(
                (session, target_id, trial_id), []
            )
        else:
            win_ids = target_cache["wins_by_trial"].get((target_id, trial_id), [])
        win_ids = np.array(win_ids, dtype=np.int64)
        if win_ids.size < k_per_trial:
            raise ValueError(f"trial {trial_id} has only {win_ids.size} windows.")
        picked = rng.choice(win_ids, size=k_per_trial, replace=False)
        trial_to_win[trial_id] = sorted(picked.tolist())

    def collect_indices(ds, subject_id):
        indices = []
        for trial_id in trials:
            for win_id in trial_to_win[trial_id]:
                if session is not None:
                    idx = source_cache["index_by_key"].get(
                        (session, subject_id, trial_id, win_id)
                    )
                else:
                    idx = source_cache["index_by_subject_trial_win"].get(
                        (subject_id, trial_id, win_id)
                    )
                if idx is None:
                    raise ValueError(
                        f"Missing trial={trial_id}, win={win_id} for subject={subject_id}."
                    )
                indices.append(int(idx))
        return np.array(indices, dtype=np.int64)

    target_idx = collect_indices(target_ds, target_id)

    def slice_ds(ds, indices):
        data = ds.data()
        label = ds.label()
        if isinstance(data, torch.Tensor):
            idx_t = torch.as_tensor(indices, dtype=torch.long)
            return data.index_select(0, idx_t), label.index_select(0, idx_t)
        return data[indices], label[indices]

    t_data, t_label = slice_ds(target_ds, target_idx)
    target_batch = (t_data, t_label)

    source_batches = {}
    for sid in source_ids:
        sid_idx = collect_indices(source_ds, sid)
        s_data, s_label = slice_ds(source_ds, sid_idx)
        source_batches[sid] = (s_data, s_label)

    return source_batches, target_batch, trial_to_win


def build_group_cache(ds):
    g = ds.group()
    if isinstance(g, torch.Tensor):
        g_np = g.cpu().numpy()
    else:
        g_np = np.asarray(g)

    index_by_key = {}
    index_by_subject_trial_win = {}
    wins_by_trial_session = {}
    wins_by_trial = {}
    trials_by_subject_session = {}
    trials_by_subject = {}

    for idx, row in enumerate(g_np):
        session = int(row[0])
        subject = int(row[1])
        trial = int(row[2])
        win = int(row[3])

        index_by_key[(session, subject, trial, win)] = idx
        if (subject, trial, win) not in index_by_subject_trial_win:
            index_by_subject_trial_win[(subject, trial, win)] = idx

        wins_by_trial_session.setdefault((session, subject, trial), set()).add(win)
        wins_by_trial.setdefault((subject, trial), set()).add(win)
        trials_by_subject_session.setdefault((session, subject), set()).add(trial)
        trials_by_subject.setdefault(subject, set()).add(trial)

    def _finalize_sets(dct):
        return {k: sorted(v) for k, v in dct.items()}

    return {
        "index_by_key": index_by_key,
        "index_by_subject_trial_win": index_by_subject_trial_win,
        "wins_by_trial_session": _finalize_sets(wins_by_trial_session),
        "wins_by_trial": _finalize_sets(wins_by_trial),
        "trials_by_subject_session": _finalize_sets(trials_by_subject_session),
        "trials_by_subject": _finalize_sets(trials_by_subject),
    }


class CustomDataset(TensorDataset):
    def __init__(self, d1, d2, d3):
        super(CustomDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def __len__(self):
        return len(self.d1)

    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx], self.d3[idx]

    def data(self):
        return self.d1

    def label(self):
        return self.d2

    def group(self):
        return self.d3
