import torch
import numpy as np
import scipy.io as scio
import os


def get_data_path(file_path):
    """
    获取指定目录下所有 mat 文件路径（忽略以 . 开头的隐藏文件）。
    """
    data_path = []
    for f in os.listdir(file_path):
        if f.startswith("."):
            continue
        else:
            data_path.append(os.path.join(file_path, f))
    return data_path


def window_slice(data, time_steps):
    """
    对单个被试的 DE 特征做时间滑动窗口。

    原始实现 (RNN 版) 的输出形状为 (B, T, C)，这里改成 (B, C, T)，
    方便直接输入到 CNN (Conv1d)，其中:
        - B: segment/batch 数
        - C: 通道或特征维度（这里固定为 310）
        - T: 时间步长度 (time_steps)
    """
    # data 原始形状一般为 (trial, channel, time) 或类似，这里与 RNN 版本保持一致预处理
    data = np.transpose(data, (1, 0, 2)).reshape(-1, 310)  # (total_channel, 310)

    xs = []
    for i in range(data.shape[0] - time_steps + 1):
        # 取连续 time_steps 帧，形状 (time_steps, 310)
        seg = data[i: i + time_steps]
        xs.append(seg)

    # 堆叠后得到 (B, T, C)
    xs = np.stack(xs, axis=0)  # (B, T, 310)
    # 转换为 (B, C, T) 以适配 CNN: Conv1d 输入 [B, C, T]
    xs = np.transpose(xs, (0, 2, 1))  # (B, 310, T)
    return xs


def get_number_of_label_n_trial(dataset_name):
    """
    获取类别数、trial 数量以及对应标签。

    Returns
    -------
    trial: int
    label: int
    label_xxx: list 3*15
    """
    # global variables
    label_seed4 = [
        [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
    ]
    label_seed3 = [
        [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
        [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
        [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
    ]
    if dataset_name == "seed3":
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == "seed4":
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print("Unexcepted dataset name")


def load_trained_data(samples_path_list, args):
    """
    加载原始 mat 文件并做滑动窗口，返回 numpy 数组列表。
    """
    # load the label eeg_data
    _, _, labels = get_number_of_label_n_trial(args.dataset_name)
    label = labels[int(args.session) - 1]
    if args.dataset_name == "seed3":
        label = np.resize(label, (15,))
        label = np.reshape(label, (1, 15))
    elif args.dataset_name == "seed4":
        label = np.resize(label, (24,))
        label = np.reshape(label, (1, 24))

    X_train_all = []
    Y_tain_all = []
    # Iterate through each subject (there are 14 source subjects in both datasets)
    for path in samples_path_list:
        # load the sample eeg_data
        sample = scio.loadmat(path, verify_compressed_data_integrity=False)
        flag = 0
        X_train = []
        y_train = []
        for key, val in sample.items():
            if key.startswith("de_LDS"):
                # 这里得到的是 (B, C, T) 形式
                X_train.append(window_slice(val, args.time_steps))
                train_label = np.full((X_train[-1].shape[0], 1), label[0, flag])
                y_train.append(train_label)
                flag += 1
        X_train_one_subject = np.concatenate(X_train)
        y_train_one_subject = np.concatenate(y_train)
        X_train_all.append(X_train_one_subject)
        Y_tain_all.append(y_train_one_subject)
    return X_train_all, Y_tain_all


def normalize(features, select_dim=0, eps=1e-6):
    """
    简单的 min-max 归一化。
    默认在 batch 维度上统计 min/max，与你原来的实现保持一致。
    features: Tensor, 形状 (B, C, T)
    """
    features_min, _ = torch.min(features, dim=select_dim)
    features_max, _ = torch.max(features, dim=select_dim)
    features_min = features_min.unsqueeze(select_dim)
    features_max = features_max.unsqueeze(select_dim)
    return (features - features_min) / (features_max - features_min + eps)


def load4train(samples_path_list, args):
    """
    load the SEED eeg_data set for CNN.

    返回：
        - sample_res: List[Tensor], 每个元素形状为 (B, C, T)
        - label_res:  List[Tensor], 每个元素形状为 (B, 1)
    """
    train_sample, train_label = load_trained_data(samples_path_list, args)
    sample_res = []
    label_res = []
    for subject_index in range(len(train_sample)):
        # transfer from ndarray to tensor
        one_subject_samples = (
            torch.from_numpy(train_sample[subject_index]).type(torch.FloatTensor)
        )  # (B, C, T)
        one_subject_labels = (
            torch.from_numpy(train_label[subject_index]).type(torch.LongTensor)
        )  # (B, 1)
        # normalize tensor
        one_subject_samples = normalize(one_subject_samples)
        sample_res.append(one_subject_samples)
        label_res.append(one_subject_labels)
    return sample_res, label_res


def getDataLoaders(one_subject, args):
    """
    根据指定被试 one_subject 和 args 构建 DataLoader。

    输出的样本张量 shape 为 (B, C, T)，可以直接送入 Conv1d 等 CNN 模型。
    """
    pre_path = args.path
    config_path = {
        "file_path": pre_path + args.session + "/",
        "label_path": pre_path + "label.mat",
    }
    path_list = get_data_path(config_path["file_path"])
    try:
        target_path_list = [
            i
            for i in path_list
            if (i.startswith(config_path["file_path"] + str(int(one_subject)) + "_"))
        ]
        target_path = target_path_list[0]
    except Exception:
        print("target eeg_data not exist")
        raise
    path_list.remove(target_path)
    source_path_list = path_list

    # read from DE feature
    sources_sample, sources_label = load4train(source_path_list, args)
    targets_sample, targets_label = load4train(target_path_list, args)

    if len(targets_label) == 1:
        target_sample = targets_sample[0]
        target_label = targets_label[0]
    else:
        # 一般不会发生，只是做个保护
        target_sample = torch.cat(targets_sample, dim=0)
        target_label = torch.cat(targets_label, dim=0)

    # Generate Data loaders
    source_dsets = []
    for i in range(len(sources_sample)):
        source_dsets.append(
            torch.utils.data.TensorDataset(sources_sample[i], sources_label[i])
        )
    target_dset = torch.utils.data.TensorDataset(target_sample, target_label)

    source_loaders = []
    for j in range(len(source_dsets)):
        source_loaders.append(
            torch.utils.data.DataLoader(
                source_dsets[j],
                args.batch_size,
                shuffle=True,
                num_workers=args.num_workers_train,
                drop_last=True,
            )
        )
    test_loader = torch.utils.data.DataLoader(
        target_dset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers_test,
        drop_last=True,
    )
    return source_loaders, test_loader

