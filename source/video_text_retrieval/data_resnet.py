import torch
import torch.utils.data as data
import numpy as np
import os
from constant import CONSTANT
# import torchvision.transforms as transforms
# import nltk
# from PIL import Image
# import json as jsonmod
# import pickle


class VTTDataset(data.Dataset):
    """
	Video to Text description class of the data set used to load and provide data
    Supports MSR-VTT and MSVD data sets
    The following input is required for the construction:
    1. Provide pkl files with textual features
    2. The npy file containing the video frame i3d feature
    Provide text and video npy features, and return data based on caption's id
    """

    def __init__(self, cap_pkl, vid_feature_dir):
        data = np.load(cap_pkl)
        path = os.path.join(vid_feature_dir, CONSTANT.resnet_path)
        data = np.array([r for r in data if os.path.isfile(os.path.join(path, str(r[0]) + ".npy"))])

        self.video_ids = data[:, 0]
        self.captions = [torch.from_numpy(x) for x in data[:, 1]]

        # imfeat_file = os.path.join(feature_file, data_name)
        self.vid_feat_dir = vid_feature_dir

    def __getitem__(self, index):
        """
		Return a training sample pair (including the video frame feature and the corresponding caption)
        According to the caption to find the corresponding video, so the need for video storage is in accordance with the id ascending order
        """
        caption = self.captions[index]
        # length = self.lengths[index]
        video_id = self.video_ids[index]
        vid_feat_dir = self.vid_feat_dir

        path = os.path.join(vid_feat_dir, CONSTANT.resnet_path, str(video_id) + ".npy")
        video_feat = torch.from_numpy(np.load(path))
        # video_feat = video_feat.mean(dim=0, keepdim=False)  #  average pooling
        video_feat = video_feat.float()

        return video_feat, caption, index, video_id

    def __len__(self):
        return len(self.captions)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_vtt_loader(cap_pkl, feature, batch_size=100, shuffle=True, num_workers=2, drop_last=False):
    v2t = VTTDataset(cap_pkl, feature)
    data_loader = torch.utils.data.DataLoader(
        dataset=v2t,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
    return data_loader


def get_loaders():
    dpath = CONSTANT.data_path
    # if opt.data_name.endswith("vtt"):
    train_caption_pkl_path = os.path.join(dpath, CONSTANT.cap_train_path)
    val_caption_pkl_path = os.path.join(dpath, CONSTANT.cap_val_path)
    # feature_path = dpath
    train_loader = get_vtt_loader(
        train_caption_pkl_path, dpath, CONSTANT.batch_size, True, CONSTANT.workers, drop_last=True
    )
    val_loader = get_vtt_loader(
        val_caption_pkl_path, dpath, CONSTANT.batch_size, False, CONSTANT.workers
    )

    return train_loader, val_loader


def get_test_loader():
    dpath = CONSTANT.data_path

    # if opt.data_name.endswith("vtt"):
    test_caption_pkl_path = os.path.join(dpath, CONSTANT.cap_test_path)
    test_loader = get_vtt_loader(
        test_caption_pkl_path, dpath, CONSTANT.batch_size, True, CONSTANT.workers
    )

    return test_loader

