import os

import numpy as np
import torch
import torch.utils.data as data

from constant import CONSTANT


class VTTDataset(data.Dataset):
    """
	Video to Text The description class of the data set used to load and provide data
    Supports MSR-VTT and MSVD data sets
    The following input is required for the construction:
    1. Provide pkl files with textual features
    2. The npy file containing the video frame i3d feature
    Provide text and video npy features, and return data based on caption's id
    """

    def __init__(self, data_dir=CONSTANT.data_path, np_cap=CONSTANT.cap_train_path,
                rgbi3d_path=CONSTANT.rgbi3d_path, soundnet_path=CONSTANT.soundnet_path, 
                flowi3d_path=CONSTANT.flowi3d_path):

        data = np.load(os.path.join(data_dir, np_cap))
        rgbi3d_path = os.path.join(data_dir, rgbi3d_path)
        soundnet_path = os.path.join(data_dir, soundnet_path)

        # check availability of feature files
        data = np.array([r for r in data if os.path.isfile(os.path.join(rgbi3d_path, str(r[0]) + ".npy"))])

        self.video_ids = data[:, 0]
        self.captions = [torch.from_numpy(x) for x in data[:, 1]]

        self.rgbi3d_path = rgbi3d_path
        self.soundnet_path = soundnet_path
        if flowi3d_path:
            flowi3d_path = os.path.join(data_dir, flowi3d_path)
            self.flowi3d_path = flowi3d_path

    def __getitem__(self, index):
        """
		Return a training sample pair (including the video frame feature and the corresponding caption)
        According to the caption to find the corresponding video, so the need for video storage is in accordance with the id ascending order
        """
        caption = self.captions[index]
        video_id = self.video_ids[index]

        features = []
        # activity (i3d) feature
        rgbi3d_file = os.path.join(self.rgbi3d_path, str(video_id) + ".npy")
        rgbi3d_feature = torch.from_numpy(np.load(rgbi3d_file))
        features.append(rgbi3d_feature)

        # audio (soundnet) Feature
        soundnet_file = os.path.join(self.soundnet_path, str(video_id) + ".npy")
        if os.path.isfile(soundnet_file): # some video has no sound
            soundnet_feature = torch.from_numpy(np.load(soundnet_file))
        else:
            soundnet_feature = torch.zeros(1024)
        features.append(soundnet_feature)

        # flow i3d feature
        if isinstance(self.flowi3d_path, str):
            flowi3d_file = os.path.join(self.flowi3d_path, str(video_id) + ".npy")
            flowi3d_feature = torch.from_numpy(np.load(flowi3d_file))
            features.append(flowi3d_feature)

        vid_feature = torch.cat(features)

        return vid_feature, caption, index, video_id

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

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_vtt_loader(np_cap, batch_size=100, shuffle=True, num_workers=2, drop_last=False):
    v2t = VTTDataset(np_cap=np_cap)
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
    train_loader = get_vtt_loader(
        CONSTANT.cap_train_path, CONSTANT.batch_size, True, CONSTANT.workers, drop_last=True
    )
    val_loader = get_vtt_loader(
        CONSTANT.cap_val_path, CONSTANT.batch_size, False, CONSTANT.workers
    )

    return train_loader, val_loader


def get_test_loader():
    test_loader = get_vtt_loader(
        CONSTANT.cap_test_path, CONSTANT.batch_size, True, CONSTANT.workers
    )

    return test_loader
