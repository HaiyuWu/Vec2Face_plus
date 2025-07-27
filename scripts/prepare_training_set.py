import os
from os import path, makedirs
import lmdb
import msgpack
import numpy as np
import pandas as pd
from PIL import Image
from os import path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class ImageListRaw(ImageFolder):
    def __init__(self, feature_list, label_file, image_list, landmark_list=None):
        image_names = np.asarray(pd.read_csv(image_list, delimiter=" ", header=None))
        feature_names = np.asarray(pd.read_csv(feature_list, delimiter=" ", header=None))
        landmark_names = np.asarray(pd.read_csv(landmark_list, delimiter=" ", header=None)) if landmark_list else None

        self.im_samples, self.feat_samples, self.landmark_samples = \
            self.get_landmark(image_names,
                              feature_names,
                              landmark_names)

        self.targets = np.loadtxt(label_file, int)
        self.classnum = np.max(self.targets) + 1

        print(self.classnum)

    def get_landmark(self, im_files, feature_files, landmark_files):
        im_files = list(im_files[:, 0])
        feature_files = list(feature_files[:, 0])
        ims, feats, landmarks = [], [], []
        im_files.sort(key=lambda x: x.split("/")[-1])
        feature_files.sort(key=lambda x: x.split("/")[-1])
        if landmark_files is not None:
            landmark_files = list(landmark_files[:, 0])
        else:
            return im_files, feature_files, None

        landmark_files.sort(key=lambda x: x.split("/")[-1])

        i = k = 0
        while i < len(im_files) and k < len(landmark_files):
            if im_files[i].split("/")[-1] != landmark_files[k].split("/")[-1]:
                i += 1
            else:
                ims.append(im_files[i])
                feats.append(feature_files[i])
                landmarks.append(landmark_files[k])
                i += 1
                k += 1

        return ims, feats, landmarks

    def __len__(self):
        return len(self.im_samples)

    def __getitem__(self, index):
        assert path.split(self.im_samples[index])[1][:-4] == path.split(self.feat_samples[index])[1][:-4]
        with open(self.im_samples[index], "rb") as f:
            img = f.read()
        with open(self.feat_samples[index], "rb") as f:
            feature = f.read()
        if self.landmark_samples is None:
            return img, feature, self.targets[index]
        else:
            assert path.split(self.im_samples[index])[1][:-4] == path.split(self.landmark_samples[index])[1][:-4]
            with open(self.landmark_samples[index], "rb") as f:
                landmark = f.read()
            return img, feature, self.targets[index], landmark


class CustomRawLoader(DataLoader):
    def __init__(self, workers, feature_list, label_file, landmark_list, image_list):
        self._dataset = ImageListRaw(feature_list, label_file, image_list, landmark_list)

        super(CustomRawLoader, self).__init__(
            self._dataset, num_workers=workers, collate_fn=lambda x: x
        )


def list2lmdb(
    feature_list,
    label_file,
    image_list,
    landmark_list,
    dest,
    file_name,
    num_workers=16,
    write_frequency=50000,
):
    print("Loading dataset from %s" % image_list)
    data_loader = CustomRawLoader(
        num_workers, feature_list, label_file, landmark_list, image_list
    )
    name = f"{file_name}.lmdb"
    if not path.exists(dest):
        makedirs(dest)
    lmdb_path = path.join(dest, name)
    isdir = path.isdir(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")

    image_size = 112
    size = len(data_loader.dataset) * image_size * image_size * 3
    print(f"LMDB max size: {size}")

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(len(data_loader.dataset))
    txn = db.begin(write=True)
    for idx, data in tqdm(enumerate(data_loader)):
        if len(data[0]) == 3:
            image, feature, label = data[0]
            txn.put(
                "{}".format(idx).encode("ascii"), msgpack.dumps((image, feature, int(label)))
            )
        else:
            image, feature, label, landmark = data[0]
            txn.put(
                "{}".format(idx).encode("ascii"), msgpack.dumps((image, feature, int(label), landmark))
            )

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)
    idx += 1

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__len__", msgpack.dumps(len(keys)))
        txn.put(b"__classnum__", msgpack.dumps(int(data_loader.dataset.classnum)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list", "-im", help="List of images.", type=str)
    parser.add_argument("--feature_list", "-f", help="List of features.", type=str)
    parser.add_argument("--landmark_list", "-landmark", help="List of features.", default=None, type=str)
    parser.add_argument("--label_file", "-l", help="Identity label file.", type=str)
    parser.add_argument("--workers", "-w", help="Workers number.", default=8, type=int)
    parser.add_argument("--dest", "-d", help="Path to save the lmdb file.", type=str)
    parser.add_argument("--file_name", "-n", help="lmdb file name.", type=str)
    args = parser.parse_args()

    list2lmdb(
        args.feature_list,
        args.label_file,
        args.image_list,
        args.landmark_list,
        args.dest,
        args.file_name,
        args.workers,
    )
