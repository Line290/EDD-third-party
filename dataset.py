import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
from scipy.misc import imread, imresize


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None, 
                 max_len_token_structure=300, max_len_token_cell=100, image_size=448):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        self.image_size = image_size
        self.max_len_token_structure = max_len_token_structure
        self.max_len_token_cell = max_len_token_cell

        # Load image path
        if isinstance(self.split, list):
            pass
        elif isinstance(self.split, str):
            self.split = [self.split]
        self.img_paths = []
        self.captions_structure = []
        self.caplens_structure = []
        self.captions_cell = []
        self.caplens_cell = []
        self.number_cell_per_images = []

        for tmp_split in self.split:
            with open(os.path.join(data_folder, tmp_split + '_IMAGE_PATHS.txt'), 'r') as f:
                for line in f:
                    self.img_paths.append(line.strip())

            print("Split: %s, number of images: %d" % (tmp_split, len(self.img_paths)))

            # Load encoded captions structure
            with open(os.path.join(data_folder, tmp_split + '_CAPTIONS_STRUCTURE' + '.json'), 'r') as j:
                self.captions_structure.extend(json.load(j))

            # Load caption structure length (completely into memory)
            with open(os.path.join(data_folder, tmp_split + '_CAPLENS_STRUCTURE' + '.json'), 'r') as j:
                self.caplens_structure.extend(json.load(j))

            # Load encoded captions cell
            with open(os.path.join(data_folder, tmp_split + '_CAPTIONS_CELL' + '.json'), 'r') as j:
                self.captions_cell.extend(json.load(j))

            # Load caption cell length
            with open(os.path.join(data_folder, tmp_split + '_CAPLENS_CELL' + '.json'), 'r') as j:
                self.caplens_cell.extend(json.load(j))

            with open(os.path.join(data_folder, tmp_split + "_NUMBER_CELLS_PER_IMAGE.json"), "r") as j:
                self.number_cell_per_images.extend(json.load(j))

        self.max_cells_per_images = max(self.number_cell_per_images)
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of data image
        sored_caplens_structure = [(i, it) for i, it in enumerate(self.caplens_structure)]
        sored_caplens_structure.sort(key=lambda x: x[1], reverse=True)
        self.idx_mp = [it[0] for i, it in enumerate(sored_caplens_structure)]
        self.dataset_size = len(self.idx_mp)

    def __getitem__(self, idx):
        # The Nth caption structure corresponds to the Nth image
        # Load image
        i = self.idx_mp[idx]
        img = imread(self.img_paths[i])
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(
            img, (self.image_size, self.image_size), interp="cubic")
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img / 255.)

        if self.transform is not None:
            img = self.transform(img)
        
        # padding caption structure, 1 dimension
        captions_structure = self.captions_structure[i]
        captions_structure += [0] * (self.max_len_token_structure + 2 - len(captions_structure))

        caption_structure = torch.LongTensor(captions_structure)
        caplen_structure = torch.LongTensor([self.caplens_structure[i]])

        # padding caption cell, 2 dimension
        captions_cell = self.captions_cell[i]
        caplen_cell = self.caplens_cell[i]

        captions_cell = [it + [0]*(self.max_len_token_cell + 2 - len(it))
                         for it in captions_cell]
        padding_enc_caption_cell = [[0]*(self.max_len_token_cell + 2)
                                    for x in range(self.max_cells_per_images - len(captions_cell))]
        padding_len_caption_cell = [0] * (self.max_cells_per_images - len(captions_cell))
        captions_cell += padding_enc_caption_cell
        caplen_cell += padding_len_caption_cell

        captions_cell = torch.LongTensor(captions_cell)
        caplen_cell = torch.LongTensor(caplen_cell)

        number_cell_per_image = torch.LongTensor(
            [self.number_cell_per_images[i]])

        return img, caption_structure, caplen_structure, captions_cell, caplen_cell, number_cell_per_image

    def __len__(self):
        return self.dataset_size

