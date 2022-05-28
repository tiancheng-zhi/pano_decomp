import random
import math
import torch
import torch.utils.data as data

from .pano_reader import get_reader


class PanoDataset(data.Dataset):
    def __init__(self, dsets, record_lists, height, modals, repeat=1, rotate=False):
        super().__init__()
        if type(dsets) is str:
            dsets = [dsets]
        if type(record_lists) is str:
            record_lists = [record_lists]
        assert(len(dsets) == len(record_lists))
        self.records = []
        self.readers = {}
        for i, dset in enumerate(dsets):
            self.readers[dset] = get_reader(dset, height)
            if type(record_lists[i]) == list:
                cur_records = record_lists[i]
            else:
                f = open(record_lists[i], 'r')
                cur_records = [i.strip() for i in f.readlines()]
                f.close()
            self.records = self.records + list(zip([dset] * len(cur_records), cur_records))
        self.height = height
        self.width = height * 2
        self.modals = modals
        self.repeat = repeat
        self.rotate = rotate

    def __len__(self):
        return len(self.records) * self.repeat

    def __getitem__(self, index):
        index = index % len(self.records)
        record = self.records[index]
        dset_id, record_id = record
        entry = [index, dset_id, record_id]
        if self.rotate:
            shifts = random.randint(0, self.width-1)
        else:
            shifts = 0
        for modal in self.modals:
            if modal == 'rgb' or modal == 'lowres' or modal == 'diffuse' or modal == 'specular' or modal == 'sunlight' or modal == 'ambient':
                cur = self.readers[dset_id].get_rgb_image(record_id, modal)
                cur = torch.cat((cur[:, :, shifts:], cur[:, :, :shifts]), 2)
            elif modal == 'tripod' or modal == 'light' or modal == 'coarsesunlight' or modal == 'finesunlight':
                cur = self.readers[dset_id].get_gray_image(record_id, modal)
                cur = torch.cat((cur[:, :, shifts:], cur[:, :, :shifts]), 2)
            elif modal == 'semantic':
                cur = self.readers[dset_id].get_semantic_image(record_id, modal)
                cur = torch.cat((cur[:, shifts:], cur[:, :shifts]), 1)
            elif modal == 'hdr':
                cur = self.readers[dset_id].get_hdr_image(record_id)
                cur = torch.cat((cur[:, :, shifts:], cur[:, :, :shifts]), 2)
            elif modal == 'sundir':
                cur = self.readers[dset_id].get_sundir_vector(record_id)
            else:
                cur = self.readers[dset_id].get_arbitrary_image(record_id)
            entry.append(cur)
        return entry
