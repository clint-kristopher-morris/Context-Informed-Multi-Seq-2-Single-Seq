import numpy as np
import torch
from torch.utils.data import Dataset
from utils import common_utils
from utils.init_paths import *

OBJ_ROOT = obj_root

class TransformerDataset(Dataset):
    """ Dataset class used for transformer models.
    """

    def __init__(self,
                 src_data: torch.tensor,
                 trg_data: torch.tensor,
                 disc: torch.tensor,
                 indices: list,
                 shuffle_aug_flag=False,
                 ts_feature_count=9,
                 fsf_feature_count=5,
                 prob_aug=1
                 ) -> None:
        """
        Args:
            data: tensor, the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple /has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """
        super().__init__()
        self.indices = indices
        self.src_data = src_data
        self.trg_data = trg_data
        self.disc = disc
        self.shuffle_aug_flag = shuffle_aug_flag
        self.ts_feature_count = ts_feature_count
        self.fsf_feature_count = fsf_feature_count
        self.prob_aug = prob_aug
        print("From get_src_trg: data size = {}".format(src_data.size()))

    def unison_shuffled_copies(self, a, b):
        assert len(a[0]) == self.ts_feature_count
        assert len(b) == self.fsf_feature_count
        p = np.random.permutation(5)
        p2 = np.concatenate((p, np.array([5, 6, 7, 8])), axis=None)
        return a[:, p2], b[p]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """ Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        cs_idx, ts_index = self.indices[index]
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = ts_index[0]
        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = ts_index[1]
        self.vals = [start_idx, end_idx]

        src = self.src_data[start_idx:end_idx]
        trg = self.trg_data[start_idx:end_idx]
        # using start_idx was a bug fix. Not ideal
        cross_sectional = self.disc[start_idx]  # fixed site features

        if self.shuffle_aug_flag:
            #             if np.random.random(1) < self.prob_aug
            src, fsf = self.unison_shuffled_copies(src, cross_sectional)

        return src, trg, cross_sectional


def load_data_from_name(fname='tmp_data_window_24hr_train', shuffle_order=True):
    # --- Load data --- #
    data = common_utils.load_obj(fname, root=OBJ_ROOT)
    out_src_data, out_trg_data = data['src'], data['trg']
    out_discrete_data, out_indices = data['disc'], data['indices']
    out_indices = [(i, x) for i, x in enumerate(out_indices)]
    np.random.shuffle(out_indices)  # batch diversity
    # --- Format DataLoader --- #
    data = TransformerDataset(torch.tensor(out_src_data).float(),
                                              torch.tensor(out_trg_data).float(),
                                              torch.tensor(out_discrete_data).float(),
                                              out_indices,
                                              shuffle_aug_flag=shuffle_order)

    return data
