import sys
sys.path.append("..")

from utils.common_utils import load_obj

def correct_format_ccs(bool_, val):
    if bool_:
        val = f"00000{val.split('_')[0].replace('-','')}"
    return val

def load_good_trend_data(root, fname='vds_median_trends'):
    # load data
    data_tmp = load_obj(fname, root=root)
    print(f'Length of data collected: {len(data_tmp)}')

    # filter good data
    good_data = {}
    for k in data_tmp.keys():
        if data_tmp[k]['Exception'] is None:
            good_data[data_tmp[k]['vds_id']] = data_tmp[k]
    print(f'Length of good data: {len(good_data)}')
    return good_data, data_tmp