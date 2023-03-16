import pandas as pd
from utils.common_utils import load_obj

def vds_name2webID(obj_root, root, vds_info_fname='{}/data/vds/location_data/vds_info_total.csv'):
    # map_ is a dict converting names to station numbers
    map_ = load_obj('map', root=obj_root)
    del map_['would_not_load']
    del map_['loaded_not_a_name']
    map_df = pd.DataFrame(map_, index=['vds_station_number']).T

    vds_info = pd.read_csv(vds_info_fname.format(root))
    matcher = dict(zip(list(vds_info['value']), list(vds_info['description'])))
    # remove if not GDOT
    delete = []
    onlyGDOT = True
    for key, val in matcher.items():
        if onlyGDOT:
            if val[0] != 'G':
                delete.append(key)
            else:
                matcher[key] = val.split(':')[0]
        else:
            matcher[key] = val.split(':')[0]
    for key in delete:
        del matcher[key]

    inv_matcher = {v: k for k, v in matcher.items()}

    return matcher, inv_matcher


def load_mapper(obj_root):
    # map_ is a dict converting names to station numbers
    map_ = load_obj('map', root=obj_root)
    del map_['would_not_load']
    del map_['loaded_not_a_name']
    map_df = pd.DataFrame(map_, index=['vds_station_number']).T
    return map_, map_df


def load_vds_location_data(obj_root, root, mapper, inv_matcher, vds_names='{}/support/inventory_xpath.csv'):
    def covertname(x, mapper):
        try:
            x = inv_matcher[x]
        except KeyError:
            try:
                x = mapper[x]
            except KeyError:
                x = 0
        return int(x)

    # info is a dict with extended station info eg lat long
    # mostly all formatting
    info_df = pd.DataFrame(load_obj('info', root=obj_root)).T
    names = pd.read_csv(vds_names.format(root), usecols=[0]).names  # col names
    info_df.columns = names
    info_df['names'] = info_df.index
    info_df.index = pd.RangeIndex(len(info_df.index))

    # convert ID name to number format
    info_df['ID2'] = info_df['ID'].map(lambda x: covertname(x, mapper=mapper))
    # only the ones being scrapped
    info_df = info_df[info_df['ID2'] != 0]

    # --- plot all GDOT VDS stations --- #
    info_df['ID3'] = info_df['ID2'].astype(str) + ' | ' + info_df['Direction'] + ' | ' + info_df['Roadway_Name']
    # plotfromdf(info_df, 'ID3', color='blue', size=3)
    info_df['ID2'] = info_df['ID2'].astype(str)
    return info_df


def load_ccs_location_data(root, ccs_fname="{}/data/vds/location_data/ccs_info.csv"):
    CCS_info = pd.read_csv(ccs_fname.format(root))
    CCS_info['fixed_idx'] = pd.RangeIndex(len(CCS_info.index))

    # --- plot all CCS stations --- #
    CCS_info['fixed_idx_v2'] = CCS_info['fixed_idx'].astype(str)+' | '+CCS_info['description']
    # plotfromdf(CCS_info, 'fixed_idx_v2', color='red', size=4)
    CCS_info['name2'] = CCS_info['name'].map(lambda x: str(x.replace('#','')) if "#" in x else str(x))
    return CCS_info