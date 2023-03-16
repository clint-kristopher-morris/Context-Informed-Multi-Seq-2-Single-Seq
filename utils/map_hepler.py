import os
import sys
import folium
import pandas as pd

root = 'C:/Users/cmorris310/Desktop/projects/GDOT/support_data' # 'D:/GDOT-2022-AUX-DATA/GDOT-Data'
sys.path.append(root)
sys.path.append('./../../src')  # path to home dir
# import GeospaceSnapCorrection as GeoSanp

# not the best here
import warnings
warnings.filterwarnings("ignore")

import init_data
import common_utils
from data_handling import simple_data_load
from mapping_tools import general_mapping

obj_root = root + '/obj/'
csv_root = root + '/outputs/data'

matcher, inv_matcher = init_data.vds_name2webID(obj_root, root)
mapper, _ = init_data.load_mapper(obj_root)
vds_info_df = init_data.load_vds_location_data(obj_root, root, mapper, inv_matcher)
df_vds = pd.DataFrame(vds_info_df) # copy

ccs_info_df = init_data.load_ccs_location_data(root)

ccs_data = common_utils.load_obj('ccs_median_trends', root=obj_root)
ccs_good_data = {k:ccs_data[k] for k in ccs_data if ccs_data[k]['Exception'] is None}
vds_good_data, _ = simple_data_load.load_good_trend_data(obj_root, fname='vds_median_trends_2')
data_all = {**vds_good_data, **ccs_good_data}

r2tmp = 'C:/Users/cmorris310/Desktop/projects/GDOT/cleanGDOT/notebooks/implemented/tmp'
selected_sites = pd.read_csv(os.path.join(r2tmp, 'tmp.csv'))
selected_sites['default_name'] = selected_sites['name']
selected_sites['name'] = selected_sites.apply(lambda x: simple_data_load.correct_format_ccs(x['ccs_site'], x['name']), axis=1)

df_vds['name'] = df_vds['names']
df_vds_selected = pd.merge(df_vds, selected_sites, on='name')
df_ccs_selected = pd.merge(ccs_info_df, selected_sites, on='name')
df = pd.concat([df_ccs_selected, df_vds_selected], axis=0)

folium_map = general_mapping.init_map(map_type='Esri')
html_to_insert = "<style>.leaflet-popup-content-wrapper, .leaflet-popup.tip {background-color: #000113 !important; }</style>"
folium_map.get_root().header.add_child(folium.Element(html_to_insert))
folium_map = general_mapping.plot_from_df(data_all, df, 'default_name', folium_map, color='red',
                                          icon_size=(90,90), opacity=1)