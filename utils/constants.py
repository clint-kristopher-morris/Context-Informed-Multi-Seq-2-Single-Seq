import datetime
import os

import pandas as pd

from utils import common_utils
from utils.init_paths import *


# --- Logos --- #
CCS_IM_RED = 'https://i.ibb.co/3f9Y0SG/SX-BLUE-RED-OL.png'
CCS_IM = 'https://i.ibb.co/Mk75ZR2/SX-BLUE-B.png'
VDS_IM = 'https://i.ibb.co/s9RXp3G/output-onlinepngtools-6.png'


# --- Load Init Data --- #
MODEL_PATH = f'{root}/outputs/model_state/model.pth'

# maps vds and ccs ids to information about them
GEO_MAP = common_utils.load_obj('geo_km_light', root=obj_root)
CCS_D_MAP = common_utils.load_obj('ccs_direction_map', root=obj_root)
# DATA_ALL = common_utils.load_obj('data_all', root=obj_root)
# DATA_ALL_NAMES = common_utils.load_obj('data_all_names', root=obj_root)

DF_VDS = pd.read_csv(f'{csv_root}/snapped_vds4app.csv')
DF_CCS = pd.read_csv(f'{csv_root}/ccs_data4app_cleaned.csv')
vds2ccs_mapping = pd.read_csv(f'{csv_root}/matched-sites_02_03_23.csv')

DIRECTION_MAP = {'N': 'North', 'S': 'South', 'W': 'West', 'E': 'East'}
DIRECTION_MAP_INV = {v: k for k, v in DIRECTION_MAP.items()}

TRAIN_SITES = [x.replace('tmp_data_window_24hr_','').replace('_v2.pkl','') for x in os.listdir(obj_root) if 'tmp_data_window_24hr_' in x]
TRAIN_SITES_BASE_NAMES = [s.split("_")[0] for s in TRAIN_SITES]
DF_CCS['opacity'] = DF_CCS['name'].map(lambda x: 1 if x in TRAIN_SITES_BASE_NAMES else 0.4)


# --- Style Guide --- #
BLUE_HEX = "#01416b"
RED_HEX = "#b22222"
GRAY_HEX = "#A7A9AC"


# --- Home Layout --- #
TITLE = 'Synergistic Multi-source Predictor'
SUB_TITLE = 'Smart Mobility and Infrastructure Lab'
TAB_ICON = "https://tinyurl.com/m2fnfmzm"
ICON_SIZE = 10
FACT_BACKGROUND = """
                    <div style="width: 100%;">
                        <div style="
                                    background-color: #ECECEC;
                                    border: 1px solid #ECECEC;
                                    padding: 1.5% 1% 1.5% 3.5%;
                                    border-radius: 10px;
                                    width: 100%;
                                    color: white;
                                    white-space: nowrap;
                                    ">
                          <p style="font-size:20px; color: black;">{}</p>
                          <p style="font-size:33px; line-height: 0.5; text-indent: 10px;""><img src="{}" alt="Example Image" style="vertical-align: middle;  width:{}px;">  {} &emsp; &emsp; </p>
                        </div>
                    </div>
                    """
UGA_LOGO = 'https://tinyurl.com/UGALOGO'
GDOT_LOGO = "https://i.ibb.co/HP5kzVh/output-onlinepngtools-8.png"


# --- Time Series Plots --- #
start = "00:00:00"
s = datetime.datetime.strptime(start, '%H:%M:%S')
TIMES = [datetime.datetime.strftime((s + datetime.timedelta(minutes=5 * x)), '%a %H:%M') for x in range(288 * 7)]
TIMESx = [datetime.datetime.strftime((s + datetime.timedelta(minutes=5 * x)), '%H:%M') for x in range(288 * 7)]