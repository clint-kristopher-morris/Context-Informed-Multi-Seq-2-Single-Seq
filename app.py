import warnings

import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from utils.constants import *
from utils.init_paths import *
from utils import common_utils
from utils.data_handling import sample_data_loader
from utils.mapping_tools import general_mapping
from utils.modeling_utils import si_dtw_utils

from modeling import CIMS2SS_model

warnings.filterwarnings("ignore")



@st.cache_resource
def load_map():
    map_data = general_mapping.init_map(center=[32.748997, -83.087985], zoom_start=6.5, map_type='Esri')
    map_data = general_mapping.plot_from_df3(DF_CCS, 'name', map_data, color='red')
    return map_data

@st.cache_resource
def load_model():
    # Load the model
    model = CIMS2SS_model.TimeSeriesNeighborPredictor(device=True, testing=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    return model

@st.cache_data
def load_median_ts_data():
    data_all = common_utils.load_obj('data_all', root=obj_root)
    data_all_names = common_utils.load_obj('data_all_names', root=obj_root)
    return data_all, data_all_names


@st.cache_data
def load_sample_data(target):
    testing_data = sample_data_loader.load_data_from_name(fname=f'tmp_data_window_24hr_{target}_v2', shuffle_order=True)
    testing_data = torch.utils.data.DataLoader(testing_data, batch_size=1, drop_last=True, shuffle=True)
    return testing_data


def load_sample_ts_df(model, data):
    device = 'cpu'
    src_in, trg_in, discrete_in = next(iter(data))

    src = torch.permute(src_in, (0, 2, 1))
    trg = torch.reshape(trg_in, (1, 1, -1))
    src, trg, discrete_in = src.to(device), trg.to(device), discrete_in.to(device)

    y_pred = model(src, discrete_in)

    df = pd.DataFrame(src[0].cpu()).T
    df['pred'] = y_pred[0].cpu().reshape(-1).detach().numpy()
    df['true'] = trg[0].cpu().reshape(-1)

    return df

def init_trend_fig(title='Weekly Median Traffic Volume Trend'):
    fig_trend = go.Figure()
    fig_trend.update_layout(margin=dict(l=5, r=5, t=20, b=2), height=220, width=850,title=title)
    return fig_trend


def add_trend(fig, name, DATA_ALL_NAMES, color='blue', opacity=1):
    y = DATA_ALL_NAMES[name]['median_trend']
    fig.add_trace(go.Scatter(x=TIMES, y=y, mode='lines', name=name,line=dict(color=color),opacity=opacity))
    return fig


def load_trend(fig, df):
    x = TIMESx
    y = df.iloc[:, -2].values
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Pred.',
                             line=dict(color=RED_HEX),
                             opacity=1
                             ))
    y = df.iloc[:, -1].values
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='True',
                             line=dict(color=BLUE_HEX),
                             opacity=1
                             ))
    for i in range(4):
        y = df.iloc[:, i].values
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='',
                                 line=dict(color=GRAY_HEX),
                                 opacity=0.4
                                 ))

    return fig


def generate_barplot(df, target='SI-DTW', title='Scale Invariant Dynamic Time Warping', tb=25, h=200):
    df = df.iloc[1:][['Site ID', 'SI-DTW', 'Geospatial Distance (km)']]
    y = np.flip(df['Site ID'].values)
    x = np.flip(df[target].values)

    colors = [GRAY_HEX] * 5
    color_map = dict(zip(y, colors))

    if st.session_state.selected_vds:
        color_map[st.session_state.selected_vds] = RED_HEX

    fig = px.bar(x=x, y=y, orientation='h', color=y, color_discrete_map=color_map,
                 labels={"y": "Site ID", "x": target}, title=title)

    fig.update_layout(height=h, width=450, showlegend=False)
    fig.update_layout(margin=dict(l=5, r=5, t=tb, b=0))
    return fig


def main():
    st.set_page_config(TITLE, page_icon=TAB_ICON, layout='wide')
    st.markdown("""
            <style>
                   .block-container {
                        padding-top: 1rem;
                        padding-bottom: 0rem;
                        padding-left: 15rem;
                        padding-right: 15rem;
                    }
            </style>
            """, unsafe_allow_html=True)

    def matchdf2map(df):
        def clean_ccs(df):
            clean_names = []
            for i, row in df.iterrows():
                if row.CCS_Flag:
                    clean_names.append(row['Site ID'].split('_')[0])
                else:
                    clean_names.append(row['Site ID'])
            return clean_names

        selected_names = clean_ccs(df)
        # selected_names = df['Site ID'].values
        vds2plot = DF_VDS[DF_VDS.names.isin(selected_names)]
        ccs2plot = DF_CCS[DF_CCS.name.isin(selected_names)]
        # init map
        mapper = general_mapping.init_map(center=st.session_state.center, zoom_start=st.session_state.zoom,
                                          map_type='Esri')
        mapper = general_mapping.plot_from_df3(vds2plot, 'names', mapper, color='blue', opacity=0.5, popup_flag=True)
        mapper = general_mapping.plot_from_df3(ccs2plot, 'name', mapper, color='red')
        return mapper

    def layout_home():
        # init or reset values
        st.gen_prediction = None
        st.session_state.selected_vds_idx = None
        st.session_state.selected_vds = None

        _, r2_col1, r2_col2, r2_col3, _ = placeholder.columns([1, 4.5, 1, 6, 1])
        with r2_col1:
            r2_col1.markdown('## Multi-Sequence Site Linker')

            text1, text2 = "Continuous Count Stations (CCS)", "326 CCS Locations"
            st.markdown(FACT_BACKGROUND.format(text1, CCS_IM, 20, text2), unsafe_allow_html=True)
            st.markdown("""<div style="padding-top: 15px"></div>""", unsafe_allow_html=True)
            text1, text2 = "Video Detection Stations (VDS)", "2568 VDS Locations"
            st.markdown(FACT_BACKGROUND.format(text1, VDS_IM, 22, text2), unsafe_allow_html=True)


            for _ in range(10):
                st.markdown("")

            logo1, logo2, _ = st.columns([1, 1, 2])
            logo1.image(UGA_LOGO, width=100)
            logo2.image(GDOT_LOGO, width=100)

        with r2_col2:
            st.write("")

        with r2_col3:
            # tmp_map = MAP_DATA
            level1_map_data = st_folium(MAP_DATA, height=520, width=600)
            clicked_ccs_id = level1_map_data['last_object_clicked_tooltip']

            if clicked_ccs_id:
                if "selected_ccs" not in st.session_state:
                    st.session_state.selected_ccs = clicked_ccs_id
                else:
                    st.session_state.selected_ccs = clicked_ccs_id
                st.session_state.zoom = level1_map_data['zoom']
                center = level1_map_data['center']
                st.session_state.center = [float(center["lat"]), float(center["lng"])]

                placeholder.empty()
                # ph_im.empty()
                st.session_state.current = 1
                pages[st.session_state.current]()

    def selection_one():
        _, r2_col1, r2_col2, r2_col3, _ = placeholder.columns([1, 4.5, 1, 6, 1])

        with r2_col1:
            directions = CCS_D_MAP[st.session_state.selected_ccs]
            direction = st.radio('Direction of Traffic Flow',
                                 [f'{DIRECTION_MAP[d]} Bound' for d in directions],
                                 horizontal=True)

            st.session_state.direction = DIRECTION_MAP_INV[direction.split(' ')[0]]
            target = f'{st.session_state.selected_ccs}_{st.session_state.direction}'
            st.target = target
            # '063-1207_E'  # '063-1207_E' '063-1207_W'
            df, matched_vds = sidt_viewer.match_compare_handVsDTW(target, filter_direction=True, filter_type=True)
            df = df.iloc[:6]

            df = df[['name', 'ccs_site', 'direction', 'distance', 'scale match']]
            df.columns = ['Site ID', 'CCS_Flag', 'Direction', 'SI-DTW', 'SI-DTW_Scale']
            clean_names = df['Site ID'].map(lambda x: x.split('_')[0] if '_' in x else x).values
            trg_geo = GEO_MAP[clean_names[0]]
            df['Geospatial Distance (km)'] = [trg_geo[x] for x in clean_names]

        with r2_col2:
            st.write("")

        with r2_col3:
            # selected_map =
            # st.write(df)
            st_data = st_folium(matchdf2map(df), height=520, width=600)
            clicked_vds_id = st_data['last_object_clicked_tooltip']
            if clicked_vds_id:
                if "selected_vds" not in st.session_state:
                    st.session_state.selected_vds = clicked_vds_id
                else:
                    st.session_state.selected_vds = clicked_vds_id
                try:
                    st.session_state.selected_vds_idx = np.where(df['Site ID'].values == clicked_vds_id)[0][0]
                except IndexError:
                    st.session_state.selected_vds_idx = None

        with r2_col1:
            fig1 = generate_barplot(df, tb=60)
            fig2 = generate_barplot(df, target='Geospatial Distance (km)',
                                    title='Geospatial Distance (km)', h=165, tb=25)
            st.plotly_chart(fig1, theme="streamlit", use_container_width=False)
            st.plotly_chart(fig2, theme="streamlit", use_container_width=False)
            for _ in range(3):
                st.markdown("")

        with r2_col1:
            colors = [BLUE_HEX] + [GRAY_HEX]*5
            opacities = [1] + [0.4]*5

            if st.session_state.selected_vds_idx:
                colors[st.session_state.selected_vds_idx] = RED_HEX
                opacities[st.session_state.selected_vds_idx] = 0.8

            fig_trend = init_trend_fig()
            for name, c, o in zip(df['Site ID'].values, colors, opacities):
                add_trend(fig_trend, name, DATA_ALL_NAMES, color=c, opacity=o)
            st.plotly_chart(fig_trend, theme="streamlit", use_container_width=False)

            if st.gen_prediction:
                fig_trend2 = init_trend_fig(title='Predicted Volume Trend')
                sample_df = load_sample_ts_df(CIMS2SS_MODEL, st.sample_data)
                load_trend(fig_trend2, sample_df)
                st.plotly_chart(fig_trend2, theme="streamlit", use_container_width=False)

    CIMS2SS_MODEL = load_model()
    DATA_ALL, DATA_ALL_NAMES = load_median_ts_data()

    # --- init viewers --- #
    sidtw = si_dtw_utils.ScaleInvariantDynamicTimeWarping(DATA_ALL, csv_root, dtw_window=12, optimize_bool=True,
                                                          opt_method='L-BFGS-B', opt_tol=1e-3, minmax_scale=True)
    sidt_viewer = si_dtw_utils.SIDTWanalyzer(DATA_ALL, csv_root, DF_VDS, vds2ccs_mapping, sidtw)

    MAP_DATA = load_map()
    pages = {0: layout_home, 1: selection_one}

    if "current" not in st.session_state:
        st.session_state.current = 0
        st.session_state.selected_ccs = None
        st.session_state.direction = None
        st.session_state.zoom = None
        st.session_state.center = None
        st.session_state.selected_vds_idx = None
        st.gen_prediction = None
        st.target = None
        st.sample_data = None

    st.title(TITLE)
    # st.markdown("***")
    _, r2_col1, r2_col2, r2_col3, _ = st.columns([1, 4.5, 1, 6, 1])
    with r2_col1:
        st.markdown(f"<p style='font-size: 27px;'><i>Smart Mobility and Infrastructure Laboratory</i></p>",
                    unsafe_allow_html=True)
    with r2_col3:
        x1, x2, _ = st.columns([1, 1, 0.2])
        # Now you can set the button click to a number and call the linked function
        with x1:
            if st.button("CCS Selection Home"):
                st.session_state.current = 0
        with x2:
            if st.session_state.current == 1:
                if st.target in TRAIN_SITES:
                    st.sample_data = load_sample_data(st.target)
                    if st.button("Randomized Sample Prediction"):
                        st.gen_prediction = True

    placeholder = st.empty()

    if st.session_state.current != None:
        pages[st.session_state.current]()


if __name__ == "__main__":
    main()
