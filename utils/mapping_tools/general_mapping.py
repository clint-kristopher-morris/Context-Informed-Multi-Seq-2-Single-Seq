import base64

import folium
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium import IFrame
from shapely.geometry import Point


def create_point_map(df, point_name):
    # cleaning
    df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].apply(pd.to_numeric, errors='coerce')
    # --- Converts df type to GeoDataFrame --- #
    # pointDF = df
    df['coordinates'] = df[['Latitude', 'Longitude']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = geopandas.GeoDataFrame(df, geometry='coordinates')
    df = df.dropna(subset=[point_name, 'Latitude', 'Longitude', 'coordinates'])
    return df


def init_map(center=[33.748997, -84.387985], zoom_start=10, map_type=None):
    # --- Generate Map --- #
    if map_type == 'Esri':
        """ Generates folium map and required layers. """
        map_out = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")
        tile = folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Esri Satellite',
            overlay=False, control=True).add_to(map_out)
        folium.LayerControl().add_to(map_out)
    elif map_type == 'Stamen Toner':
        map_out = folium.Map(center, zoom_start=zoom_start, tiles="Stamen Toner")
    elif map_type == 'cartodbpositron':
        map_out = folium.Map(center, zoom_start=zoom_start, tiles="cartodbpositron")
    else:
        map_out = folium.Map(center, zoom_start=zoom_start)
    return map_out


def plot_from_df(data, df, name, folium_map, color='blue', icon_size=(20, 20), opacity=1):
    def save_trend_fig(target_trend, target_name, i, data, row, width, height, color_val, resolution):
        trend = data[row.site]['median_trend']
        fig = plt.figure(figsize=(width, height))
        if not (trend == target_trend).all():
            plt.plot(np.arange(len(trend)), target_trend, color=base_red, label=target_name, alpha=0.9)
        plt.plot(np.arange(len(trend)), trend, color=color_val, label=row[name], alpha=0.9)

        plt.ylabel('Volumn', color='w')
        plt.title(f'Weekly Volumn Trend for Site: {row[name]} SI-DTW Distance Value {row.distance:.3f}', color='w')
        plt.legend(fancybox=True, framealpha=0, labelcolor='w')
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        #         plt.tight_layout()
        png = '/tmp/{}.png'.format(i)
        fig.savefig(png, dpi=resolution, transparent=True)
        return png

    base_red = '#005288'  # '#FEA0A0' #'#D40000'
    base_blue = '#8C001A'  # '#A7A9AC'#'#334d7d'#'#A0D0FE' #'#818AB7'
    txt_color = '#989994'

    height = 2.5
    width = 13
    resolution = 80

    params = {"ytick.color": txt_color,
              "xtick.color": txt_color,
              "axes.labelcolor": txt_color,
              "axes.edgecolor": txt_color}
    plt.rcParams.update(params)

    # CCS target
    target_trend, target_name = data[df.iloc[0].site]['median_trend'], df.iloc[0].default_name

    df = create_point_map(df, name)
    df = df.iloc[::-1]  # reverse order to plot target last
    for i, row in df.iterrows():
        if row.ccs_site is True:
            mul = -1  # used for text only
            color_val = outline = base_red  # '#BE0000' #'red'
            logo = 'https://i.ibb.co/Mk75ZR2/SX-BLUE-B.png'
        else:
            mul = 1  # used for text only
            color_val = base_blue  # '#5e8ebf' # 'blue' #
            outline = '#1F1F1F'
            logo = 'https://i.ibb.co/s9RXp3G/output-onlinepngtools-6.png'
            # 'https://i.ibb.co/FWkn381/SX-GRAY-BB3.png'#'https://i.ibb.co/HGG3mC5/SW-GRAY-BB.png'
            # #'https://i.ibb.co/yRXCwjn/SX-DGRAY-B.png'

        # --- POPUP IMAGE: CREATE --- #
        png = save_trend_fig(target_trend, target_name, i, data, row, width, height, color_val, resolution)
        # --- POPUP IMAGE: FORMAT --- #
        encoded = base64.b64encode(open(png, 'rb').read())

        html = '<img src="data:image/png;base64,{}">'.format

        iframe = IFrame(html(encoded.decode('UTF-8')), width=(width * resolution) + 20,
                        height=(height * resolution) + 20)
        popup = folium.Popup(iframe, max_width=2650)
        # --- ICON IMAGE --- #
        icon = folium.features.CustomIcon(logo, icon_size=icon_size)
        folium.Marker([row.Latitude, row.Longitude],
                      tooltip=row[name],
                      opacity=opacity,
                      popup=popup,
                      icon=icon).add_to(folium_map)

        # --- ADD TEXT --- # -webkit-text-fill-color: transparent;
        text_html = f"""<div style="font-size:24px; 
                                    font-family:arial black; 
                                    color:{outline}; 
                                    width:1000px;
                                    -webkit-text-fill-color:white;
                                    -webkit-text-stroke: 1px;
                                    font-weight:950">{row[name]}</div>"""
        folium.Marker(
            location=[row.Latitude - (0.0025 - 0.001 * mul), row.Longitude + 0.001],
            icon=folium.DivIcon(html=text_html)
        ).add_to(folium_map)

    return folium_map


def plot_from_df3(df, name, folium_map, color='blue', icon_size=(30, 30), opacity=1, popup_flag=True):
    if color == 'red':
        logo = 'https://i.ibb.co/Mk75ZR2/SX-BLUE-B.png'
    else:
        logo = 'https://i.ibb.co/s9RXp3G/output-onlinepngtools-6.png'

    coded_opacity = True if 'opacity' in df.columns else False

    df = create_point_map(df, name)
    for i, row in df.iterrows():

        popup = row[name] if popup_flag else None
        if coded_opacity:
            opacity = row.opacity
            if opacity == 1:
                CCS_IM_RED = 'https://i.ibb.co/pz0sF6V/SX-BLUE-RED-OLv2.png'
                icon = folium.features.CustomIcon(CCS_IM_RED, icon_size=icon_size)
            else:
                icon = folium.features.CustomIcon(logo, icon_size=icon_size)
        else:
            icon = folium.features.CustomIcon(logo, icon_size=icon_size)
        folium.Marker([row.Latitude, row.Longitude],
                      tooltip=f'{popup}',
                      # popup=popup,
                      opacity=opacity,
                      icon=icon).add_to(folium_map)
    return folium_map