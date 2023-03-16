import os
import copy
import time

import numpy as np
import pandas as pd
import seaborn as sns
from dtaidistance import dtw
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def fetch_distance(t1, t2, window=5):
    d, paths = dtw.warping_paths(t1, t2, window=window)
    return d


def overlay(trends, names=None, title = ' ', alphas=None, figsize=(17,4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    if names is None:
        names = [' ' for x in range(len(trends))]
    for i, (name, trend) in enumerate(zip(names, trends)):
        alpha = 1 if alphas is None else alphas[i]
        plt.plot(np.arange(len(trend)), trend, label=name, alpha=alpha)
        title = title #+ f' - {name}'
    plt.title(title)
    plt.legend()

def scale_df(df):
    # min max
    mx = df.max().max()
    mn = df.min().min()
    mtrx = (df-mn)/(mx-mn)
    # standard
    mu = mtrx.values.mean()
    sig = mtrx.values.std()
    mtrx = (mtrx-mu)/sig
    return mtrx, mtrx.values.max(), mtrx.values.min()


def bin_quartiles(mtrx, bins=16):
    df = pd.DataFrame(mtrx.values.flatten())
    df = pd.qcut(df[0], q=bins, labels=False)
    binned = pd.DataFrame(df.values.reshape((237, 237)))
    binned.columns = mtrx.columns
    binned.index = mtrx.index
    return binned, binned.values.max(), binned.values.min()


def plot_hand_match_comp(mtrx, vds_hm, mx, mn):
    #     def format_sub_matrix(mtrx, idxs):
    #         sub_mtrx = np.zeros((len(idxs),len(idxs)))
    #         for i, idx in enumerate(idxs):
    #             sub_mtrx[i,:] = mtrx[idx][idxs]
    #         return sub_mtrx

    def format_sub_matrix(mtrx, idxs):
        return mtrx.loc[mtrx.index.intersection(idxs), mtrx.index.intersection(idxs)]

    plot_format = """
                AAAB
                AAAC
                AAAC
                """
    fig, axes = plt.subplot_mosaic(plot_format, figsize=(15, 15))
    _ = sns.heatmap(mtrx, linewidth=0, vmin=mn, vmax=mx, ax=axes['A'],
                    linewidths=0.8, linecolor="grey", cmap='viridis_r',
                    cbar_kws={"shrink": 1, "pad": 0.05, "orientation": "horizontal"})
    axes['A'].title.set_text('Dynamic Time Warping Distance (All Sites)')

    # hand matched
    sub = format_sub_matrix(mtrx, vds_hm)
    #     sub.columns = sub.index = vds_hm
    _ = sns.heatmap(sub, linewidth=0.5, vmin=mn, vmax=mx, cbar_kws={"shrink": 1}, cmap='viridis_r',
                    cbar=False, ax=axes['B'], linewidths=0.8, linecolor="white")
    axes['B'].title.set_text('DTW Distance (Hand Matched Sites)')

    # dtw matched
    #     sub = format_sub_matrix(mtrx, vds_ids)
    #     _ = sns.heatmap(sub, linewidth=0.5,vmin=mn, vmax=mx, cbar_kws={"shrink": 1}, cmap = 'viridis_r',
    #                      cbar=False, ax=axes['C'], linewidths=0.8,linecolor="white")
    #     axes['C'].title.set_text('Dynamic Time Warping Distance (Matched Sites)')

    axes['C'].axis('off')
    plt.tight_layout()
    plt.show()


class ScaleInvariantDynamicTimeWarping:
    def __init__(self, data, root_dir, dtw_window=45, optimize_bool=False,
                 opt_method=None, opt_tol=None, minmax_scale=False, out_path='dtw_data'):
        self.data = data
        self.root_dir = root_dir
        self.dtw_window = dtw_window
        self.optimize_bool = optimize_bool
        self.opt_method = opt_method
        self.opt_tol = opt_tol
        self.minmax_scale = minmax_scale
        self.out_path = out_path
        self.folder_format = 'WindowSize-{}_Opt-{}_MinMax-{}_Tol-{}'
        self.file_format = 'Distances_TargetSite{}.csv'

    def fetch_trend_by_id(self, id_val):
        """ general trend loader """
        trend = self.data[id_val]['median_trend']
        if self.minmax_scale:
            mn, mx = trend.min(), trend.max()
            trend = (trend - mn) / (mx - mn)
        return trend

    def fetch_distance(self, t1, t2):
        d, paths = dtw.warping_paths(t1, t2, window=self.dtw_window)
        return d

    def scale_optimize(self, target_trend, match_trend):
        def opt_function(x, target_trend, match_trend, window):
            match_augmented = match_trend * x
            loss = fetch_distance(target_trend, match_augmented, window=window)
            return loss

        res = minimize(opt_function, 0, method=self.opt_method, tol=self.opt_tol,
                       args=(target_trend, match_trend, self.dtw_window))
        multiplier = res.x[0]

        return multiplier

    def collect_sidwt_distances(self, target_id, pre_loaded=None):
        """ collects DTW distance data from one trend to all others """
        # init pre_loaded to empty df if it is None
        preprocessed_sites = [] if pre_loaded is None else pre_loaded.site.to_list()
        pre_loaded = pd.DataFrame() if pre_loaded is None else pre_loaded

        distances = {}
        # load subject trend
        target_trend = self.fetch_trend_by_id(target_id)
        for i, site in enumerate(self.data.keys()):
            # skip preloaded
            if site in preprocessed_sites:
                continue
            # load trend to compare to
            match_trend = self.fetch_trend_by_id(site)
            # collect scale
            scale = self.scale_optimize(target_trend, match_trend) if self.optimize_bool else 1
            # compute distances
            d = self.fetch_distance(target_trend, match_trend * scale)
            distances[i] = {'site': site, 'distance': d, 'scale match': scale,
                            'opt_method': self.opt_method, 'minmax flag': self.minmax_scale}

        df_collected_data = pd.DataFrame(distances).T
        # will add an empty df to the collected data if pre_loaded is None
        df_all_sites = pd.concat([pre_loaded, df_collected_data]).sort_values(by=['distance'])
        return df_all_sites

    def collect_sidwt_distances_and_save(self, target_id):
        """ wrapper to collect_sidwt_distances() for saving data """
        # make dirs if needed
        folder_name = self.folder_format.format(self.dtw_window, self.opt_method, self.minmax_scale, self.opt_tol)
        path2dtw = os.path.join(self.root_dir, self.out_path, folder_name)
        if not os.path.exists(path2dtw):
            os.makedirs(path2dtw)

        # preload from a early version
        path2csv = os.path.join(path2dtw, self.file_format.format(target_id))
        if os.path.exists(path2csv):
            df_distances_pre_loaded = pd.read_csv(path2csv)

        # given a site_id find all matches
        df_distances = self.collect_sidwt_distances(target_id, pre_loaded=df_distances_pre_loaded)
        # format and save data
        df_distances.to_csv(path2csv, index=False)
        print(f'Saved: {path2dtw}/' + self.file_format.format(target_id))

    def collect_sidwt_distances_and_save_all_sites(self, cycle_data):
        """ uses collect_sidwt_distances_and_save() as a helper function for iteration. """
        for site in cycle_data.keys():
            start_time = time.time()
            self.collect_sidwt_distances_and_save(site)
            print(f'Site collection time: {round(time.time() - start_time)}')
        print('Collected all sites!')


class SIDTWanalyzer(ScaleInvariantDynamicTimeWarping):
    def __init__(self, data_all, csv_root, df_vds_info, vds2ccs_mapping, ScaleInvariantDynamicTimeWarping):
        self.df_vds_info = df_vds_info
        self.vds2ccs_mapping = vds2ccs_mapping
        super(SIDTWanalyzer, self).__init__(
            data_all,
            csv_root,
            dtw_window=ScaleInvariantDynamicTimeWarping.dtw_window,
            optimize_bool=ScaleInvariantDynamicTimeWarping.optimize_bool,
            opt_method=ScaleInvariantDynamicTimeWarping.opt_method,
            opt_tol=ScaleInvariantDynamicTimeWarping.opt_tol,
            minmax_scale=ScaleInvariantDynamicTimeWarping.minmax_scale
        )

    def ccs_bool(self, target):
        if 'ccs_id' in self.data[target]:
            return True
        elif 'vds_id' in self.data[target]:
            return False
        else:
            raise

    def fetch_name(self, target, ccs_bool):
        if ccs_bool:
            return self.data[target]['ccs_name']
        else:
            return self.data[target]['vds_name']

    def fetch_direction_by_name(self, name, ccs_bool):
        if ccs_bool:
            return name.split('_')[-1]
        else:
            d = self.df_vds_info[self.df_vds_info.ID == name]
            return d.Direction.values[0]

    def id2name(self, id_val):
        return self.data[id_val]['vds_name'] if 'vds_name' in self.data[id_val].keys() else self.data[id_val][
            'ccs_name']

    def name2id(self):
        return {self.id2name(k): k for k in self.data.keys()}

    def data2df(self):
        tmp_dict = copy.deepcopy(self.data)
        for k in self.data:
            if 'vds_name' in tmp_dict[k].keys():
                tmp_dict[k]['name'] = tmp_dict[k]['vds_name'];
                del tmp_dict[k]['vds_name']
                tmp_dict[k]['id'] = tmp_dict[k]['vds_id'];
                del tmp_dict[k]['vds_id']
            elif 'ccs_name' in tmp_dict[k].keys():
                tmp_dict[k]['name'] = tmp_dict[k]['ccs_name'];
                del tmp_dict[k]['ccs_name']
                tmp_dict[k]['id'] = tmp_dict[k]['ccs_id'];
                del tmp_dict[k]['ccs_id']
            else:
                raise
            del tmp_dict[k]['median_trend']
        return pd.DataFrame(tmp_dict).T

    def match_compare_handVsDTW(self, target_name, filter_direction=True, filter_type=True):
        def extract_info(df):
            ids = df.site.to_list()
            names = [f'{n} Rank:{round(d, 2)} Scale:{round(s, 2)}' for n, s, d in zip(df.name.to_list(),
                                                                                      df['scale match'].to_list(),
                                                                                      df['rank'].to_list())]
            return ids, names

        name2id = self.name2id()
        target_id = name2id[target_name]

        # init paths
        folder_name = self.folder_format.format(self.dtw_window, self.opt_method, self.minmax_scale, self.opt_tol)
        path2dtw = os.path.join(self.root_dir, self.out_path, folder_name)
        path2csv = os.path.join(path2dtw, self.file_format.format(target_id))

        # best matches df
        df = pd.read_csv(path2csv)

        # format df
        df['ccs_site'] = df.site.map(lambda x: self.ccs_bool(int(x)))
        df['name'] = df.apply(lambda x: self.fetch_name(int(x['site']), x['ccs_site']), axis=1)
        df['direction'] = df.apply(lambda x: self.fetch_direction_by_name(x['name'], x['ccs_site']), axis=1)
        df['rank'] = df.index

        # target type
        target_row = pd.DataFrame(df.iloc[0]).T
        target_name, target_direction = target_row.iloc[0]['name'], target_row.iloc[0]['direction']
        target_type_ccs_bool = target_row.iloc[0]['ccs_site']
        # filter to match opposites only
        if filter_type:
            df = df[df['ccs_site'] != target_type_ccs_bool]
            df = pd.concat([target_row, df])

        # --- DTW MATCH --- #
        ids_dwt, names_dwt = extract_info(df.iloc[:5])
        # collect trend data for plotting
        trends_dwt = [self.data[int(k)]['median_trend'] for k in ids_dwt]
        alphas = [1] + [0.25] * (len(ids_dwt) - 1)
        overlay(trends_dwt, names=names_dwt, title='DTW Matched', alphas=alphas)

        # --- HAND MATCH --- #
        if target_type_ccs_bool:
            ccsID = target_name.split('_')[0]
        else:
            # implied ccs site give vds
            # ccsID = self.vds2ccs_mapping[self.vds2ccs_mapping.VDS == target_vds].CCS_ID.to_list()[0]
            ccsID = self.vds2ccs_mapping[self.vds2ccs_mapping.VDS == target_id].CCS_ID.to_list()[0]

        # hand mapped vds sites
        vds = self.vds2ccs_mapping[self.vds2ccs_mapping.CCS_ID == ccsID].VDS.to_list()
        matched_vds = df[df.name.isin(vds)]
        # filter based on direction of target_vds
        if filter_direction:
            matched_vds = matched_vds[matched_vds.direction == target_direction]

        matched_vds = pd.concat([target_row, matched_vds])
        ids_handmatched, names_handmatched = extract_info(matched_vds)
        trends_handmatched = [self.data[int(k)]['median_trend'] for k in ids_handmatched]
        alphas = [1] + [0.25] * (len(ids_handmatched) - 1)
        overlay(trends_handmatched, names=names_handmatched, title='Hand Matched', alphas=alphas)
        plt.show()

        return df, matched_vds

    def gen_confusion_mtrx(self):
        # init paths
        folder_name = self.folder_format.format(self.dtw_window, self.opt_method, self.minmax_scale, self.opt_tol)
        path2dtw = os.path.join(self.root_dir, self.out_path, folder_name)

        keys = sorted(list(self.data.keys()))
        dfs = []
        for target_id in keys:
            # load dataframe
            path2csv = os.path.join(path2dtw, self.file_format.format(target_id))
            df = pd.read_csv(path2csv)
            df['site'] = df['site'].apply(np.int64)
            # format dataframe
            df['ccs_site'] = df.site.map(lambda x: self.ccs_bool(int(x)))
            df['name'] = df.apply(lambda x: self.fetch_name(int(x['site']), x['ccs_site']), axis=1)
            df['direction'] = df.apply(lambda x: self.fetch_direction_by_name(x['name'], x['ccs_site']), axis=1)
            df['rank'] = df.index
            # sort dataframe
            df = df.sort_values(by=['site'])
            df.index = df.site
            df = df[['distance']]
            df.columns = [target_id]
            dfs.append(df)
        confusionMatrix = pd.concat(dfs, axis=1)
        return confusionMatrix

    def plot_cfm(self, target_name, filter_direction=False):
        # --- Plot Confusion Matrix Data --- #
        # format target
        name2id = self.name2id()
        target_id = name2id[target_name]

        confusionMatrix = self.gen_confusion_mtrx()
        mtrx, mx, mn = scale_df(confusionMatrix)
        mtrx, mx, mn = bin_quartiles(mtrx, bins=32)
        # -- filter set -- #
        # target type
        target_bool = self.ccs_bool(int(target_id))
        target_direction = self.fetch_direction_by_name(target_name, target_bool)

        matched_vds = self.vds2ccs_mapping[self.vds2ccs_mapping.CCS_ID == target_id].VDS.to_list()
        # filter direction
        if filter_direction:
            matched_vds = matched_vds[matched_vds.direction == target_direction]

        vds_ids = matched_vds.site.to_list()
        # plot data
        plot_hand_match_comp(mtrx, vds_ids, mx, mn)
