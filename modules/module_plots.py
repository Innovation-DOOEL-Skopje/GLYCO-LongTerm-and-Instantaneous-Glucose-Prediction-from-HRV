from typing import List, Dict, Tuple, Union
import matplotlib.gridspec as gridspec
import module_debug
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import importlib
importlib.reload(module_debug)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.figsize"] = (10, 7)
#____________________________________________________________________________________________________________________________________


def setup_figures(shape: Tuple = None) -> None:

    plt.rcParams["figure.figsize"] = shape
#____________________________________________________________________________________________________________________________________


def series_apply_pval(s):
    return s.apply(p_values_apply)


def series_apply_percents(s):
    return s.apply(percents_apply)


def p_values_apply(p_val):
    if p_val <= 0.05: return 1
    elif p_val <= 0.1: return .25
    else: return .05


def percents_apply(percent):
    if abs(percent) <= 0.01: return .05
    if abs(percent) <= 0.05: return .25
    else: return 1

#____________________________________________________________________________________________________________________________________


def heatmap__scaled_by_pvalue__1x2(dict_by_position: Dict[int, List],
                                   save_path: str = None,
                                   size_scale: int = 250,
                                   max_corr: float = 1,
                                   annot_shift_x: float = .39,
                                   annot_shift_y: float = .14,
                                   annot_white_treshold = None,
                                   annot_fontsize: Union[int, float] = 10,
                                   figsize: Tuple = (12, 5),
                                   show_point: bool = True,
                                   percentages: bool = False,
                                   ) -> None:

    """
    Plotting a 1x2 grid of correlation matrices.
    Positions increase from left to right and then from top to bottom.

    Positions:    0   1

    :param dict_by_position: keys are positions, values are a list of correlation and p-value dataframes
    :param save_path: [str, None]
        If None, only show the plot.
    :param max_corr: defines the max range of the color spectre
    :param annot_fontsize: fontsize of the annotations
    :param annot_white_treshold: absolute threshold above which annotations turn white for better visibility
    :param size_scale: scale of the plot
    :param annot_shift_y: shift points down on y axis
    :param annot_shift_x: shift points left on x axis
    :param figsize: the shape of the matplotlib figures

    """
    # defaults
    if not annot_white_treshold:
        annot_white_treshold = max_corr * 0.7
    plt.rcParams["figure.figsize"] = figsize
    if percentages: show_point = False

    # define plot structure
    fig = plt.figure(tight_layout=True, dpi=500)
    gs = gridspec.GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # iterate axes and plot
    axes = [ax0, ax1]
    for i in range(2):

        ax = axes[i]

        # take correlation and p value matrix
        corr_pval_list = dict_by_position[i]
        corr = corr_pval_list[0]
        pval = corr_pval_list[1]

        # scaling cells by percentage
        if percentages:
            # consider p values and improvement percentage scaling
            pval_scaling = pval.apply(series_apply_pval)
            percent_scaling = corr.apply(series_apply_percents)

            # scale by the minimum of the two values
            pval = np.minimum(pval_scaling, percent_scaling)
        # scaling cells by p-val
        else:
            pval = pval.apply(series_apply_pval)

        # melt the dataframe, so scatter plot can be drawn
        corr = pd.melt(corr.reset_index(), id_vars = 'index')
        corr.columns = ['x', 'y', 'value']
        pval = pd.melt(pval.reset_index(), id_vars = 'index')
        pval.columns = ['x', 'y', 'p-value']

        # x goes on x axis
        # y goes on y axis
        # size is determined by the p-value
        x = corr['x']
        y = corr['y']
        size = pval['p-value']

        # define labels and positions on scatter axis
        x_labels = [v for v in x.unique()]
        y_labels = [v for v in y.unique()]
        x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
        y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

        # limits of the plot
        ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
        ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

        # calculations on colors
        n_colors = 256
        palette = sns.color_palette("RdBu_r", n_colors = n_colors)

        # range of correlations
        if max_corr is not None: color_max, color_min = [max_corr * x for x in [1, -1]]
        else: color_max, color_min = [var for var in [1, -1]]

        # nested function for mapping the correlation value to a color
        def value_to_color(val):

            # if outside of correlation scale
            if val >= color_max:
                return palette[-1]
            if val <= color_min:
                return palette[0]

                # position of value in the input range, relative to the length of the input range
            val_position = float((val - color_min)) / (color_max - color_min)
            # target index in the color palette
            ind = int(val_position * (n_colors - 1))

            return palette[ind]


        # def value_to_color_corr(corr_matrix):
        #     corr_matrix['color'] = corr_matrix['value'].apply(value_to_color)
        #     return corr_matrix

        # take nans as zero correlations
        corr.replace(to_replace = {np.NaN: 0}, inplace = True)

        # add color column
        # arg_c = value_to_color_corr(corr)
        corr['color'] = corr['value'].apply(value_to_color)

        scatter = ax.scatter(x = x.map(x_to_num),
                             y = y.map(y_to_num),
                             s = size * size_scale,
                             c = corr['color'],
                             marker = 's')

        # put correlation annotations in each cell
        for index_annot in range(corr.shape[0]):
            if size.at[index_annot] == 1:

                # temps
                annot_fontsize_temp = annot_fontsize

                # used for displaying percentages
                if not show_point:

                    if corr['value'].at[index_annot] >= 1:

                        annotation = str(round(corr['value'].at[index_annot], 2)).replace('.','')
                        if len(annotation) == 2: annotation += '0'
                        annot_fontsize_temp = annot_fontsize - 2
                        if corr['value'].at[index_annot] >= annot_white_treshold:
                            color = 'white'
                        else:
                            color = 'black'

                    elif corr['value'].at[index_annot] > 0:

                        annotation = str(round(corr['value'].at[index_annot], 2))[2:]
                        if corr['value'].at[index_annot] >= annot_white_treshold:
                            color = 'white'
                        else:
                            color = 'black'

                    elif corr['value'].at[index_annot] == 0:

                        annotation = '00'

                    else:
                        annotation = str(round(corr['value'].at[index_annot], 2))[3:]
                        if corr['value'].at[index_annot] <= -annot_white_treshold:
                            color = 'white'
                        else:
                            color = 'black'

                    if len(annotation) == 1:
                        annotation += '0'

                # for positive floats
                else:

                    if corr['value'].at[index_annot] >= 0:

                        annotation = str(round(corr['value'].at[index_annot], 2))[1:]
                        if corr['value'].at[index_annot] >= annot_white_treshold:
                            color = 'white'
                        else:
                            color = 'black'

                    # negative floats
                    else:

                        annotation = str(round(corr['value'].at[index_annot], 2))[2:]
                        if corr['value'].at[index_annot] <= -annot_white_treshold:
                            color = 'w'
                        else:
                            color = 'black'

                    if len(annotation) == 2:
                        annotation += '0'

                # annotate the point on the scatter plot
                ax.annotate(text = annotation,
                            fontstyle = 'normal', fontsize = annot_fontsize_temp,
                            fontweight = 'ultralight', c = color,
                            xy = (x.map(x_to_num).at[index_annot] - annot_shift_x,
                                  y.map(y_to_num).at[index_annot] - annot_shift_y))

        # Show column labels on the axes
        if i == 0:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x_to_num[v] for v in x_labels])
            ax.set_xticklabels(x_labels, rotation = 55, ha = 'right', fontsize = 9)
            ax.set_yticks([y_to_num[v] for v in y_labels])
            ax.set_yticklabels([v.replace(' min window', '') for v in y_labels])

        if i == 1:
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([x_to_num[v] for v in x_labels])
            ax.set_xticklabels(x_labels, rotation = 55, ha = 'right', fontsize = 9)
            ax.set_yticks([y_to_num[v] for v in y_labels])
            ax.set_yticklabels([v.replace(' min window', '') for v in y_labels])
            ax.yaxis.set_ticks_position( "right")

    plt.subplots_adjust(top = .93, bottom = .07, left = .06, right = .94, hspace = .3, wspace = .25)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path.replace('SD1/SD2', 'SD1divSD2').replace('(‰)', ''))
    else:
        plt.show()

    plt.clf()
    plt.close('all')

#____________________________________________________________________________________________________________________________________


def heatmap__scaled_by_pvalue__1x1(corr_pval_list: List,
                                   save_path: str = None,
                                   size_scale: int = 1000,
                                   max_corr: float = 1,
                                   annot_shift_x: float = .3,
                                   annot_shift_y: float = .12,
                                   annot_white_treshold: float = None,
                                   annot_fontsize: int = 18,
                                   xtickslabels_fontsize: int = 9,
                                   yticklabels_fontsize: int = 9,
                                   figsize: Tuple = (15, 8),
                                   show_point: bool = True,
                                   percentages: bool = False,
                                   ) -> None:

    """ Plot a correlation matrix with correlation cells scaled by p-value

    :param corr_pval_list: list of correlation matrix and p-value matrix. correlation matrix comes first
    :param save_path: [str, None]
        If None, only show the plot.
    :param max_corr: defines the max range of the color spectre
    :param annot_fontsize: fontsize of the annotations
    :param annot_white_treshold: absolute threshold above which annotations turn white for better visibility
    :param size_scale: scale of the plot
    :param annot_shift_y: shift points down on y axis
    :param annot_shift_x: shift points left on x axis

    """
    # defaults
    plt.rcParams["figure.figsize"] = figsize
    if percentages: show_point = False

    if not annot_white_treshold:
        annot_white_treshold = max_corr * 0.7

    # # define plot structure
    fig, ax = plt.subplots()

    # take correlation and p value matrix
    corr = corr_pval_list[0]
    pval = corr_pval_list[1]

    # scaling cells by percentage
    if percentages:
        # consider p values and improvement percentage scaling
        pval_scaling = pval.apply(series_apply_pval)
        percent_scaling = corr.apply(series_apply_percents)

        # scale by the minimum of the two values
        pval = np.minimum(pval_scaling, percent_scaling)
    # scaling cells by p-val
    else:
        pval = pval.apply(series_apply_pval)

    # melt the dataframe, so scatter plot can be drawn
    corr = pd.melt(corr.reset_index(), id_vars = 'index')
    corr.columns = ['x', 'y', 'value']
    pval = pd.melt(pval.reset_index(), id_vars = 'index')
    pval.columns = ['x', 'y', 'p-value']

    # x goes on x axis
    # y goes on y axis
    # size is determined by the p-value
    x = corr['x']
    y = corr['y']
    size = pval['p-value']

    # define labels and positions on scatter axis
    x_labels = [v for v in x.unique()]
    y_labels = [v for v in y.unique()]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    # limits of the plot
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # calculations on colors
    n_colors = 256
    palette = sns.color_palette("RdBu_r", n_colors = n_colors)

    # range of correlations
    if max_corr is not None: color_max, color_min = [max_corr * x for x in [1, -1]]
    else: color_max, color_min = [var for var in [1, -1]]

    # nested function for mapping the correlation value to a color
    def value_to_color(val):

        # if outside of correlation scale
        if val >= color_max:
            return palette[-1]
        if val <= color_min:
            return palette[0]

        # position of value in the input range, relative to the length of the input range
        val_position = float((val - color_min)) / (color_max - color_min)
        # target index in the color palette
        ind = int(val_position * (n_colors - 1))

        return palette[ind]

    # take nans as zero correlations
    corr.replace(to_replace = {np.NaN: 0}, inplace = True)

    # add color column
    # arg_c = value_to_color_corr(corr)
    corr['color'] = corr['value'].apply(value_to_color)

    ax.scatter(x = x.map(x_to_num),
               y = y.map(y_to_num),
               s = size * size_scale,
               c = corr['color'],
               marker = 's')

    # put correlation annotations in each cell
    for index_annot in range(corr.shape[0]):

        # default
        color = 'black'

        if size.at[index_annot] == 1:
            # temps
            annot_fontsize_temp = annot_fontsize
            reduce_y_shift_by = 0

            # used for displaying percentages
            if not show_point:
                if corr['value'].at[index_annot] >= 1:
                    annotation = str(round(corr['value'].at[index_annot], 2)).replace('.', '')
                    if len(annotation) == 2: annotation += '0'
                    annot_fontsize_temp = annot_fontsize - 4
                    if corr['value'].at[index_annot] >= annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'

                elif corr['value'].at[index_annot] > 0:
                    annotation = str(round(corr['value'].at[index_annot], 2))[2:]
                    if corr['value'].at[index_annot] >= annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'

                elif corr['value'].at[index_annot] == 0:
                    annotation = '00'

                else:
                    annotation = str(round(corr['value'].at[index_annot], 2))[3:]
                    if corr['value'].at[index_annot] <= -annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'

                if len(annotation) == 1:
                    annotation += '0'

            # for positive floats
            else:

                if corr['value'].at[index_annot] >= 0:

                    annotation = str(round(corr['value'].at[index_annot], 2))[1:]
                    if corr['value'].at[index_annot] >= annot_white_treshold:
                        color = 'white'
                    else:
                        color = 'black'

                # negative floats
                else:

                    annotation = str(round(corr['value'].at[index_annot], 2))[2:]
                    if corr['value'].at[index_annot] <= -annot_white_treshold:
                        color = 'w'
                    else:
                        color = 'black'

                if len(annotation) == 2:
                    annotation += '0'

            # annotate the point on the scatter plot
            ax.annotate(text = annotation,
                        fontstyle = 'normal', fontsize = annot_fontsize_temp,
                        fontweight = 'ultralight', c = color,
                        xy = (x.map(x_to_num).at[index_annot] - annot_shift_x,
                              y.map(y_to_num).at[index_annot] - annot_shift_y))

    # Show column labels on the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation = 55, ha = 'right', fontsize = xtickslabels_fontsize)
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels([v.replace(' min window', '') for v in y_labels], fontsize = yticklabels_fontsize)

    plt.subplots_adjust(top = .93, bottom = .07, left = .06, right = .94, hspace = .3, wspace = .25)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path.replace('SD1/SD2', 'SD1divSD2').replace('(‰)', ''))
    else:
        plt.show()

    plt.clf()
    plt.close('all')

#____________________________________________________________________________________________________________________________________


def bar_plot_vertical_short(corr_limit: float,
                            plot_save_path: str = None
                            ) -> None:
    """
    Generate a heatmap barplot showing the scale of the correlations

    :param corr_limit: abs maximum for the correlations
    :param plot_save_path: path for saving the figure. If None, the plot is shown.
    """

    fig, ax = plt.subplots()

    n_colors = 256
    palette = sns.color_palette("RdBu_r", n_colors=n_colors)
    max_corr_val_group = corr_limit

    if max_corr_val_group is not None: color_max, color_min = [max_corr_val_group * x for x in [1, -1]]
    else: color_max, color_min = [var for var in [1, -1]]

    col_x = [0]*len(palette)

    bar_y = np.linspace(color_min, color_max, n_colors)

    bar_height = bar_y[1] - bar_y[0]

    ax.barh(
        y=bar_y,
        width=2,
        left=col_x,
        height=bar_height,
        color=palette,
        linewidth=0,
        align = 'edge'
    )

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.set_xlim(1, 2)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    ax.yaxis.tick_right()

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10000)

    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.06, right=0.2, hspace=0.25, wspace=2.5)
    plt.tight_layout()

    if plot_save_path is None: plt.show()
    else: plt.savefig(plot_save_path)

    plt.show()
    plt.clf()
    plt.close('all')

#____________________________________________________________________________________________________________________________________


def bar_plot_horisontal_long(corr_range: Tuple = None,
                             plot_save_path: str = None
                             ) -> None:
    """
        Generate a horizontal heatmap barplot showing the scale of the correlations

        :param corr_range: range for the correlation color bar plot
        :param plot_save_path: path for saving the figure. If None, the plot is shown.
    """

    plt.rcParams["figure.figsize"] = (1, 10)
    plt.rcParams['ytick.major.pad'] = '30'

    fig, ax = plt.subplots()

    n_colors = 256
    palette = sns.color_palette("RdBu", n_colors = n_colors)

    # module_debugging.print_debug(palette)

    if corr_range is not None:

        corr_range_min, corr_range_max = corr_range
        color_min, color_max = [x * max(abs(corr_range_max), abs(corr_range_min)) for x in [1, -1]]

    else: color_max, color_min = [var for var in [1, -1]]
    col_x = [0] * len(palette)

    bar_y = np.linspace(color_min, color_max, n_colors)
    bar_height = bar_y[1] - bar_y[0]

    height =bar_height

    ax.barh(y = bar_y, width = 2, left = col_x, height = bar_height, color = palette, linewidth = 0, align = 'edge')

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

    ax.set_xlim(1, 2)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    ax.yaxis.tick_right()
    ax.set_yticklabels(np.linspace(min(bar_y), max(bar_y), 3), rotation = 90, ha = 'right', fontsize = 15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(1000)

    plt.subplots_adjust(top = 0.93, bottom = 0.07, left = 0.06, right = 0.2, hspace = 0.25, wspace = 2.5)
    plt.tight_layout()

    if plot_save_path is None: plt.show()
    else: plt.savefig(plot_save_path)

    plt.show()
    plt.clf()
    plt.close('all')


#____________________________________________________________________________________________________________________________________


def statistical_plots_binary__hist_kde_scatter_whiskers(dataframe: pd.DataFrame,
                                                        hrv_feature: str,
                                                        bad_class: str,
                                                        good_class: str,
                                                        hba1c_feature: str = 'HbA1C(%)',
                                                        class_feature: str = 'Class',
                                                        custom_xlim: Tuple = None,
                                                        title: str = None,
                                                        save_path: str = None,
                                                        ) -> None:

    # create figure layout
    fig1, ax = plt.subplots(figsize=(15,9), dpi=310, constrained_layout=False)
    gs = fig1.add_gridspec(2, 2, height_ratios=[5,1])

    # first axes with distribution plot for every class
    ax1 = fig1.add_subplot(gs[0, :-1])

    ax1 = sns.distplot(dataframe[dataframe[class_feature] == bad_class][hrv_feature], color = 'red', rug = False,
                       kde_kws = {'alpha': .7})

    ax1 = sns.distplot(dataframe[dataframe[class_feature] == good_class ][hrv_feature], color = 'green', rug = False,
                       kde_kws = {'alpha': .7})

    # second ax - KDE plots
    ax2 = fig1.add_subplot(gs[0, -1])

    ax2 = sns.kdeplot(
        data = dataframe,
        x = hrv_feature,
        y = hba1c_feature,
        hue = class_feature,
        palette=[sns.color_palette("Greens")[3], sns.color_palette("Reds")[3]],
        shade = False,
        shade_lowest=True,
        cut = 0,
        legend = True,
        alpha = .8,
        bw_adjust = 1
    )

    ax2 = sns.scatterplot(
            data = dataframe,
            x = hrv_feature,
            y = "HbA1C(%)",
            hue = "Class",
            palette=[sns.color_palette("Greens")[3], sns.color_palette("Reds")[3]],
            alpha = 0.7,
            sizes = [6, 6]
    )

    # 3rd and 4th ax - whiskers
    ax3 = fig1.add_subplot(gs[1, 0])
    sns.boxplot(
        data=dataframe,
        x=hrv_feature,
        y=class_feature,
        palette=[sns.color_palette("Greens")[3], sns.color_palette("Reds")[3]],
        notch=True
    )

    ax4 = fig1.add_subplot(gs[1, 1])
    sns.boxplot(
        data=dataframe,
        x=hrv_feature,
        y=class_feature,
        palette=[sns.color_palette("Greens")[3], sns.color_palette("Reds")[3]],
        notch=True
    )

    # limit for the plots - 3 sigma so the outliers do not make the scale too large
    x_limit = (-1, dataframe[hrv_feature].mean() + (4 * dataframe[hrv_feature].std()))
    y_limit = (dataframe[hba1c_feature].min() - .5, dataframe[hba1c_feature].max() + .5)

    if custom_xlim: x_limit = custom_xlim

    ax1.set(xlim=x_limit)
    ax2.set(xlim=x_limit, ylim = y_limit)
    ax3.set(xlim=x_limit)
    ax4.set(xlim=x_limit)

    sns.despine()
    ax.set_axis_off()
    plt.tight_layout()
    if title: plt.title(title)

    if save_path: plt.savefig(save_path)
    else:  plt.show()

    plt.clf()
    plt.close('all')

# ____________________________________________________________________________________________________________________________________


def statistical_plots_unary__hist_kde_scatter_whiskers(dataframe: pd.DataFrame,
                                                       hrv_feature: str,
                                                       hba1c_feature: str = 'HbA1C(%)',
                                                       class_feature: str = 'Class',
                                                       save_path: str = None, ) -> None:

    # create figure layout
    fig1, ax = plt.subplots(figsize = (15, 9), dpi = 310, constrained_layout = False)
    gs = fig1.add_gridspec(2, 2, height_ratios = [5, 1])

    # first axes with distribution plot for every class
    ax1 = fig1.add_subplot(gs[0, :-1])

    ax1 = sns.distplot(dataframe[hrv_feature], color = 'green', rug = False,
                       kde_kws = {'alpha': .7})

    # second ax - KDE plots
    ax2 = fig1.add_subplot(gs[0, -1])

    ax2 = sns.kdeplot(data = dataframe, x = hrv_feature, y = hba1c_feature, hue = class_feature,
                      palette = [sns.color_palette("Greens")[3]],
                      shade = False,
                      shade_lowest = True, cut = 0, legend = True, alpha = .8, bw_adjust = 1)

    ax2 = sns.scatterplot(data = dataframe, x = hrv_feature, y = "HbA1C(%)", hue = "Class",
                          palette = [sns.color_palette("Greens")[3]], alpha = 0.7, sizes = [6, 6])

    # 3rd and 4th ax - whiskers
    ax3 = fig1.add_subplot(gs[1, 0])
    sns.boxplot(data = dataframe, x = hrv_feature, y = class_feature,
                palette = [sns.color_palette("Greens")[3]], notch = True)

    ax4 = fig1.add_subplot(gs[1, 1])
    sns.boxplot(data = dataframe, x = hrv_feature, y = class_feature,
                palette = [sns.color_palette("Greens")[3]], notch = True)

    # limit for the plots - 3 sigma so the outliers do not make the scale too large
    x_limit = (-1, dataframe[hrv_feature].mean() + (4 * dataframe[hrv_feature].std()))
    y_limit = (dataframe[hba1c_feature].min() - .5, dataframe[hba1c_feature].max() + .5)
    ax1.set(xlim = x_limit)
    ax2.set(xlim = x_limit, ylim = y_limit)
    ax3.set(xlim = x_limit)
    ax4.set(xlim = x_limit)

    sns.despine()
    ax.set_axis_off()
    plt.tight_layout()

    if save_path: plt.savefig(save_path)
    else:  plt.show()

    plt.clf()
    plt.close('all')

# ____________________________________________________________________________________________________________________________________

# def custom_line_plot_single(x: Union[pd.Series, np.ndarray],
#                             y: Union[pd.Series, np.ndarray],
#                             styles_dict: Dict = None,
#                             ) -> None:
#
#     fig, ax = plt.subplots(1, 1, dpi = 300)
#
#     if styles_dict:
#         sns.lineplot(x = x, y = y,
#                      sort = False,
#                      lw = 3,
#                      color = 'g',
#                      style = True,
#                      dashes = [(2, 2)],
#                      marker = 'o',
#                      size = 20,
#                      legend = False)
#
#     else:
#
#         sns.lineplot(x = x, y = y,
#                      sort = False, lw = 3, color = 'g',
#                      style = True, dashes = [(2, 2)], marker = 'o', size = 20, legend = False)
#
#     plt.show()


# ____________________________________________________________________________________________________________________________________


def line_plot_quick_help_refactor_soon(df: pd.DataFrame(),
                                       features_column_name: str,
                                       y_column_name: str,
                                       features: List[str],
                                       intervals: List[int] = None,
                                       save_path: str = None
                                       ) -> None:

    # defaults
    if intervals is None: intervals = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]

    plt.rcParams["figure.figsize"] = (10, 7)

    fig, ax = plt.subplots(1, 1, dpi = 300)

    features_dict = dict()

    y_axis = df[y_column_name]
    x_axis = intervals
    # x_axis.reverse()

    i = -1
    for feature_iter in features:
        i += 1
        if i == len(features): break

        features_dict[feature_iter] = df[df[features_column_name] == feature_iter]

        color = 'white'
        if i == 0: color = 'g'
        if i == 1: color = 'b'
        if i == 2: color = 'r'
        if i == 3: color = 'y'
        if i == 4: color = 'darkorange'
        if i == 5: color = 'darkmagenta'
        if i == 6: color = 'deeppink'
        if i == 7: color = 'plum'
        if i == 8: color = 'brown'
        if i == 9: color = 'gray'
        if i == 10: color = 'black'
        if i == 11: color = 'c'

        sns.lineplot(x = x_axis, y = df[df['feature'] == feature_iter][y_column_name], sort = False, lw = 5,
                     style = True, dashes = [(2, 2)], color = color, legend = False, )

    plt.legend(title = 'HRV Parameters', loc = 'upper left', labels = features, prop = {'size': 12})
    plt.grid(True)
    plt.ylabel(y_column_name)
    ax.set(ylim = (0, -1), xlim = (min(intervals) - 1, max(intervals) + 1))
    ax.set_xticks(x_axis)
    ax.set_xticklabels([f'{x}h' for x in x_axis])
    sns.despine(right = True, top = True)
    sns.plotting_context()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


#
# ____________________________________________________________________________________________________________________________________

def map_sns_palette_colors(input_string: str):

    if input_string.lower() == 'g': return 'Greens'
    if input_string.lower() == 'r': return 'Reds'
    if input_string.lower() == 'b': return 'Blues'


def distribution_plots(dataframe: pd.DataFrame,
                       hrv_feature: str,
                       classes: List[Union[str, int]],
                       classes_colors: List[str],
                       hba1c_feature: str = 'HbA1C(%)',
                       class_feature: str = 'Class',
                       draw_shade: bool = True,
                       custom_xlim: Tuple = None,
                       custom_ylim: Tuple = None,
                       draw_kde: bool = True,
                       draw_scatter: bool = True,
                       draw_errorbar: bool = False,
                       draw_fitline: bool = False,
                       fitline_order: int = 1,
                       fig_title: str = None,
                       save_path: str = None,
                       ) -> None:

    plt.rc('legend', fontsize = 18)
    plt.rc('legend', title_fontsize = 18)

    # create figure layout
    fig1, ax = plt.subplots(figsize=(15,9), dpi=310, constrained_layout=False)
    gs = fig1.add_gridspec(2, 2, height_ratios=[5,1])

    # first axes with distribution plot for every class
    ax1 = fig1.add_subplot(gs[0, :-1])


    # sort dataframe to match classes labels passed on as arguments to assure correct color of class elements
    sorted_dataframe = pd.DataFrame()
    for class_i in classes:
        sorted_dataframe =  sorted_dataframe.append(dataframe[dataframe[class_feature].isin([class_i])], ignore_index = True)
    dataframe = sorted_dataframe

    # should be refactored so that classes come in dictionary with colors
    for class_iter in classes:
        ax1 = sns.distplot(
                dataframe[dataframe[class_feature] == class_iter][hrv_feature],
                color = [sns.color_palette(palette_colors)[3]
                         for palette_colors in map(map_sns_palette_colors, classes_colors)][classes.index(class_iter)],
                rug = False,
                kde_kws = {'alpha': .7},
                ax = ax1
        )

    # second ax - KDE plots
    ax2 = fig1.add_subplot(gs[0, -1])

    if len(classes) == 1:
        kde_scatter_kwargs = dict(color = classes_colors[0])
    else:
        kde_scatter_kwargs = dict(
                hue = class_feature,
                palette = [sns.color_palette(palette_colors)[3]
                           for palette_colors in map(map_sns_palette_colors, classes_colors)]
        )

    if draw_kde:
        sns.kdeplot(
            data = dataframe,
            x = hrv_feature,
            y = hba1c_feature,
            shade = draw_shade,
            shade_lowest=False,
            cut = 0,
            legend = True,
            alpha = 1,
            bw_adjust = 1,
            ax = ax2,
            **kde_scatter_kwargs,
        )

    if draw_scatter:

        sns.scatterplot(
            data = dataframe,
            x = hrv_feature,
            y = "HbA1C(%)",
            alpha = 0.3,
            sizes = [6, 6],
            ax = ax2,
            **kde_scatter_kwargs,
        )

        if draw_errorbar:

            plt.errorbar()

    if draw_fitline:

        sns.regplot(
            data = dataframe,
            x = hrv_feature,
            y = "HbA1C(%)",
            scatter = False,
            order = fitline_order,
            ax = ax2,
            x_ci = 'sd',
            color = 'black'
        )

    if len(classes) == 1:
        boxplot_kwargs = dict(color = classes_colors[0])
    else:
        boxplot_kwargs = dict(
                palette = [sns.color_palette(palette_colors)[3]
                           for palette_colors in map(map_sns_palette_colors, classes_colors)]
        )

    # 3rd and 4th ax - whiskers
    ax3 = fig1.add_subplot(gs[1, 0])
    sns.boxplot(
        data=dataframe,
        x=hrv_feature,
        y=class_feature,
        notch=True,
        ax = ax3,
        **boxplot_kwargs,
    )

    ax4 = fig1.add_subplot(gs[1, 1])
    sns.boxplot(
        data = dataframe,
        x = hrv_feature,
        y = class_feature,
        notch = True,
        ax = ax4,
        **boxplot_kwargs,
    )

    # limit for the plots - 3 sigma so the outliers do not make the scale too large
    x_limit = (-1, dataframe[hrv_feature].mean() + (4 * dataframe[hrv_feature].std()))
    y_limit = (dataframe[hba1c_feature].min() - .5, dataframe[hba1c_feature].max() + .5)

    if custom_xlim: x_limit = custom_xlim
    if custom_ylim: y_limit = custom_ylim

    ax1.set(xlim=x_limit)
    ax2.set(xlim=x_limit, ylim = y_limit)
    ax3.set(xlim=x_limit)
    ax4.set(xlim=x_limit)

    axes = [ax1, ax2, ax3, ax4]

    for ax_iter in axes:
        ax_iter.set_ylabel(ax_iter.get_ylabel(), fontsize = 20)
        ax_iter.set_xlabel(ax_iter.get_xlabel(), fontsize = 20)
        ax_iter.xaxis.set_tick_params(labelsize = 14)
        ax_iter.yaxis.set_tick_params(labelsize = 14)

    sns.despine()
    ax.set_axis_off()
    if fig_title: plt.title(fig_title)
    plt.tight_layout()

    if save_path:
        if save_path.endswith('.pdf'): plt.savefig(save_path)
        else: plt.savefig(save_path + '.pdf')
    else:  plt.show()

    plt.clf()
    plt.close('all')





