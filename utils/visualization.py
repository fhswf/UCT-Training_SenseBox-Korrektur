import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from cycler import cycler
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


def plot_learning_curves(session_name, results_dir, eval_metric):
    
    # Settings
    colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
    SMALL_SIZE = 8
    BIGGER_SIZE = 10
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE, direction='out')
    plt.rc('ytick', labelsize=BIGGER_SIZE, direction='out')
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('lines', linewidth=2)
    plt.rcParams['axes.linewidth'] = 0.1    
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='#000000',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')

    # Data
    df = pd.read_csv(os.path.join(results_dir, f"results__{session_name}.csv"), index_col="epoch")

    df_eval_metric = df[[f"val_{eval_metric}", eval_metric]].reindex(columns=[f"val_{eval_metric}", eval_metric])
    df_loss = df[["val_loss", "loss", "loss_l", "loss_u"]].reindex(columns=['val_loss', 'loss', "loss_l", "loss_u"])
    df_lr = df[[col for col in df.columns if "lr" in col]]
    df_list = [df_eval_metric, df_loss, df_lr]
    contains_unlabeled = df_loss['loss_u'].any()

    cfg_axes = {
        0: {"ylabel": eval_metric, "ylimits": [0, 1.5]},
        1: {"ylabel": "Loss", "ylimits": [0, 1.5]},
        2: {"ylabel": "LR", "ylimits": [0, df_lr['lr'].max() * 1.1]},
    }

    # Plots
    fig, axes = plt.subplots(len(cfg_axes))
    fig.suptitle("Learning Curves")
    plt.xlabel("Epoch")
    plt.subplots_adjust(hspace=0.5)   

    for idx in range(len(cfg_axes)):
        if idx == 1:
            # loss plot
            df_list[idx]['val_loss'].plot(ax=axes[idx], color='#EE6666', linestyle='-')
            df_list[idx]['loss_l'].plot(ax=axes[idx], color='#3388bb', linestyle='-')
            if contains_unlabeled:
                df_list[idx]['loss_u'].plot(ax=axes[idx], color='#3388bb', linestyle=':')
            axes[idx].set_ylabel(cfg_axes[idx]["ylabel"])
            axes[idx].set_ylim(cfg_axes[idx]["ylimits"])
            axes[idx].set_xlabel("")
        else:
            df_list[idx].plot(ax=axes[idx])
            axes[idx].get_legend().remove()
            axes[idx].set_ylabel(cfg_axes[idx]["ylabel"])
            axes[idx].set_ylim(cfg_axes[idx]["ylimits"])
            axes[idx].set_xlabel("")

    # Specifics 
    axes[0].legend(['Val', 'Train'])
    if contains_unlabeled:
        axes[1].legend(['Val', 'Train (Labeled)', 'Train (Unlabeled)'])
    else:
        axes[1].legend(['Val', 'Train'])
    axes[2].get_lines()[0].set_color("black")
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.savefig(f"{results_dir}/learning_curves__{session_name}.pdf")

def plot_evaluation_overview(df_eval, results_pre_train, results_post_train, session_name, directory, binwidth=1, include_pdf=False):
    eval_size_days = int(((len(df_eval)*10)/60) // 24)
    
    y_axis_label = "Häufigkeit"
    plt.figure(figsize=(10, 5))

    errors_predictions = abs(df_eval['target'] - df_eval['predictions'])
    errors_sb = abs(df_eval['target'] - df_eval['measurement_sb'])

    # Calculate bin edges with width 1
    min_val = min(
        errors_predictions.min(),
        errors_sb.min()
    )
    max_val = max(
        errors_predictions.max(),
        errors_sb.max()
    )

    bins = np.arange(min_val, max_val + binwidth, binwidth)
    plt.hist(errors_predictions, bins=bins, density=include_pdf, color='blue', alpha=0.6)
    plt.hist(errors_sb, bins=bins, density=include_pdf, color='red', alpha=0.6)

    # PDFs
    if include_pdf:
        y_axis_label = "Häufigkeitsdichte in 1/°C"
        kde_pred = gaussian_kde(errors_predictions)
        x_vals_pred = np.linspace(min(errors_predictions), max(errors_predictions), 1000)
        kde_sb = gaussian_kde(errors_sb)
        x_vals_sb = np.linspace(min(errors_sb), max(errors_sb), 1000)

        plt.plot(x_vals_pred, kde_pred(x_vals_pred), color='blue', linewidth=1, label='PDF (KDE)')
        plt.plot(x_vals_sb, kde_sb(x_vals_sb), color='red', linewidth=1, label='PDF (KDE)')

    plt.axvline(x=2, color='darkgray', linestyle='--', linewidth=1)
    plt.title(f'Histogramm der Abweichungen (Datenmenge: {eval_size_days} Tage, {len(df_eval)} Samples)')
    plt.xlabel('Abweichung in °C')
    plt.ylabel(y_axis_label)
    plt.grid(True)
    plt.xlim(0, 20)
    handles = [
        plt.Line2D([0], [0], color='darkgray', linestyle='--', linewidth=1),
        Patch(facecolor='blue', edgecolor='blue', alpha=0.6),
        Patch(facecolor='red', edgecolor='red', alpha=0.6)
    ]
    plt.legend(handles, ['2°C-Markierung', 'KI-Korrektur', 'SB-Messungen'])

    # Add table with metrics
    summary_data = {
        'SB-Messungen': [
            f"{results_pre_train['MSE']:.2f}",
            f"{results_pre_train['RMSE']:.2f}",
            f"{results_pre_train['MaxError']:.2f}",
            f"{results_pre_train['PoT@2']:.2f}"
        ],
        'KI-Korrektur': [
            f"{results_post_train['MSE']:.2f}",
            f"{results_post_train['RMSE']:.2f}",
            f"{results_post_train['MaxError']:.2f}",
            f"{results_post_train['PoT@2']:.2f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data, index=["MSE", "RMSE", "MaxError", "PoT@2/%"])

    table = plt.table(
        cellText=summary_df.values,
        rowLabels=summary_df.index,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='upper right',
        colWidths=[0.15, 0.15],
        bbox=[0.69, 0.55, 0.3, 0.25]  # [left, bottom, width, height]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.set_zorder(10)

    # Set background and frame
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('gray')
        cell.set_linewidth(0.8)
        if key[0] == 0 or key[1] == -1:
            cell.set_facecolor('lightgray')
        else:
            cell.set_facecolor('white')

    # Set font color for columns: first column red, second blue
    for (row, col), cell in table.get_celld().items():
        if col == 0 and row > 0:
            cell.get_text().set_color('red')
        elif col == 1 and row > 0:
            cell.get_text().set_color('blue')
            
    # Header font colors
    table[0, 0].get_text().set_color('red')
    table[0, 1].get_text().set_color('blue')
    
    plt.savefig(f"{directory}/evaluation_overview__{session_name}.pdf")

def scatter_pseudo_vs_gt(df_results):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_results['true_label'], df_results['pseudo_label'], c=df_results['max_similarity'], cmap='viridis', alpha=0.7)
    plt.plot([df_results['true_label'].min(), df_results['true_label'].max()], [df_results['true_label'].min(), df_results['true_label'].max()], 'r--')
    plt.xlabel('True Labels')  
    plt.ylabel('Pseudo Labels')
    plt.title('Pseudo Labels vs True Labels')
    scatter.set_clim(0, 1)
    cb = plt.colorbar(scatter, label='Max Similarity')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.mappable.set_clim(0, 1)
    plt.show()

def scatter_similarity_vs_error(df_results):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_results['max_similarity'], df_results['abs_error'], c=df_results['true_label'], cmap='viridis', alpha=0.7)
    plt.xlabel('Max Similarity')
    plt.ylabel('Absolute Error')
    plt.title('Max Similarity vs Absolute Error')
    plt.colorbar(scatter, label='True Labels')
    plt.show()

def hist_error_distribution(df_results):
    plt.figure(figsize=(10, 6))
    plt.hist(df_results['abs_error'], bins=50)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.show()

# plot true labels vs abs error 
def scatter_true_label_vs_abs_error(df_results):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_results['true_label'], df_results['abs_error'], c=df_results['max_similarity'], cmap='viridis', alpha=0.7)
    plt.xlabel('True Labels')
    plt.ylabel('Absolute Error')
    plt.title('True Labels vs Absolute Error')
    scatter.set_clim(0, 1)
    cb = plt.colorbar(scatter, label='Max Similarity')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.mappable.set_clim(0, 1)
    plt.show()

def plot_interval_results(interval_results):
    plt.figure(figsize=(12, 6))
    intervals = [pd.to_datetime(res['From_labeled']).strftime('%Y-%m-%d') for res in interval_results]
    mses = [res['mse'] for res in interval_results]

    x = list(range(len(intervals)))
    plt.plot(x, mses, marker='o')

    step = 10
    if len(intervals) <= step:
        tick_pos = x
        tick_labels = intervals
    else:
        tick_pos = x[::step]
        tick_labels = [intervals[i] for i in tick_pos]

    plt.xticks(tick_pos, tick_labels, rotation=45)
    plt.xlabel('Time Intervals')
    plt.ylabel('MSE')
    plt.title('MSE over Time Intervals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()