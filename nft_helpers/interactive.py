# Interactive widgets.
# Functions:
# - model_bars
# - confusion_matrices
from pandas import DataFrame
from typing import List
from ipywidgets import (
    Dropdown, interactive, interact, ToggleButtons, VBox, HBox, Checkbox
)
from .plot import plot_bars
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def model_bars(df: DataFrame, datasets: List[str], models: List[str], **kwargs):
    """Interactive for viewing model validation results as bar plots.
    
    Args:
        df: Model results.
        datasets: List of datasets.
        models: List of models.
        
    """
    hatches = {'Pre-NFT': '.', 'iNFT': 'O', 'all': '/'}
    colors = {'Pre-NFT': '#1E88E5', 'iNFT': '#D81B60', 'all': '#FFC107'}
    
    # Create widgets.
    dataset_selector = Dropdown(options=datasets, description='Dataset:')
    label_selector = ToggleButtons(
        options=['all', 'Pre-NFT', 'iNFT'], description='Label:', 
        button_style='success'
    )
    
    df = df[df.model.isin(models)]
    
    
    def _select_metric(dataset, label):
        """Select the metric and call the plot bars function."""
        metrics = ['mAP50', 'mAP50-95', 'P', 'R', 'F1 score', 'Precision', 'Recall',
                   'TP', 'FP', 'FN']

        if label == 'all':
            metrics += ['micro F1 score', 'macro F1 score', 'weighted F1 score']
            
            
        def _plot_metrics(metric):
            """Plot barplots for metrics."""
            ylim = None if metric in ('TP', 'FP', 'FN') else [0, 1]

            if label in ('Pre-NFT', 'iNFT') and \
               metric in ('F1 score', 'Precision', 'Recall','TP', 'FP', 'FN'):
                metric = f'{metric} ({label})'

            # Check that the plot is not empty
            plot_df = df[
                (df.dataset == dataset) & \
                (df.label == label)
            ]

            kkwargs = kwargs.copy()
            
            kkwargs['hatch'] = hatches[label]
            kkwargs['color'] = colors[label]
                        
            ax = plot_bars(
                plot_df,
                'model',
                metric,
                order=models,
                x_tick_rotation=90,
                hide_xlabel=True,
                y_lim=ylim,
                title=f'Model Performance on \"{dataset}\" Dataset\n({label} class)',
                **kkwargs
            )

            for i in ax.containers:
                ax.bar_label(i, fmt='%.2f', padding=20)

            plt.show()


        _ = interact(
            _plot_metrics, 
            metric=ToggleButtons(options=metrics, description='Metrics:',
                                 button_style='info')
        )
    
    
    w = interactive(_select_metric, dataset=dataset_selector, label=label_selector)

    gui = VBox([
        HBox([dataset_selector, label_selector]),
        w.children[-1]
    ])

    display(gui)


def confusion_matrices(df: DataFrame, datasets: List[str], models: List[str]):
    """Interactive for confusion matrices for models / dataset pairs.
    
    Args:
        df: Model results.
        datasets: List of datasets.
        models: List of models.
        
    """
    # Create widgets.
    model_selector = Dropdown(options=models, description='Model:')
    dataset_selector = Dropdown(options=datasets, description='Dataset:')
    single_cls_check = Checkbox(value=False, description='Single Class')
    
    df = df[df.model.isin(models)]

    def _plot_cms(model, dataset, single_cls):
        """Plot confusion matrices."""
        # Get the models for this model name.    
        model_df = df[
            (df.model == model) & (df.label == 'all') & (df.dataset == dataset)
        ].sort_values(by='split').reset_index(drop=True)

        cms = []

        fig = plt.figure(figsize=(15, 5))

        for i, r in model_df.iterrows():
            cm = np.array(r.cm)
            labels = ['Background', 'Pre-NFT', 'iNFT']

            if single_cls:
                labels = ['Background', 'NFT']
                cm[0, 1] = cm[0, 1:].sum()
                cm[1, 1] = cm[1:, 1:].sum()
                cm[1, 0] = cm[1:, 0].sum()
                cm = cm[:2, :2]

            cm = DataFrame(data=cm, columns=labels, index=labels)

            fig.add_subplot(1, 3, i+1)
            ax = sns.heatmap(
                cm, cmap='viridis', annot=True, cbar=False, fmt=".0f", square=True, 
                linewidths=1, linecolor='black', annot_kws={"size": 18}
            )
            ax.xaxis.set_ticks_position("none")
            ax.yaxis.set_ticks_position("none")

            if i == 0:
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
                plt.ylabel('True', fontsize=18, fontweight='bold')
            else:
                plt.tick_params(left=False, labelleft = False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=14)

            plt.xlabel('Pred', fontsize=18, fontweight='bold')
            plt.title(f'Split {r.split}', fontsize=18, fontweight='bold')

        plt.suptitle(f'Train / Val Splits for model {model}', fontsize=18, 
                     fontweight='bold')
        plt.show()

    w = interactive(_plot_cms, model=model_selector, dataset=dataset_selector,
                    single_cls=single_cls_check)


    display(VBox([
        model_selector,
        HBox([dataset_selector, single_cls_check]),
        w.children[-1]
    ]))
