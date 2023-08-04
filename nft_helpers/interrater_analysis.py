# Code to calculate inter-rater analysis
import numpy as np
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from pandas import DataFrame
from os.path import join


def montine_interrater_analysis(df: DataFrame, save_dir: str = None, fs: int = 16, title_fs: int = 18, bootstrap_n: int = 1000,
                                random_state: int = 93, show_plots: bool = False):
    """Apply inter-rater analysis on Braak staging data as applied in the Montine paper (2016, Alzheimers Dementia). 
    Analysis: (1) compute the weighted Cohen's kappa between pair of annotators, (2) jackknife approach, calculate the 
    average kappa by removing an annotator at a time (repeat by removing case instead), (3) bootstrap approach by 
    sampling cases with replacement to report the 95% CI.
    
    Args:
        df: Braak stages, columns the annotators, rows are the cases.
        save_dir: Directory to save figures and output files.
        fs: Size of font on the axis.
        title_fs: Font size of the title.
        bootstrap_n: Number of iterations to use when calculating the boostrap.
        random_state: Seed the random sampling performed in bootstrap.
        show_plots: If True show the plots.
        
    """
    np.random.seed(random_state)
    annotators = df.columns.tolist()
    
    pair_kappas = {}  # pair kappas
    
    braak_results = "Pair weighted Cohen's kappas:\n"
    for pair in combinations(annotators, 2):
        k = cohen_kappa_score(df[pair[0]], df[pair[1]], weights='quadratic')
        pair_kappas['+'.join(pair)] = k
        braak_results += f'  - {pair[0]} & {pair[1]}: {k:.4f}\n'
        
        
    # Jackknife approach: (1) case
    case_jk_kappas = []
    
    for idx in df.index:
        jk_df = df[df.index != idx]
        jk_kappas = []
        
        for pair in combinations(annotators, 2):
            jk_kappas.append(cohen_kappa_score(jk_df[pair[0]], jk_df[pair[1]], weights='quadratic'))
            
        case_jk_kappas.append(np.mean(jk_kappas))
        
    # Jackknife approach: (2) annotator
    ann_jk_kappas = []
    
    for ann in annotators:
        jk_df = df[[c for c in df.columns if c != ann]]
        jk_kappas = []
        
        for pair in combinations(jk_df.columns, 2):
            jk_kappas.append(cohen_kappa_score(jk_df[pair[0]], jk_df[pair[1]], weights='quadratic'))
            
        ann_jk_kappas.append(np.mean(jk_kappas))
        
    # bootstrap approach to report 95% confidence intervals
    bs_mean_kappas = []
    
    print('  Bootstrap:')
    for _ in tqdm(range(bootstrap_n)):
        bs_df = df.sample(frac=1, replace=True)
        
        # for each bootstrap calculate the average of the weighted pair kappas
        bs_kappas = []
        for pair in combinations(annotators, 2):
            bs_kappas.append(cohen_kappa_score(bs_df[pair[0]], bs_df[pair[1]], weights='quadratic'))
            
        # average the pair kappas and add it
        bs_mean_kappas.append(np.mean(bs_kappas))
        
    # calculate the mean and the 95% confidence interval
    # source: https://www.geeksforgeeks.org/how-to-calculate-confidence-intervals-in-python/
    ci = np.percentile(bs_mean_kappas,[2.5,97.5])
    
    # save the CI to text file
    braak_results += f'\nBoostrap (N={bootstrap_n}) average and 95% CI: {np.mean(bs_mean_kappas)} ({ci[0]:0.4f}-{ci[1]:0.4f})'
        
    # Histogram for pair kappas
    fig, ax = plt.subplots()
    bins = plt.hist(list(pair_kappas.values()), color='w', ec='k', bins=np.arange(0, 1.04, 0.05), lw=1)
    plt.xlim([0, 1])
    plt.yticks(range(0, int(max(bins[0])) + 3))
    
    plt.axvline(x = 0.75, linestyle = 'dotted', linewidth=2, color='k')
    plt.axvline(x = 0.4, linestyle = 'dashed', linewidth=2, color='k')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    
    plt.ylabel('Frequency', fontsize=fs)
    plt.xlabel('Pair Kappas', fontsize=fs)
    ax.tick_params(axis='both', which='both', direction='out', length=10, width=1)
    plt.title('Braak Stage', weight='bold', fontsize=title_fs)
    
    if save_dir is not None:
        plt.savefig(join(save_dir, 'pair-kappas-histogram.png'), bbox_inches='tight', dpi=300)
        
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # scatter plot for jackknife approach
    fig, ax = plt.subplots()
    plt.scatter(range(1, len(case_jk_kappas)+1), case_jk_kappas, facecolors='white', edgecolors='k', lw=2)
    plt.ylim([0, 1])
    plt.xlabel('Case', fontsize=fs)
    plt.ylabel('Average Weighted Kappa', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.title('Jackknife Analysis (Remove Single Cases)', fontsize=title_fs, weight='bold', y=1.05)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', which='both', direction='out', length=10, width=1)
    
    if save_dir is not None:
        plt.savefig(join(save_dir, 'jackknife-case.png'), bbox_inches='tight', dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    fig, ax = plt.subplots()
    plt.scatter(range(1, len(ann_jk_kappas)+1), ann_jk_kappas, facecolors='white', edgecolors='k', lw=2, s=40)
    plt.ylim([0, 1])
    plt.xlabel('Annotator', fontsize=fs)
    plt.ylabel('Average Weighted Kappa', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(np.arange(1, len(ann_jk_kappas)+1), fontsize=fs)
    plt.title('Jackknife Analysis (Remove Single Annotator)', fontsize=title_fs, weight='bold', y=1.05)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', which='both', direction='out', length=10, width=1)
    
    if save_dir is not None:
        plt.savefig(join(save_dir, 'jackknife-annotator.png'), bbox_inches='tight', dpi=300)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    if save_dir is not None:
        with open(join(save_dir, 'braak-stage-agreement-analysis.txt'), 'w') as fh:
            fh.write(braak_results.strip())
