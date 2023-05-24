import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import torch
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

if __name__ == '__main__':
    kernel_size = 40
    min_loss = 14
    max_scaler = 1
    log_level = 1  #+ len(modules)


    fig = plt.figure(figsize=(6*3, 5*3))#, layout='tight')
    gs = gridspec.GridSpec(3, 3)

    exp_dir = '/fsx/home-mitchellw/experimetns/lm/'


    for k, (task, metric) in enumerate([
        # ('arc_challenge', 'acc_norm'),
        # ('arc_easy', 'acc_norm'),
        # ('boolq', 'acc'),
        # ('copa', 'acc'),
        ('hellaswag', 'acc_norm'),
        ('lambada_openai', 'ppl'),
        # ('piqa', 'acc_norm'),
        # ('triviaqa', 'acc'),
        # ('winogrande', 'acc'),
    ]):
        
        ax = fig.add_subplot(gs[k // 3, k % 3])
        for exp, name in [
            ('200b-rpj-100k-m1b-10-2e-3-0.1-nodes16-v0', 'tiktoken'),
            ('200b-rpj-neox100k-m1b_neox-10-2e-3-0.1-nodes16-v0', 'neox'),
        ]:

            xs, ys = [], []
            for x in range(1, 21):
                eval_file = os.path.join(exp_dir, exp, 'checkpoints', f'eval_epoch_{x}.pt')
                if os.path.exists(eval_file):
                    print(f'Loading {eval_file}')

                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)

                    y = eval_data['results'][task][metric]
                    xs.append(x)
                    ys.append(y)

            ax.plot(xs, ys, label=name, marker='o')



        baseline = '/fsx/home-mitchellw/mpt1bevals.pt'
        with open(baseline, 'r') as f:
            eval_data = json.load(f)
        # draw horizontal line
        y = eval_data['results'][task][metric]
        ax.axhline(y=y, color='k', linestyle='--', label='mosaic baseline')
        ax.scatter(20, y, marker='o', color='k', s=100)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)


        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid()


        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(task, fontsize=12)
        ax.set_xlabel('Checkpoint interval (= 10B tokens)', fontsize=12)
        ax.set_xlim([0, 20])
        if metric == 'ppl':
            ax.set_yscale('log')
    leg = ax.legend( bbox_to_anchor=(-0.75, -0.3), fontsize=12, ncol=1)

    fig.subplots_adjust(
        top=0.95, left=0.07, right=0.9, bottom=0.3, wspace=0.32, hspace=0.4
    )

    plt.savefig('/admin/home-mitchellw/git/lm-evaluation-harness/plots/plot_tokenizer.png', bbox_inches='tight')