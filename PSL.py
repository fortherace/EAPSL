import os

from sympy.codegen.fnodes import lbound

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from problem import get_problem
from model import ParetoSetModel
import schedulefree

import timeit
from matplotlib import pyplot as plt


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


ins_list = ['polygons']
# ins_list = ['omnitest']
# ins_list = ['sympart']
# ins_list = ['sympart', 'omnitest']
# number of independent runs
n_run = 30

# number of learning steps
n_steps = 1200
# number of sampled preferences per step
n_pref_update = 5
# number of sampled solutions for gradient estimation
n_sample = 5
# sampling method for evolutionary gradient approximation
sampling_method = 'Bernoulli'

# device
device = 'cpu'
# -----------------------------------------------------------------------------
hv_list = {}

for test_ins in ins_list:
    print(test_ins)

    problem = get_problem(test_ins)
    n_dim = problem.n_var
    n_obj = problem.n_obj
    ub = problem.xu
    lb = problem.xl
    if test_ins == 'omnitest':
        ideal_point = np.ones(2) * (-2.0)
        nadir_point = np.zeros(2)+0.00000001

    if test_ins == 'sympart':
        ideal_point = np.ones(2) * 0.0
        nadir_point = np.ones(2) * 4.0

    if test_ins == 'polygons':
        if n_obj == 2:
            ideal_point = np.ones(2) * 0.0
            nadir_point = np.ones(2) * 2.0
        else:
            ideal_point = np.ones(3) * 0.0
            nadir_point = np.ones(3) * 1.75

    if n_obj == 2:
        n_pref_update = 5
        n_steps = 1200
    else:
        n_pref_update = 8
        n_steps = 1500

    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):

        print(run_iter)

        start = timeit.default_timer()
        # intitialize the model and optimizer
        psmodel = ParetoSetModel(n_dim, n_obj)
        psmodel.to(device)

        # optimizer
        optimizer = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025, warmup_steps=10)

        z = torch.ones(n_obj).to(device)

        for t_step in range(n_steps):
            psmodel.train()
            optimizer.train()

            sigma = 0.01

            # sample n_pref_update preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha, n_pref_update)
            pref_vec = torch.tensor(pref).to(device).float()

            # get the current coressponding solutions
            x = psmodel(pref_vec)

            grad_es_list = []
            for k in range(pref_vec.shape[0]):

                if sampling_method == 'Gaussian':
                    delta = torch.randn(n_sample, n_dim).to(device).double()

                if sampling_method == 'Bernoulli':
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / 0.5
                    delta = delta.to(device).double()

                if sampling_method == 'Bernoulli-Shrinkage':
                    m = np.sqrt((n_sample + n_dim - 1) / (4 * n_sample))
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / m
                    delta = delta.to(device).double()

                x_plus_delta = x[k] + sigma * delta
                delta_plus_fixed = delta
                x_plus_delta[x_plus_delta > 1] = 1
                x_plus_delta[x_plus_delta < 0] = 0
                x_plus_delta = x_plus_delta.detach().cpu().numpy()
                x_plus_delta = x_plus_delta * (ub - lb) + lb
                value_plus_delta = problem.evaluate(x_plus_delta)

                ideal_point_tensor = torch.tensor(ideal_point).to(device)
                nadir_point_tensor = torch.tensor(nadir_point).to(device)
                value_plus_delta = torch.tensor(value_plus_delta).to(device)
                value_plus_delta = (value_plus_delta -ideal_point_tensor) / (nadir_point_tensor - ideal_point_tensor)
                # value_plus_delta = value_plus_delta / ref_point

                # z = torch.min(torch.cat((z.reshape(1, n_obj), value_plus_delta - 0.1)), axis=0).values.data
                z = torch.full((n_obj,), -0.1).to(device)

                # STCH Scalarization
                u = 0.05

                tch_value = u * torch.logsumexp((1 / pref_vec[k]) * torch.abs(value_plus_delta - z) / u, axis=1)
                tch_value = tch_value.detach()

                rank_idx = torch.argsort(tch_value)
                tch_value_rank = torch.ones(len(tch_value)).to(device)
                tch_value_rank[rank_idx] = torch.linspace(-0.5, 0.5, len(tch_value)).to(device)

                grad_es_k = 1.0 / (n_sample * sigma) * torch.sum(
                    tch_value_rank.reshape(len(tch_value), 1) * delta_plus_fixed, axis=0)
                grad_es_list.append(grad_es_k)

            grad_es = torch.stack(grad_es_list)

            # gradient-based pareto set model update
            optimizer.zero_grad()
            psmodel(pref_vec).backward(gradient=grad_es)
            optimizer.step()

        stop = timeit.default_timer()

        psmodel.eval()
        optimizer.eval()

        # ---------------------------------small solution set-------------------------------------
        ps = np.empty((0, n_dim))
        if n_obj == 2:
            pref = np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T
        if n_obj == 3:
            pref_size = 105
            pref = das_dennis(13, 3)

        pref = torch.tensor(pref).to(device).float()
        sol = psmodel(pref)
        generated_ps = sol.detach().cpu().numpy() * (ub - lb) + lb
        ps = np.vstack((ps, generated_ps))
        ps = np.array(ps)
        pf = problem.evaluate(ps)
        if test_ins == 'polygons':
            if n_obj == 3:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    pf[:, 0],
                    pf[:, 1],
                    pf[:, 2],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$f_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$f_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4)) 
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30)  
                ax.view_init(elev=28, azim=30)  

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right', 
                    frameon=True, 
                    framealpha=0.9, 
                    edgecolor='#333333',  
                    fontsize=35,
                    borderpad=0.2,  
                )

                plt.savefig(
                    f"pareto_front_small_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600, 
                    bbox_inches='tight', 
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))

                for poly in problem.polygons:
                    closed_poly = np.vstack([poly, poly[0]]) 
                    plt.plot(*closed_poly.T, '--', color='black', alpha=1.0)

                ax.scatter(
                    ps[:, 0],
                    ps[:, 1],
                    c='tomato', 
                    marker='o', 
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$x_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$x_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4)) 
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30) 

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right',  
                    frameon=True,  
                    framealpha=0.2,  
                    edgecolor='#333333', 
                    fontsize=35,
                    borderpad=0.2, 
                )

                plt.savefig(
                    f"pareto_set_small_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.scatter(
                    pf[:, 0],
                    pf[:, 1],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$f_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$f_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4))  
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30)  

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right', 
                    frameon=True,  
                    framealpha=0.9,  
                    edgecolor='#333333', 
                    fontsize=35,
                    borderpad=0.2,  
                )

                plt.savefig(
                    f"pareto_front_small_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600, 
                    bbox_inches='tight', 
                    pad_inches=0.6, 
                    transparent=False  
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))

                for poly in problem.polygons:
                    closed_poly = np.vstack([poly, poly[0]])  
                    plt.plot(*closed_poly.T, '--', color='black', alpha=1.0)

                ax.scatter(
                    ps[:, 0],
                    ps[:, 1],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$x_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$x_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4))  
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30)  

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right', 
                    frameon=True, 
                    framealpha=0.2,  
                    edgecolor='#333333',  
                    fontsize=35,
                    borderpad=0.2,  
                )

                plt.savefig(
                    f"pareto_set_small_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600, 
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()
        else:
            True_PS = problem._calc_pareto_set(n_pareto_points=1000)
            True_PF = problem._calc_pareto_front(n_pareto_points=1000)
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.scatter(
                pf[:, 0],
                pf[:, 1],
                c='tomato',  
                marker='o',  
                s=10,
                linewidths=0.8,
                alpha=0.9,
                label='PSL',
                zorder=2
            )

            ax.scatter(
                True_PF[:, 0],
                True_PF[:, 1],
                color='dodgerblue', 
                marker='o',  
                s=100,
                linewidths=0.8,
                alpha=0.5,
                label='Pareto Front',
                zorder=1
            )

            from matplotlib.ticker import ScalarFormatter

            ax.set_xlabel('$f_1$', fontsize=50, labelpad=20, fontweight='bold')
            ax.set_ylabel('$f_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=40)
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_powerlimits((-3, 4)) 
            ax.yaxis.set_major_formatter(formatter)

            # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
            ax.yaxis.offsetText.set_fontsize(30) 

            ax.grid(
                True,
                linestyle='--',
                linewidth=0.6,
                alpha=0.6,
                color='gray'
            )

            legend = ax.legend(
                loc='upper right',  
                frameon=True,  
                framealpha=0.9, 
                edgecolor='#333333', 
                fontsize=35,
                borderpad=0.2,  
            )

            plt.savefig(
                f"pareto_front_small_{test_ins}_PSL_{run_iter}.pdf",
                format='pdf',
                dpi=600,  
                bbox_inches='tight',  
                pad_inches=0.6,  
                transparent=False  
            )

            # plt.show()

            fig, ax = plt.subplots(figsize=(12, 8))

            ax.scatter(
                ps[:, 0],
                ps[:, 1],
                c='tomato',  
                marker='o',  
                s=10,
                linewidths=0.8,
                alpha=0.9,
                label='PSL',
                zorder=2
            )

            ax.scatter(
                True_PS[:, 0],
                True_PS[:, 1],
                color='dodgerblue',  
                marker='o',  
                s=100,
                linewidths=0.8,
                alpha=0.5,
                label='Pareto Set',
                zorder=1
            )

            from matplotlib.ticker import ScalarFormatter

            ax.set_xlabel('$x_1$', fontsize=50, labelpad=20, fontweight='bold')
            ax.set_ylabel('$x_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=40)
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_powerlimits((-3, 4)) 
            ax.yaxis.set_major_formatter(formatter)

            # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
            ax.yaxis.offsetText.set_fontsize(30)  

            ax.grid(
                True,
                linestyle='--',
                linewidth=0.6,
                alpha=0.6,
                color='gray'
            )

            legend = ax.legend(
                loc='upper right',  
                frameon=True,  
                framealpha=0.2,  
                edgecolor='#333333',  
                fontsize=35,
                borderpad=0.2,  
            )

            plt.savefig(
                f"pareto_set_small_{test_ins}_PSL_{run_iter}.pdf",
                format='pdf',
                dpi=600,  
                bbox_inches='tight',  
                pad_inches=0.6,  
                transparent=False 
            )

            # plt.show()

        # ---------------------------------large solution set-------------------------------------
        ps = np.empty((0, n_dim))
        if n_obj == 2:
            pref = np.stack([np.linspace(0, 1, 1000), 1 - np.linspace(0, 1, 1000)]).T
        if n_obj == 3:
            pref_size = 990
            pref = das_dennis(43, 3)
        pref = torch.tensor(pref).to(device).float()

        pref = torch.tensor(pref).to(device).float()
        sol = psmodel(pref)
        generated_ps = sol.detach().cpu().numpy() * (ub - lb) + lb
        ps = np.vstack((ps, generated_ps))
        ps = np.array(ps)
        pf = problem.evaluate(ps)
        if test_ins == 'polygons':
            if n_obj == 3:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    pf[:, 0],
                    pf[:, 1],
                    pf[:, 2],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$f_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$f_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4))  
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30)  
                ax.view_init(elev=28, azim=30)  

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right',  
                    frameon=True,  
                    framealpha=0.9,  
                    edgecolor='#333333', 
                    fontsize=35,
                    borderpad=0.2,  
                )

                plt.savefig(
                    f"pareto_front_large_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))

                for poly in problem.polygons:
                    closed_poly = np.vstack([poly, poly[0]])  
                    plt.plot(*closed_poly.T, '--', color='black', alpha=1.0)

                ax.scatter(
                    ps[:, 0],
                    ps[:, 1],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$x_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$x_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4))  
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30)  

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right', 
                    frameon=True, 
                    framealpha=0.2,  
                    edgecolor='#333333', 
                    fontsize=35,
                    borderpad=0.2,  
                )

                plt.savefig(
                    f"pareto_set_large_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.scatter(
                    pf[:, 0],
                    pf[:, 1],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$f_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$f_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4))  
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30)  

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right', 
                    frameon=True,  
                    framealpha=0.9, 
                    edgecolor='#333333',  
                    fontsize=35,
                    borderpad=0.2, 
                )

                # 保存图像
                plt.savefig(
                    f"pareto_front_large_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))

                for poly in problem.polygons:
                    closed_poly = np.vstack([poly, poly[0]]) 
                    plt.plot(*closed_poly.T, '--', color='black', alpha=1.0)

                ax.scatter(
                    ps[:, 0],
                    ps[:, 1],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='PSL',
                    zorder=2
                )
                from matplotlib.ticker import ScalarFormatter

                ax.set_xlabel('$x_1$', fontsize=50, labelpad=20, fontweight='bold')
                ax.set_ylabel('$x_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=40)
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_powerlimits((-3, 4)) 
                ax.yaxis.set_major_formatter(formatter)

                # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
                ax.yaxis.offsetText.set_fontsize(30)  

                ax.grid(
                    True,
                    linestyle='--',
                    linewidth=0.6,
                    alpha=0.6,
                    color='gray'
                )

                legend = ax.legend(
                    loc='upper right',  
                    frameon=True,  
                    framealpha=0.2,  
                    edgecolor='#333333', 
                    fontsize=35,
                    borderpad=0.2, 
                )

                plt.savefig(
                    f"pareto_set_large_{test_ins}_PSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()
        else:
            True_PS = problem._calc_pareto_set(n_pareto_points=1000)
            True_PF = problem._calc_pareto_front(n_pareto_points=1000)
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.scatter(
                pf[:, 0],
                pf[:, 1],
                c='tomato', 
                marker='o',  
                s=10,
                linewidths=0.8,
                alpha=0.9,
                label='PSL',
                zorder=2
            )

            ax.scatter(
                True_PF[:, 0],
                True_PF[:, 1],
                color='dodgerblue',  
                marker='o', 
                s=100,
                linewidths=0.8,
                alpha=0.5,
                label='Pareto Front',
                zorder=1
            )

            from matplotlib.ticker import ScalarFormatter

            ax.set_xlabel('$f_1$', fontsize=50, labelpad=20, fontweight='bold')
            ax.set_ylabel('$f_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=40)
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_powerlimits((-3, 4))  
            ax.yaxis.set_major_formatter(formatter)

            # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
            ax.yaxis.offsetText.set_fontsize(30)  

            ax.grid(
                True,
                linestyle='--',
                linewidth=0.6,
                alpha=0.6,
                color='gray'
            )

            legend = ax.legend(
                loc='upper right',  
                frameon=True, 
                framealpha=0.9, 
                edgecolor='#333333',  
                fontsize=35,
                borderpad=0.2,  
            )

            plt.savefig(
                f"pareto_front_large_{test_ins}_PSL_{run_iter}.pdf",
                format='pdf',
                dpi=600, 
                bbox_inches='tight',  
                pad_inches=0.6,  
                transparent=False 
            )

            # plt.show()

            fig, ax = plt.subplots(figsize=(12, 8))

            ax.scatter(
                ps[:, 0],
                ps[:, 1],
                c='tomato', 
                marker='o',  
                s=10,
                linewidths=0.8,
                alpha=0.9,
                label='PSL',
                zorder=2
            )

            ax.scatter(
                True_PS[:, 0],
                True_PS[:, 1],
                color='dodgerblue', 
                marker='o', 
                s=100,
                linewidths=0.8,
                alpha=0.5,
                label='Pareto Set',
                zorder=1
            )

            from matplotlib.ticker import ScalarFormatter

            ax.set_xlabel('$x_1$', fontsize=50, labelpad=20, fontweight='bold')
            ax.set_ylabel('$x_2$', fontsize=50, labelpad=20, rotation=0, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=40)
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_powerlimits((-3, 4)) 
            ax.yaxis.set_major_formatter(formatter)

            # ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
            ax.yaxis.offsetText.set_fontsize(30) 

            ax.grid(
                True,
                linestyle='--',
                linewidth=0.6,
                alpha=0.6,
                color='gray'
            )

            legend = ax.legend(
                loc='upper right',  
                frameon=True, 
                framealpha=0.2,  
                edgecolor='#333333', 
                fontsize=35,
                borderpad=0.2,  
            )

            plt.savefig(
                f"pareto_set_large_{test_ins}_PSL_{run_iter}.pdf",
                format='pdf',
                dpi=600, 
                bbox_inches='tight', 
                pad_inches=0.6, 
                transparent=False 
            )

            # plt.show()
