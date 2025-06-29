import os

from sympy.codegen.fnodes import lbound

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from problem import get_problem
from multi_model import ParetoSetModel
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

# ins_list = ['omnitest']
# ins_list = ['sympart']
# ins_list = ['polygons']
ins_list = ['sympart', 'omnitest']
# number of independent runs
n_run = 30

# EPSL
# number of learning steps
n_steps = 300
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
        nadir_point = np.zeros(2) + 0.00000001

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
        n_steps = 134
    else:
        n_pref_update = 8
        n_steps = 375

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

        # EPSL steps
        for t_step in range(n_steps):
            psmodel.train()
            optimizer.train()
            optimizer.zero_grad()

            sigma = 0.01

            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha, n_pref_update)
            pref_vec = torch.tensor(pref, dtype=torch.float32).to(device)

            outputs = psmodel(pref_vec)
            n_heads = len(outputs)
            grad_es_lists = [[] for _ in range(n_heads)]

            for k in range(n_pref_update):
                for head_idx in range(n_heads):
                    xk = outputs[head_idx][k]  

                    if sampling_method == 'Gaussian':
                        delta = torch.randn(n_sample, n_dim).to(device).double()
                    elif sampling_method == 'Bernoulli':
                        delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / 0.5
                        delta = delta.to(device).double()
                    elif sampling_method == 'Bernoulli-Shrinkage':
                        m = np.sqrt((n_sample + n_dim - 1) / (4 * n_sample))
                        delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / m
                        delta = delta.to(device).double()
                    else:
                        raise ValueError("Unsupported sampling method")

                    x_plus_delta = xk.detach() + sigma * delta
                    x_plus_delta = torch.clamp(x_plus_delta, 0.0, 1.0)
                    x_np = x_plus_delta.cpu().numpy() * (ub - lb) + lb
                    value = problem.evaluate(x_np)  # shape: (n_sample, n_obj)
                    value = (value - ideal_point) / (nadir_point - ideal_point)
                    value = torch.tensor(value, dtype=torch.float32).to(device)

                    z = torch.full((n_obj,), -0.1).to(device)
                    u = 0.05
                    stch_loss = u * torch.logsumexp((1.0 / pref_vec[k]) * torch.abs(value - z) / u, dim=1)
                    stch_loss = stch_loss.detach()

                    rank_idx = torch.argsort(stch_loss)
                    rank = torch.ones_like(stch_loss).to(device)
                    rank[rank_idx] = torch.linspace(-0.5, 0.5, len(stch_loss)).to(device)
                    rank -= rank.mean()

                    grad_es_k = 1.0 / (n_sample * sigma) * torch.sum(rank[:, None] * delta, dim=0)
                    grad_es_lists[head_idx].append(grad_es_k)

            grad_es_heads = [torch.stack(grads).to(device).float() for grads in grad_es_lists]

            optimizer.zero_grad()
            for head_idx in range(n_heads):
                outputs[head_idx].backward(gradient=grad_es_heads[head_idx], retain_graph=(head_idx != n_heads - 1))
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
        sols = psmodel(pref)
        ps_list = [(sol.detach().cpu().numpy() * (ub - lb) + lb) for sol in sols]
        if test_ins == 'polygons':
            if n_obj == 3:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                for i, ps in enumerate(ps_list):
                    pf = problem.evaluate(ps)
                    ax.scatter(
                        pf[:, 0],
                        pf[:, 1],
                        pf[:, 2],
                        c='tomato',  
                        marker='o',  
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
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
                    f"pareto_front_small_{test_ins}_IPSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False 
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))

                for i, ps in enumerate(ps_list):
                    ax.scatter(
                        ps[:, 0],
                        ps[:, 1],
                        c='tomato',  
                        marker='o',  
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
                        zorder=2
                    )
                for idx, poly in enumerate(problem.polygons):
                    closed_poly = np.vstack([poly, poly[0]])  
                    if idx == 0:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0, label='Pareto Set Region')
                    else:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0)
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
                    f"pareto_set_small_{test_ins}_IPSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600, 
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                for i, ps in enumerate(ps_list):
                    pf = problem.evaluate(ps)
                    ax.scatter(
                        pf[:, 0],
                        pf[:, 1],
                        c='tomato',  
                        marker='o',  
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
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
                    f"pareto_front_small_{test_ins}_IPSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))
                for i, ps in enumerate(ps_list):
                    ax.scatter(
                        ps[:, 0],
                        ps[:, 1],
                        c='tomato',  
                        marker='o',  
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
                        zorder=2
                    )
                for idx, poly in enumerate(problem.polygons):
                    closed_poly = np.vstack([poly, poly[0]])  
                    if idx == 0:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0, label='Pareto Set Region')
                    else:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0)
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
                    f"pareto_set_small_{test_ins}_IPSL_{run_iter}.pdf",
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
            for i, ps in enumerate(ps_list):
                pf = problem.evaluate(ps)
                ax.scatter(
                    pf[:, 0],
                    pf[:, 1],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='Multi-head PSL' if i == 0 else None,
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
                f"pareto_front_small_{test_ins}_IPSL_{run_iter}.pdf",
                format='pdf',
                dpi=600,  
                bbox_inches='tight',  
                pad_inches=0.6,  
                transparent=False  
            )

            # plt.show()

            
            fig, ax = plt.subplots(figsize=(12, 8))
            for i, ps in enumerate(ps_list):
                ax.scatter(
                    ps[:, 0],
                    ps[:, 1],
                    c='tomato',  
                    marker='o', 
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='Multi-head PSL' if i == 0 else None,
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
                f"pareto_set_small_{test_ins}_IPSL_{run_iter}.pdf",
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
        sols = psmodel(pref)
        ps_list = [(sol.detach().cpu().numpy() * (ub - lb) + lb) for sol in sols]
        if test_ins == 'polygons':
            if n_obj == 3:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                for i, ps in enumerate(ps_list):
                    pf = problem.evaluate(ps)
                    ax.scatter(
                        pf[:, 0],
                        pf[:, 1],
                        pf[:, 2],
                        c='tomato',  
                        marker='o',  
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
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
                    f"pareto_front_large_{test_ins}_IPSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6, 
                    transparent=False 
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))

                for i, ps in enumerate(ps_list):
                    ax.scatter(
                        ps[:, 0],
                        ps[:, 1],
                        c='tomato',  
                        marker='o',  
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
                        zorder=2
                    )
                for idx, poly in enumerate(problem.polygons):
                    closed_poly = np.vstack([poly, poly[0]])  
                    if idx == 0:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0, label='Pareto Set Region')
                    else:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0)
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
                    f"pareto_set_large_{test_ins}_IPSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                for i, ps in enumerate(ps_list):
                    pf = problem.evaluate(ps)
                    ax.scatter(
                        pf[:, 0],
                        pf[:, 1],
                        c='tomato',  
                        marker='o',  
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
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
                    f"pareto_front_large_{test_ins}_IPSL_{run_iter}.pdf",
                    format='pdf',
                    dpi=600,  
                    bbox_inches='tight',  
                    pad_inches=0.6,  
                    transparent=False  
                )

                # plt.show()

                fig, ax = plt.subplots(figsize=(12, 8))

                for i, ps in enumerate(ps_list):
                    ax.scatter(
                        ps[:, 0],
                        ps[:, 1],
                        c='tomato',  
                        marker='o', 
                        s=10,
                        linewidths=0.8,
                        alpha=0.9,
                        label='Multi-head PSL' if i == 0 else None,
                        zorder=2
                    )
                for idx, poly in enumerate(problem.polygons):
                    closed_poly = np.vstack([poly, poly[0]])  
                    if idx == 0:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0, label='Pareto Set Region')
                    else:
                        plt.plot(*closed_poly.T, '--', color='dodgerblue', alpha=1.0)
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
                    f"pareto_set_large_{test_ins}_IPSL_{run_iter}.pdf",
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
            for i, ps in enumerate(ps_list):
                pf = problem.evaluate(ps)
                
                ax.scatter(
                    pf[:, 0],
                    pf[:, 1],
                    c='tomato',  
                    marker='o', 
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='Multi-head PSL' if i == 0 else None,
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
                f"pareto_front_large_{test_ins}_IPSL_{run_iter}.pdf",
                format='pdf',
                dpi=600,  
                bbox_inches='tight',  
                pad_inches=0.6,  
                transparent=False  
            )

            # plt.show()

            fig, ax = plt.subplots(figsize=(12, 8))
            for i, ps in enumerate(ps_list):
                ax.scatter(
                    ps[:, 0],
                    ps[:, 1],
                    c='tomato',  
                    marker='o',  
                    s=10,
                    linewidths=0.8,
                    alpha=0.9,
                    label='Multi-head PSL' if i == 0 else None,
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
                f"pareto_set_large_{test_ins}_IPSL_{run_iter}.pdf",
                format='pdf',
                dpi=600,  
                bbox_inches='tight',  
                pad_inches=0.6,  
                transparent=False  
            )

            # plt.show()
