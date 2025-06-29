import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from torch.utils.data import DataLoader, Dataset
from problem import get_problem
from model1 import MultiHeadParetoSetModel
from Algorithms.DN_NSGA2 import DN_NSGA2_Survival
import schedulefree
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import timeit
from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist

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
# ins_list = ['sympart', 'omnitest', 'polygons']

# number of independent runs
n_run = 30
# number of learning steps
n_steps = 600
# number of sampled preferences per step
n_pref_update = 5
# number of sampled solutions for gradient estimation
n_sample = 5
# number of iterations for DN-NSGA-Ⅱ
nsga_generations = 100
# population size per generation
pop_size = 100
# number of Pretraining epochs
pretrain_epochs = 20
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

    if n_obj == 2:
        n_pref_update = 5
        nsga_generations = 150     # 50%
        n_steps = 600
    else:
        n_pref_update = 8
        nsga_generations = 300     # 50%
        n_steps = 750


    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):

        print(run_iter)

        start = timeit.default_timer()
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.termination import get_termination

        algorithm = NSGA2(
            pop_size=100,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=1.0, eta=20),
            mutation=PM(prob=1/n_dim, eta=20),
            survival=DN_NSGA2_Survival(niche_eps=0.1),
            eliminate_duplicates=True
        )

        res = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", nsga_generations),
            save_history=True,
            verbose=True
        )
        print('number of solutions:', res.F.shape[0])

        nadir_point = np.max(res.F, axis=0)
        ideal_point = np.min(res.F, axis=0)
        #-------------------------------auto-detect-------------------------
        X_std = StandardScaler().fit_transform(res.X)

        db = DBSCAN(eps=0.3, min_samples=5).fit(X_std)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"自动识别出的簇数：{n_clusters}")

        cluster_data = []
        for cluster_id in range(n_clusters):
            idx = labels == cluster_id
            X_cluster = res.X[idx]
            F_cluster = res.F[idx]
            cluster_data.append((X_cluster, F_cluster))
        for i, (Xc, Fc) in enumerate(cluster_data):
            plt.scatter(Xc[:, 0], Xc[:, 1], label=f'Pareto Set {i + 1}')
        plt.title("Auto-Detected Pareto Sets in Decision Space")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid(True)
        # plt.show()

        psmodel = MultiHeadParetoSetModel(n_dim = n_dim, n_obj = n_obj, n_sets = n_clusters)
        psmodel.to(device)

        ps = np.empty((0, n_dim))
        ps_pre = np.empty((0, n_dim))

        all_prefs = []
        all_X = []
        all_ps_ids = []
        min_var = np.empty((0, Xc.shape[1]))
        max_var = np.empty((0, Xc.shape[1]))
        for i, (Xc, Fc) in enumerate(cluster_data):
            approx_pareto_front = (Fc - ideal_point) / (nadir_point - ideal_point)
            approx_pareto_front += 0.1

            min_temp = 0.8 * np.min(Xc, axis=0)
            max_temp = 1.2 * np.max(Xc, axis=0)
            min_var = np.vstack((min_var, min_temp))
            max_var = np.vstack((max_var, max_temp))
            approx_pareto_set = (Xc - min_temp) / (max_temp - min_temp)

            prefs = approx_pareto_front / approx_pareto_front.sum(axis=1, keepdims=True)
            all_prefs.append(prefs)
            all_X.append(approx_pareto_set)
            all_ps_ids.append(np.full(prefs.shape[0], i))

        all_prefs_tensor = torch.tensor(np.vstack(all_prefs), dtype=torch.float32, device=device)
        all_X_tensor = torch.tensor(np.vstack(all_X), dtype=torch.float32, device=device)
        all_ps_ids_tensor = torch.tensor(np.concatenate(all_ps_ids), dtype=torch.long, device=device)

        class PreTrainDataset(Dataset):
            def __init__(self, prefs, x, ids):
                self.prefs = prefs
                self.x = x
                self.ids = ids

            def __len__(self):
                return len(self.prefs)

            def __getitem__(self, idx):
                return self.prefs[idx], self.x[idx], self.ids[idx]


        pretrain_dataset = PreTrainDataset(all_prefs_tensor, all_X_tensor, all_ps_ids_tensor)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=128, shuffle=True)

        optimizer_p = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025,
                                                     warmup_steps=10)

        # Pretrain steps
        loss_history = []
        for pretrain_epoch in range(pretrain_epochs):
            psmodel.train()
            optimizer_p.train()
            for batch_prefs, batch_X, batch_id in pretrain_loader:
                pred_X = psmodel(batch_prefs, batch_id)
                loss = torch.mean((pred_X - batch_X) ** 2)
                optimizer_p.zero_grad()
                loss.backward()
                loss_history.append(loss.item())
                optimizer_p.step()

        optimizer = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025,
                                                   warmup_steps=10)

        for t_step in range(n_steps):
            psmodel.train()
            optimizer.train()

            sigma = 0.01

            # sample n_pref_update preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha, n_pref_update)
            pref_vec = torch.tensor(pref).to(device).float()

            ids = torch.randint(low=0, high=n_clusters, size=(n_pref_update,), device=device)
            x = psmodel(pref_vec, ids)

            grad_es_list = []
            for k in range(pref_vec.shape[0]):
                i_k = ids[k].item()

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
                x_plus_delta = x_plus_delta.detach().cpu().numpy() * (max_var[i_k] - min_var[i_k]) + min_var[i_k]
                value_plus_delta = problem.evaluate(x_plus_delta)

                value_plus_delta = (value_plus_delta - ideal_point) / (
                            nadir_point - ideal_point)
                value_plus_delta = torch.from_numpy(value_plus_delta).float().to(device)

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
            psmodel(pref_vec, ids).backward(grad_es)
            optimizer.step()

        #---------------------------------small solution set-------------------------------------
        if n_obj == 2:
            pref = np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T
        if n_obj == 3:
            pref_size = 105
            pref = das_dennis(13, 3)

        pref = torch.tensor(pref).to(device).float()

        for i in range(n_clusters):
            ps_ids = torch.full((pref.shape[0],), i, dtype=torch.long, device=device)
            sol = psmodel(pref, ps_ids)
            generated_ps = sol.detach().cpu().numpy() * (max_var[i] - min_var[i]) + min_var[i]
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
                    label='EAPSL',
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
                    f"pareto_front_small_{test_ins}_EAPSL_{run_iter}.pdf",
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
                    label='EAPSL',
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
                    f"pareto_set_small_{test_ins}_EAPSL_{run_iter}.pdf",
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
                    label='EAPSL',
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
                    f"pareto_front_small_{test_ins}_EAPSL_{run_iter}.pdf",
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
                    label='EAPSL',
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
                    f"pareto_set_small_{test_ins}_EAPSL_{run_iter}.pdf",
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
                label='EAPSL',
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
                f"pareto_front_small_{test_ins}_EAPSL_{run_iter}.pdf",
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
                label='EAPSL',
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
                f"pareto_set_small_{test_ins}_EAPSL_{run_iter}.pdf",
                format='pdf',
                dpi=600,  
                bbox_inches='tight',  
                pad_inches=0.6, 
                transparent=False  
            )

            # plt.show()

        # ---------------------------------large solution set-------------------------------------
        if n_obj == 2:
            pref = np.stack([np.linspace(0, 1, 1000), 1 - np.linspace(0, 1, 1000)]).T
        if n_obj == 3:
            pref_size = 990
            pref = das_dennis(43, 3)
        pref = torch.tensor(pref).to(device).float()

        for i in range(n_clusters):
            ps_ids = torch.full((pref.shape[0],), i, dtype=torch.long, device=device)
            sol = psmodel(pref, ps_ids)
            generated_ps = sol.detach().cpu().numpy() * (max_var[i] - min_var[i]) + min_var[i]
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
                    label='EAPSL',
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
                    f"pareto_front_large_{test_ins}_EAPSL_{run_iter}.pdf",
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
                    label='EAPSL',
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
                    f"pareto_set_large_{test_ins}_EAPSL_{run_iter}.pdf",
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
                    label='EAPSL',
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
                    f"pareto_front_large_{test_ins}_EAPSL_{run_iter}.pdf",
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
                    label='EAPSL',
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
                    f"pareto_set_large_{test_ins}_EAPSL_{run_iter}.pdf",
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
                label='EAPSL',
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
                f"pareto_front_large_{test_ins}_EAPSL_{run_iter}.pdf",
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
                label='EAPSL',
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
                f"pareto_set_large_{test_ins}_EAPSL_{run_iter}.pdf",
                format='pdf',
                dpi=600, 
                bbox_inches='tight', 
                pad_inches=0.6,  
                transparent=False  
            )

            # plt.show()
