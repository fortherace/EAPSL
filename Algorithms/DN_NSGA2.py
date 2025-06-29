import numpy as np
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from sklearn.cluster import DBSCAN


class DN_NSGA2_Survival(RankAndCrowdingSurvival):

    def __init__(self, niche_eps=0.1):
        super().__init__()
        self.niche_eps = niche_eps

    def _do(self, problem, pop, n_survive, **kwargs):
        F = NonDominatedSorting().do(pop.get("F"), only_non_dominated_front=False)
        survivors = []

        for rank, front in enumerate(F):
            pop[front].set("rank", rank)

            if len(survivors) + len(front) <= n_survive:
                crowding = calc_crowding_distance(pop[front].get("F"))
                pop[front].set("crowding", crowding)
                survivors.extend(front)
            else:
                X = pop[front].get("X")
                clustering = DBSCAN(eps=self.niche_eps, min_samples=1).fit(X)
                labels = clustering.labels_

                selected = []
                for cluster_id in np.unique(labels):
                    idxs = np.where(labels == cluster_id)[0]
                    sub = [front[i] for i in idxs]
                    crowding = calc_crowding_distance(pop[sub].get("F"))
                    pop[sub].set("crowding", crowding)
                    best = sub[np.argmax(crowding)]
                    selected.append(best)
                    if len(survivors) + len(selected) >= n_survive:
                        break

                survivors.extend(selected[:n_survive - len(survivors)])
                break

        return pop[survivors]
