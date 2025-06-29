import numpy as np
from pymoo.core.problem import Problem

def get_problem(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        'polygons': PolygonProblem,
        'omnitest': OmniTest,
        'sympart': SYMPART,
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)



def create_polygons(row, col, distance, M):
    polygons = []
    for i in range(row):
        for j in range(col):
            center = np.array([i * distance, j * distance]) 
            theta = np.linspace(0, 2 * np.pi, M, endpoint=False)  
            radius = 1.0 
            polygon = np.stack([
                center[0] + radius * np.cos(theta),  
                center[1] + radius * np.sin(theta)   
            ], axis=1)  
            polygons.append(polygon) 
    return polygons  


class PolygonProblem(Problem):
    def __init__(self, row=2, col=2, distance=5, M=3, D=2, lower=-50, upper=50):
        self.row = row
        self.col = col
        self.distance = distance
        self.polygons = create_polygons(row, col, distance, M)
        self.M = M  # Number of objectives
        self.D = D  # Number of decision variables

        xl = np.full(D, lower)
        xu = np.full(D, upper)

        super().__init__(n_var=D,
                         n_obj=M,
                         n_constr=0,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        import numpy as np

        try:
            import torch
            is_torch = isinstance(X, torch.Tensor)
        except ImportError:
            is_torch = False

        N = self.row * self.col
        n_samples = X.shape[0]
        dtype = X.dtype
        device = X.device if is_torch else None

        if is_torch:
            inf = torch.tensor(float('inf'), dtype=dtype, device=device)
            pop_obj = inf * torch.ones((n_samples, self.M), dtype=dtype, device=device)
        else:
            pop_obj = np.full((n_samples, self.M), np.inf, dtype=dtype)

        # Loop over all polygons
        for i in range(N):
            polygon = self.polygons[i]  # assumed to be 2D numpy array of shape [n_vertices, 2]
            tiled_vertices = np.tile(polygon, (1, self.D // 2))  # shape [n_vertices, D]

            if is_torch:
                vertices_tensor = torch.tensor(tiled_vertices, dtype=dtype, device=device)
                dists = torch.norm(X[:, None, :] - vertices_tensor[None, :, :], dim=2)
                pop_obj = torch.minimum(pop_obj, dists)
            else:
                dists = np.linalg.norm(X[:, None, :] - tiled_vertices[None, :, :], axis=2)
                pop_obj = np.minimum(pop_obj, dists)

        if is_torch:
            out["F"] = pop_obj.detach().cpu().numpy()
        else:
            out["F"] = pop_obj
    def evaluate_torch(self, X):
        """
        PyTorch-based evaluation for backpropagation training.
        X: torch.Tensor of shape [n_samples, D]
        Returns: torch.Tensor of shape [n_samples, M]
        """
        assert isinstance(X, torch.Tensor), "Input must be a torch.Tensor"
        device = X.device
        dtype = X.dtype

        n_samples = X.shape[0]
        N = self.row * self.col
        pop_obj = torch.full((n_samples, self.M), float('inf'), dtype=dtype, device=device)

        for i in range(N):
            polygon = self.polygons[i]  # shape [n_vertices, 2]
            tiled_vertices = np.tile(polygon, (1, self.D // 2))  # shape [n_vertices, D]
            vertices_tensor = torch.tensor(tiled_vertices, dtype=dtype, device=device)

            dists = torch.norm(X[:, None, :] - vertices_tensor[None, :, :], dim=2)
            pop_obj = torch.minimum(pop_obj, dists)

        return pop_obj  


import pymoo.gradient.toolbox as anp

class OmniTest(Problem):
    """
    The Omni-test problem proposed by Deb in [1].

    Parameters
    ----------
    n_var: number of decision variables

    References
    ----------
    [1] Deb, K., Tiwari, S. "Omni-optimizer: A generic evolutionary algorithm for single and multi-objective optimization"
    """
    def __init__(self, n_var=2):
        assert (n_var >= 2), "The dimension of the decision space should at least be 2!"
        super().__init__(
            n_var=n_var, n_obj=2, vtype=float, xl=np.full(n_var, 0), xu=np.full(n_var, 6)
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F1 = anp.sum(anp.sin(anp.pi * X), axis=1)
        F2 = anp.sum(anp.cos(anp.pi * X), axis=1)
        out["F"] = anp.vstack((F1, F2)).T

    def _calc_pareto_set(self, n_pareto_points=500):
        # The Omni-test problem has 3^D Pareto subsets
        num_ps = int(3 ** self.n_var)
        h = int(n_pareto_points / num_ps)
        PS = np.zeros((num_ps * h, self.n_var))

        candidates = np.array([np.linspace(2 * m + 1, 2 * m + 3 / 2, h) for m in range(3)])
        # generate combination indices
        candidates_indices = [[0, 1, 2] for _ in range(self.n_var)]
        a = np.meshgrid(*candidates_indices)
        combination_indices = np.array(a).T.reshape(-1, self.n_var)
        # generate 3^D combinations
        for i in range(num_ps):
            PS[i * h:i * h + h, :] = candidates[combination_indices[i]].T
        return PS

    def _calc_pareto_front(self, n_pareto_points=500):
        PS = self._calc_pareto_set(n_pareto_points)
        return self.evaluate(PS, return_values_of=["F"])

import pymoo.gradient.toolbox as anp
import numpy as np


from pymoo.core.problem import Problem


class SYMPARTRotated(Problem):
    """
    The SYM-PART test problem proposed in [1].

    Parameters:
    -----------
    length: the length of each line (i.e., each Pareto subsets), default is 1.
    v_dist: vertical distance between the centers of two adjacent lines, default is 10.
    h_dist: horizontal distance between the centers of two adjacent lines, default is 10.
    angle: the angle to rotate the equivalent Pareto subsets counterclockwisely.
        When set to a negative value, Pareto subsets are rotated clockwisely.

    References:
    ----------
    [1] G. Rudolph, B. Naujoks, and M. Preuss, “Capabilities of EMOA to detect and preserve equivalent Pareto subsets”
    """

    def __init__(self, length=1, v_dist=10, h_dist=10, angle=np.pi / 4):
        self.a = length
        self.b = v_dist
        self.c = h_dist
        self.w = angle

        # Calculate the inverted rotation matrix, store for fitness evaluation
        self.IRM = np.array([
            [np.cos(self.w), np.sin(self.w)],
            [-np.sin(self.w), np.cos(self.w)]])

        r = max(self.b, self.c)
        xl = np.full(2, -10 * r)
        xu = np.full(2, 10 * r)

        super().__init__(n_var=2, n_obj=2, vtype=float, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        if self.w == 0:
            X1 = X[:, 0]
            X2 = X[:, 1]
        else:
            # If rotated, we rotate it back by applying the inverted rotation matrix to X
            Y = anp.array([anp.matmul(self.IRM, x) for x in X])
            X1 = Y[:, 0]
            X2 = Y[:, 1]

        a, b, c = self.a, self.b, self.c
        t1_hat = anp.sign(X1) * anp.ceil((anp.abs(X1) - a - c / 2) / (2 * a + c))
        t2_hat = anp.sign(X2) * anp.ceil((anp.abs(X2) - b / 2) / b)
        one = anp.ones(len(X))
        t1 = anp.sign(t1_hat) * anp.min(anp.vstack((anp.abs(t1_hat), one)), axis=0)
        t2 = anp.sign(t2_hat) * anp.min(anp.vstack((anp.abs(t2_hat), one)), axis=0)

        p1 = X1 - t1 * c
        p2 = X2 - t2 * b

        f1 = (p1 + a) ** 2 + p2 ** 2
        f2 = (p1 - a) ** 2 + p2 ** 2
        out["F"] = anp.vstack((f1, f2)).T

    def _calc_pareto_set(self, n_pareto_points=500):
        # The SYM-PART test problem has 9 equivalent Pareto subsets.
        h = int(n_pareto_points / 9)
        PS = np.zeros((h * 9, self.n_var))
        cnt = 0
        for row in [-1, 0, 1]:
            for col in [1, 0, -1]:
                X1 = np.linspace(row * self.c - self.a, row * self.c + self.a, h)
                X2 = np.tile(col * self.b, h)
                PS[cnt * h:cnt * h + h, :] = np.vstack((X1, X2)).T
                cnt = cnt + 1
        if self.w != 0:
            # If rotated, we apply the rotation matrix to PS
            # Calculate the rotation matrix
            RM = np.array([
                [np.cos(self.w), -np.sin(self.w)],
                [np.sin(self.w), np.cos(self.w)]
            ])
            PS = np.array([np.matmul(RM, x) for x in PS])
        return PS

    def _calc_pareto_front(self, n_pareto_points=500):
        PS = self.pareto_set(n_pareto_points)
        return self.evaluate(PS, return_values_of=["F"])


class SYMPART(SYMPARTRotated):
    def __init__(self, length=1, v_dist=10, h_dist=10):
        super().__init__(length, v_dist, h_dist, 0)
