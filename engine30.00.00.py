
import numpy as np
import nsga_func as nf
from pymoo.model.problem import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=3,
                         xl=np.array([1, 1, 350, 1]),
                         xu=np.array([5, 10, 600, 179]))

    def _evaluate(self, X, out, *args, **kwargs):
        f5 = nf.joist_beam_vibration_level_inc(22500, 7500, 4250, 1, X[:, 0], X[:, 1], X[:, 2], 1000, 25, 11000, 9000, 55, X[:, 3], 1.8, 88, 1.28, 1, 1, 2, 1.5)
        f6 = nf.joist_carbon_inc(X[:, 0], X[:, 1], X[:, 2], X[:, 3])

        g1 = -f5[0] + 4
        g2 = f5[0] - 8
        g3 = 3 - f5[1]  #acceptance class restriction

        '''print("f5[0] = ", f5[0])
        print("f6[0] = ", f6[0])'''


        out["F"] = np.column_stack([f5[0], f6[0]])
        out["G"] = np.column_stack([g1, g2, g3])


problem = MyProblem()


###3. Initialization of an Algorithm

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=50,
    n_offsprings=50,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

###4. Definition of a Termination Criterion

from pymoo.factory import get_termination

termination = get_termination("n_gen", 50)


###5. Optimize
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)



###6. Visualization of Results and Convergence
'''Design space is the illustration of the parameters with areas dipicting valid positions
Objective space is the numerical evalution of the functions, the curve is the minimised outline of the possible solutions
'''
print("res.X = ", res.X)
print("res.F = ", res.F)

from pymoo.factory import get_problem, get_reference_directions

ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=5) * [2, 4, 8, 16]
F = get_problem("dtlz1").pareto_front(ref_dirs)


###PLOTTING OBJECTIVE SPACE

import matplotlib.pyplot as plt

F = res.F
xl, xu = problem.bounds()


fl = F.min(axis=0)
fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)
'''
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
plt.title("Objective Space")
plt.legend()
plt.show()
'''

nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")
'''
plt.figure(figsize=(7, 5))
plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()
'''
#COMPROMISE PROGRAMING
weights = np.array([0.5, 0.5])

from pymoo.decomposition.asf import ASF

decomp = ASF()

i = decomp.do(nF, 1/weights).argmin()

print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))
'''
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Objective Space")
plt.show()
'''
#PSEUDO-WEIGHTS

from pymoo.decision_making.pseudo_weights import PseudoWeights

i = PseudoWeights(weights).do(nF)

print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Objective Space")
plt.show()

#PCP CHART DEFINITION
a = res.X
a = np.append(a, res.F, axis=1)

from pymoo.visualization.pcp import PCP
#PCP().add(a).show()

plot = PCP(title=("engine30.00.00 nsga2", {'pad': 30}),
           n_ticks=10,
           legend=(True, {'loc': "upper left"}),
           labels=["joist breath", "joist depth", "joist spacing", "steel index", "response factor (R)", "embodied carbon (kgC02e/kg)"]
           )

index_min_R = min(range(len(a[:, 4])), key=a[:, 4].__getitem__)
index_min_EC = min(range(len(a[:, 5])), key=a[:, 5].__getitem__)

plot.set_axis_style(color="grey", alpha=1)
plot.add(a, color="grey", alpha=0.3)
plot.add(a[index_min_R], linewidth=2.5, color="orange", label="R_min")
plot.add(a[index_min_EC], linewidth=2.5, color='#008000', label="EC_min")
plot.add(a[5], linewidth=2.5, color='#069AF3', label="sol_5")
plot.add(a[i], linewidth=2.5, color='#E50000', label="i")

plot.show()


print(res.X[i])
print(res.F[i])

#attributes = F[i]
#print(attributes)

