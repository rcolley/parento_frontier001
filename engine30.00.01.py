import numpy as np
import nsga_func as nf
import sectionLibrary as sl
import math
from pymoo.model.problem import Problem

#PART II: Solution set

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=5,
                         n_obj=4,
                         n_constr=3,
                         xl=np.array([0, 0, 350, 0, 0]), #minimum of: index of ply thickness, index of breath, index of depth, spacing, index of beam
                         xu=np.array([5, 10, 600, 49, 1])) #maximum of: index of ply thickness, index of breath, index of depth, spacing, index of beam

    def _evaluate(self, X, out, *args, **kwargs):
        l1 = 2000
        l2 = 2000
        f5 = nf.joist_beam_vibration_level_inc(10000, 8000, l1, l2, X[:, 0], X[:, 1], X[:, 2], 1000, X[:, 4], 11000, 9000, 100, X[:, 3], 1.8, 88, 1.28, 1, 1, 2, 1.5)
        #-------------------------------------(b_floor, l_beam, l1, l2, b_st, h_st, cc, b_board, t_board, E0_mean_ST, E0_mean_fb, m, beamIndex, a, b_BSEN, ke1j, ke1b, ke2b, sides_supp, fw):
        f6 = nf.joist_carbon_inc(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], l1, l2)
        f7 = nf.hybrid_floor_cost(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], l2)

        g1 = -f5[0] + 4
        g2 = f5[0] - 8
        g3 = f5[1] - 3  #acceptance class restriction


        out["F"] = np.column_stack([f5[0], f6[0][0], f7[0][0], f5[3]])
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

print("res.X = ", res.X)
print("res.F = ", res.F)

from pymoo.factory import get_problem, get_reference_directions

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


nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")


#COMPROMISE PROGRAMING
from pymoo.factory import get_problem
F = res.F
weights = np.array([0.1, 0.5, 0.3, 0.1])

from pymoo.decomposition.asf import ASF


decomp = ASF()

i = decomp.do(nF, 1/weights).argmin()

print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))


#PSEUDO-WEIGHTS

from pymoo.decision_making.pseudo_weights import PseudoWeights

i = PseudoWeights(weights).do(nF)

print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))


plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Objective Space")
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
plt.show()


#PCP CHART DEFINITION
a = res.X
a = np.append(a, res.F, axis=1)

from pymoo.visualization.pcp import PCP
#PCP().add(a).show()

plot = PCP(title=("engine30.00.00 nsga2", {'pad': 30}),
           n_ticks=10,
           legend=(True, {'loc': "upper left"}),
           labels=["joist breath", "joist depth", "joist spacing", "steel index", "layers ply", "response factor (R)", "embodied carbon (kgC02e/kg)", "cost ??", "depth mm"]
           )

index_min_R = min(range(len(a[:, 5])), key=a[:, 5].__getitem__)
index_min_EC = min(range(len(a[:, 6])), key=a[:, 6].__getitem__)
index_min_cost = min(range(len(a[:, 7])), key=a[:, 7].__getitem__)
index_min_depth = min(range(len(a[:, 8])), key=a[:, 8].__getitem__)


plot.set_axis_style(color="grey", alpha=1)
plot.add(a, color="grey", alpha=0.3)
plot.add(a[index_min_R], linewidth=2.5, color="orange", label="R_min")
plot.add(a[index_min_EC], linewidth=2.5, color='#008000', label="EC_min")
plot.add(a[index_min_cost], linewidth=2.5, color='#069AF3', label="index_min_cost")
plot.add(a[index_min_depth], linewidth=2.5, color='#000000', label="index_min_depth")
plot.add(a[i], linewidth=2.5, color='#E50000', label="i")

plot.show()


'''REPORTING'''
varibles = res.X[i]
results = res.F[i]
print("varibles = ", varibles)
print("results = ", results)

beamAttInt = math.ceil(varibles[3])
attributes = sl.UCsectionAgregator(beamAttInt)

joist_breath_array = [38, 47, 50, 63, 75, 100]
joist_depth_array = [97, 122, 140, 147, 170, 184, 195, 220, 235, 250, 300]
b = np.take(joist_breath_array, [varibles[0]])
h = np.take(joist_depth_array, [varibles[1]])
t_board = np.where(varibles[4] >= 0.5, 50, 25)
b = b[0]
h = h[0]


print("t_board = ", t_board)
print("breath = ", b)
print("height = ", h)
print("spacing = ", varibles[2])
print(attributes)

#attributes = np.take(sl.UBsectionAgregator, [beamAttInt])
#attributes = sl.UBsectionAgregator(beamAtt[3])


