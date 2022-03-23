import numpy as np
import nsga_func as nf
from pymoo.model.problem import Problem


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=5,
                         n_constr=1,
                         xl=np.array([0, 7.5, 7.5]),
                         xu=np.array([2.999, 12, 12]))



    def _evaluate(self, X, out, *args, **kwargs):
        gk = 1
        qk = 3
        X[:, 0] = np.floor(X[:, 0])
        #print(X[:,0])

        f1 = nf.RFaggregator(X[:, 0], gk, qk, X[:, 1], X[:, 2])
        f2 = -nf.spaceQualityRMS(X[:, 1], X[:, 2])

        g1 = f1[0]-1000


        out["F"] = np.column_stack([f1[0], f1[1], f1[2], f1[3], f2])
        out["G"] = np.column_stack([g1])


problem = MyProblem()


###3. Initialization of an Algorithm

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=100,
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
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

from pymoo.decomposition.asf import ASF

decomp = ASF()

i = decomp.do(nF, 1/weights).argmin()

print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))

#floor continuous typology figure to descrete
res.X[:, 0] = np.floor(res.X[:, 0])

#PSEUDO-WEIGHTS

from pymoo.decision_making.pseudo_weights import PseudoWeights

i = PseudoWeights(weights).do(nF)

print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))


from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_problem, get_reference_directions
import matplotlib.pyplot as plt

plot = Scatter(labels=["h", "CO2e", "£", "t", "l_rms"] )
plot.add(F, s=10)
plot.add(F[i], s=30, color="red")
plot.show()



#PCP CHART DEFINITION
a = res.X
a = np.append(a, res.F, axis=1)

from pymoo.visualization.pcp import PCP

plot = PCP(title=("engine31.00.00 nsga2", {'pad': 30}),
           n_ticks=10,
           legend=(True, {'loc': "upper left"}),
           labels=["typology", "lx", "ly", "h", "carbon", "cost", "program", "spaceRMS"]
           )

index_min_h = min(range(len(a[:, 3])), key=a[:, 3].__getitem__)
index_min_carbon = min(range(len(a[:, 4])), key=a[:, 4].__getitem__)
index_min_cost = min(range(len(a[:, 5])), key=a[:, 5].__getitem__)
index_min_program = min(range(len(a[:, 6])), key=a[:, 6].__getitem__)
index_max_space = min(range(len(a[:, 7])), key=a[:, 7].__getitem__)



plot.set_axis_style(color="grey", alpha=1)
plot.add(a, color="grey", alpha=0.3)
plot.add(a[index_min_cost], linewidth=2.5, color='#069AF3', label="cost_min")
plot.add(a[index_min_program], linewidth=2.5, color='#E50000', label="program_min")
plot.add(a[index_max_space], linewidth=2.5, color="c", label="space_max")
plot.add(a[index_min_h], linewidth=2.5, color="orange", label="h_min")
plot.add(a[index_min_carbon], linewidth=2.5, color='#008000', label="carbon_min")

plot.show()


#print("res.X = ", res.X)
#print("res.F = ", res.F)

#print("res.X = ", res.X[i])
#print("res.F = ", res.F[i])


