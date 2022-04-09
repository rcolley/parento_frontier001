import numpy as np
import nsga_func000 as nf
from pymoo.model.problem import Problem
import sectionLibrary as SL


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=6,
                         n_obj=4,
                         n_constr=1,
                         xl=np.array([0, 0, 0, 0, 100, 0]),
                         xu=np.array([5, 0.1, 2, 3, 210, 179]))

    #                      (gk, qk, lx, ly, no_sec, deck_type, conc_type, deck_t, slab_h, fire_t, bm_index)
    # (gk, qk, lx, ly,
    # no_sec(range 0 to 5.0 ceiling'd), deck_type(0=51+), conc_type (range 0 to 2, <1=NW), deck_t (range 0 to 3, <1=0.9, <2=1.0, <3=1.2), slab_h (range 100 to 210, floored to 10mm inc), fire_t(60, 90, 120), bm_index(0-179))
    # no_sec, deck_type, conc_type, deck_t, slab_h, bm_index
    def _evaluate(self, X, out, *args, **kwargs):
        gk = 0.85
        qk = 3.5
        lx = 7.5
        ly = 9
        typology = 1

        f1 = nf.detailed_composite_sec(gk, qk, lx, ly, X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], 60, X[:, 5])
        #f1 = nf.RFaggregator(X[:, 0], gk, qk, X[:, 1], X[:, 2])
        #f2 = -nf.spaceQualityRMS(X[:, 1], X[:, 2])
        f2 = nf.embodied(f1[2], f1[3], f1[4], f1[5], f1[6])
        f3 = nf.costs(f1[2], f1[3], f1[4], f1[5], f1[6])
        f4 = nf.program(typology, f1[2], f1[3], f1[4], f1[5], f1[6])

        g1 = f1[0]-1

        '''
        return gate000, h, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb
        print(f1[1])
        print(f1[2])
        print(f1[3])
        print(f1[4])
        print(f1[5])
        print(f1[6])
        '''

        out["F"] = np.column_stack([f1[1], f2[0], f3[0], f4])
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
weights = np.array([0.25, 0.25, 0.25, 0.25])

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

plot = Scatter(labels=["h", "CO2e", "£", "t"])
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
           labels=["no. sec", "deck_type", "conc_type", "deck_t", "slab_h", "bm_index", "h", "carbon", "cost", "program"]
           )
# no_sec, deck_type, conc_type, deck_t, slab_h, bm_index
index_min_h = min(range(len(a[:, 5])), key=a[:, 5].__getitem__)
index_min_carbon = min(range(len(a[:, 6])), key=a[:, 6].__getitem__)
index_min_cost = min(range(len(a[:, 7])), key=a[:, 7].__getitem__)
index_min_program = min(range(len(a[:, 8])), key=a[:, 8].__getitem__)




plot.set_axis_style(color="grey", alpha=1)
plot.add(a, color="grey", alpha=0.3)
plot.add(a[index_min_cost], linewidth=2.5, color='#069AF3', label="cost_min")
plot.add(a[index_min_program], linewidth=2.5, color='#E50000', label="program_min")
plot.add(a[index_min_h], linewidth=2.5, color="orange", label="h_min")
plot.add(a[index_min_carbon], linewidth=2.5, color='#008000', label="carbon_min")

plot.show()


#print("res.X = ", res.X)
#print("res.F = ", res.F)

#print("res.X = ", res.X[i])
#print("res.F = ", res.F[i])


