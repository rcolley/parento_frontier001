import numpy as np
import math

b = 200
h = 300
dia = 10 #varible


'''edge distances'''
a1_para = 5*dia
a1_perp = 4*dia
a2 = 4*dia
a3_tens = 7*dia
a3_comp = 7*dia
a4_tens = 4*dia
a4_comp = 3*dia

'''space calcs'''
#pin beam to col
h_pbc = h-a4_tens-a4_comp
b_pbc = b-(a3_tens*2)

no_fix_height = np.floor(h_pbc/a2)
no_fix_breath = np.floor(b_pbc/a1_para)

print("no_fix_height = ", no_fix_height)
print("no_fix_breath = ", no_fix_breath)

