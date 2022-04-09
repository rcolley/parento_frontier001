import math as math

import numpy as np

import sectionLibrary as sl

'''BASIC SECTION PROPERTIES'''

def secondMomentRec(b, d):
    Iyy = b * d ** 3 / 12 / 1000000000
    return Iyy

def plasticModRec(b, d):
    Wply = b * d ** 2 / 4 / 1000000
    return Wply


'''CONCRETE A&D FUNCTIONS'''

def slab_section_design(M, b, h, c, fck):
    fyd = 500  # MPa
    m = np.absolute(M) / b  # moment per m
    d = 1000 * h - c - 10  # assumes 20mm dia bar
    K = m * 1000 / (d ** 2) / fck
    K_dash = 0.168
    z = np.where(K <= K_dash, d * np.minimum(0.5 * (1 + np.sqrt(1 - (3.53 * K))), 0.95), 0.01)
    As_req = 1000000 * m / 500 / z
    As_lib = np.array([393, 524, 646, 754, 1005, 1340, 1571, 2094, 2513, 3142])
    As_prov = np.where(As_req <= As_lib[0], As_lib[0], np.where(As_req <= As_lib[1], As_lib[1],
                                                                np.where(As_req <= As_lib[2], As_lib[2],
                                                                         np.where(As_req <= As_lib[3], As_lib[3],
                                                                                  np.where(As_req <= As_lib[4],
                                                                                           As_lib[4],
                                                                                           np.where(As_req < As_lib[5],
                                                                                                    As_lib[5], np.where(
                                                                                                   As_req < As_lib[6],
                                                                                                   As_lib[6], np.where(
                                                                                                       As_req < As_lib[
                                                                                                           7],
                                                                                                       As_lib[7],
                                                                                                       np.where(
                                                                                                           As_req <=
                                                                                                           As_lib[8],
                                                                                                           As_lib[8],
                                                                                                           As_lib[
                                                                                                               9])))))))))
    return As_prov

'''BASIC TYPOLOGY FUNCTIONS'''

def basic_fs(gk, qk, L1, L2):
    h_ratio = np.where(qk <= 2.5, 28, np.where(qk <= 5, 26, np.where(qk <= 7.5, 25, 23)))
    h = np.maximum(L1, L2) / h_ratio
    As_ave = 0.03  # 3% guess
    vol_steel = 0
    vol_conc = h  # volume of concrete per m2
    vol_rebar = vol_conc * As_ave
    vol_galv = 0.000
    vol_timb = 0.0
    return h, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb

def basic_comp(gk, qk, L1, L2):
    Lmin = np.minimum(L1, L2)
    Lmax = np.maximum(L1, L2)
    sec_spacing = Lmin / 3
    sw = 3
    UDL_sls = gk + sw + qk
    w_sls = UDL_sls * sec_spacing
    def_max = Lmax / 0.250

    Ireq = 5 / 384 * w_sls * Lmax**4 / 0.210 / def_max * 1000**2
    vol_steel = Ireq * 0.000022 / 1000000 * 4  # assume 3 secondarys 2 half primaries
    vol_conc = 0.130
    vol_rebar = 0.00025
    vol_galv = 0.001
    vol_timb = 0
    h = (130 + Ireq * 0.00000112 *2) / 1000
    return h, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb


def basic_hybrid(gk, qk, L1, L2):
    Lmin = np.minimum(L1, L2)
    Lmax = np.maximum(L1, L2)
    sec_spacing = Lmin / 3
    sw = 1
    UDL_sls = gk + sw + qk
    w_sls = UDL_sls * sec_spacing
    def_max = Lmax / 0.250
    Ireq = 5 / 384 * w_sls * (Lmax ** 4) / 0.21 / def_max * 1000**2
    vol_steel = Ireq * 0.000022 / 1000000 * 4  # assume 3 secondarys 2 half primaries
    vol_conc = 0
    vol_rebar = 0
    vol_galv = 0
    vol_timb = 0.12
    h = (Ireq * 0.00000112*4) / 1000
    return h, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb


'''def aggregator(typology, gk, qk, L1, L2):
    conc = basic_fs(gk, qk, L1, L2)
    comp = basic_comp(gk, qk, L1, L2)
    hybrid = basic_hybrid(gk, qk, L1, L2)
    ans = np.where(typology == 0, conc, np.where(typology == 1, comp, hybrid))
    return ans'''


'''ATTRIBUTE DEFINITION AND AGGREGGATION FUNCTIONS'''

def embodied(vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb):
    ec_steel = 1.21  # kgCO2e - recycled ave
    ec_conc = 0.12  # kgCO2e - 25% GGBS
    ec_rebar = 1.2  # kgCO2e - recycled ave
    ec_galv = 1.52  # kgCO2e - recycled ave
    ec_timb = 0.263  # kgCO2e - non-sequ

    mass_steel = 7850 * vol_steel
    mass_conc = 2500 * vol_conc
    mass_rebar = 7850 * vol_rebar
    mass_galv = 7850 * vol_galv
    mass_timb = 500 * vol_timb

    EC_steel = ec_steel * mass_steel
    EC_conc = ec_conc * mass_conc
    EC_rebar = ec_rebar * mass_rebar
    EC_galv = ec_galv * mass_galv
    EC_timb = ec_timb * mass_timb
    EMBODIED_tot = EC_steel + EC_conc + EC_rebar + EC_galv + EC_timb

    return EMBODIED_tot, EC_steel, EC_conc, EC_rebar, EC_galv, EC_timb

def costs(vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb):
    rate_steel = 4.5  # £/kg
    rate_conc = 0.044  # £/kg
    rate_rebar = 0.5  # £/kg
    rate_galv = 3.18  # £/kg
    rate_timb = 2.28  # £/kg

    mass_steel = 7850 * vol_steel
    mass_conc = 2500 * vol_conc
    mass_rebar = 7850 * vol_rebar
    mass_galv = 7850 * vol_galv
    mass_timb = 500 * vol_timb

    COST_steel = rate_steel * mass_steel
    COST_conc = rate_conc * mass_conc
    COST_rebar = rate_rebar * mass_rebar
    COST_galv = rate_galv * mass_galv
    COST_timb = rate_timb * mass_timb
    COST_tot = COST_steel + COST_conc + COST_rebar + COST_galv + COST_timb

    return COST_tot, COST_steel, COST_conc, COST_rebar, COST_galv, COST_timb

def program(typology, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb):
    conc_mod = 1.2 * np.sqrt(np.sqrt(vol_rebar/0.01)) #graph sesitive to this number - changes duration significantly
    steel_mod = 1
    hybrid_mod = 0.9
    time = np.where(typology == 0, conc_mod, np.where(typology == 1, steel_mod, hybrid_mod))
    return time

def spaceQualityRMS(lx, ly):
    lRMS = np.sqrt(lx**2 + ly**2)
    return lRMS

'''RE-FRAMED TYPOLOGY FUNCTIONS'''

def RFflatslab(typology, gk, qk, lx, ly):
    design = detailed_fs(gk, qk, lx, ly)
    carbon = embodied(design[1], design[2], design[3], design[4], design[5])
    cost = costs(design[1], design[2], design[3], design[4], design[5])
    time = program(typology, design[1], design[2], design[3], design[4], design[5])
    return design[0], carbon[0], cost[0], time

def RFcomposite(typology, gk, qk, lx, ly):
    design = basic_comp(gk, qk, lx, ly)
    carbon = embodied(design[1], design[2], design[3], design[4], design[5])
    cost = costs(design[1], design[2], design[3], design[4], design[5])
    time = program(typology, design[1], design[2], design[3], design[4], design[5])
    return design[0], carbon[0], cost[0], time

def RFhybrid(typology, gk, qk, lx, ly):
    design = basic_hybrid(gk, qk, lx, ly)
    carbon = embodied(design[1], design[2], design[3], design[4], design[5])
    cost = costs(design[1], design[2], design[3], design[4], design[5])
    time = program(typology, design[1], design[2], design[3], design[4], design[5])
    return design[0], carbon[0], cost[0], time

def RFaggregator(typology, gk, qk, lx, ly):
    ans = np.where(typology == 0, RFflatslab(typology, gk, qk, lx, ly), np.where(typology == 1, RFcomposite(typology, gk, qk, lx, ly), RFhybrid(typology, gk, qk, lx, ly)))
    return ans


'''DETAILED TYPOLOGY FUNCTIONS'''

def detailed_composite_sec(gk, qk, lx, ly, no_sec, deck_type, conc_type, deck_t, slab_h, fire_t, bm_index):
    #(gk, qk, lx, ly, no_sec(range 0 to 3.0 ceiling'd), deck_type(0=51+), conc_type (range 0 to 2, <1=NW), deck_t (range 0 to 3, <1=0.9, <2=1.0, <3=1.2), slab_h (range 100 to 210, floored to 10mm inc), fire_t(60, 90, 120))
    '''paras'''
    fck = 30 #MPa
    fy = 355 #MPa
    Ecm = 31
    gammaG = 1.35
    gammaQ = 1.5
    gammaM0 = 1
    gammaM1 = 1
    gammaM2 = 1.1
    gammaV = 1.25 #shear studs
    gammaC = 1.5 #concrete
    gammaS = 1.15 #persistent adn transient design for reinforcements
    xi = 0.925
    psi_0_1_cons = 1
    psi_0_1_comp = 0.7

    '''bay-geometry'''
    no_sec = np.ceil(no_sec)
    Lmin = np.minimum(lx, ly)
    Lmax = np.maximum(lx, ly)
    sec_spacing = Lmin / no_sec


    deck_type=np.floor(deck_type)

    #slab_h = np.where(slab_t < 135, 1, np.where(slab_t < 145, 2, np.where(slab_t < 155, 3, np.where(slab_t < 165, 4, np.where(slab_t < 175, 5, np.where(slab_t < 185, 6, np.where(slab_t < 195, 7, np.where(slab_t < 225, 8, 250))))))))
    slab_h = np.floor(slab_h/10)*10 #slab_h to be in range 130 to 210mm
    slab_wt_wet = np.where(conc_type<1, 2.16+((slab_h-100)*0.025), 1.71+((slab_h-100)*0.02))
    slab_wt_dry = np.where(conc_type<1, 2.12+((slab_h-100)*0.024), 1.62+((slab_h-100)*0.019))
    slab_conc_vol = (slab_h-10)/1000
    deck_attributes = np.array([[0.9, 0.86, 1578, 0.13, 15.9], [1, 0.96, 1762, 0.14, 16.5], [1.2, 1.16, 2137, 0.17, 16.8]])


    deck_index = np.floor(deck_t)
    deck_index = deck_index.astype(int)
    deck_attributes = deck_attributes[deck_index]

    slab_index = np.where(slab_h == 100, 0,np.where(slab_h == 110, 1, np.where(slab_h == 120, 2, np.where(slab_h == 130, 3, np.where(slab_h == 140, 4, np.where(slab_h == 150, 5, np.where(slab_h == 160, 6, np.where(slab_h == 170, 7, np.where(slab_h == 180, 8, np.where(slab_h == 190, 9, 10))))))))))
    max_deck_span_60min_5kPa_1mm = [3.01, 2.94, 2.86, 2.79, 2.73, 2.67, 2.61, 2.56, 2.51, 2.46, 2.42]
    mesh_60min_5kPa_1mm = [142, 142, 142, 193, 193, 252, 252, 252, 393, 393, 393]

    slab_and_deck_wt_wet = slab_wt_wet + np.take(deck_attributes, 3)
    slab_and_deck_wt_dry = slab_wt_dry + np.take(deck_attributes, 3)
    deck_crossSec_area = deck_attributes[:, 2]
    slab_mesh_span = np.take(mesh_60min_5kPa_1mm, slab_index)

    max_deck_span = np.take(max_deck_span_60min_5kPa_1mm, slab_index)

    gate001 = np.where(sec_spacing<=max_deck_span, 0, 1)#deck span check max_deck_span_60min_5kPa_1mm[slab_index]
    gate002 = np.where(fire_t==60, 0, 1)#fire period check

    '''beam selection'''
    bm_index = np.floor(bm_index)
    bm_index = bm_index.astype(int)
    beam_identity = sl.UBnames(bm_index)
    beam_attributes = sl.UBsectionAgregator(bm_index)

    # naturalises beam properties
    sec_name = beam_identity
    sec_G = beam_attributes[0]
    sec_hs = beam_attributes[1]
    sec_b = beam_attributes[2]
    sec_tw = beam_attributes[3]
    sec_tf = beam_attributes[4]
    sec_r = beam_attributes[5]
    sec_hi = beam_attributes[6]
    sec_d = beam_attributes[7]
    sec_Iy = beam_attributes[8]
    sec_Wely = beam_attributes[9]
    sec_Wply = beam_attributes[10]
    sec_iy = beam_attributes[11]
    sec_Avz = beam_attributes[12]
    sec_Iz = beam_attributes[13]
    sec_Welz = beam_attributes[14]
    sec_Wplz = beam_attributes[15]
    sec_iz = beam_attributes[16]
    sec_Ss = beam_attributes[17]
    sec_It = beam_attributes[18]
    sec_Iw = beam_attributes[19]
    sec_Aa = beam_attributes[20]


    '''loads'''
    cll = 1.5 #kPa
    bm_wt = sec_G

    '''combinations'''
    Fd1 = gammaG*(sec_G/sec_spacing + np.take(deck_attributes, 3)) + gammaG*(slab_wt_wet+cll) #construction stage ULS
    Fd = xi*gammaG*(sec_G/sec_spacing + gk + slab_and_deck_wt_dry) + gammaQ*qk #composite stage ULS
    g1 = slab_and_deck_wt_dry + sec_G/sec_spacing #frame selfweight dry
    g2 = gk #superdead
    q1 = qk #live

    '''bm and v'''
    #construction
    M_Ed_sec_cons = Fd1 * Lmax**2 / 8
    V_Ed_sec_cons = Fd1 * Lmax / 2

    #composite
    M_Ed_sec_comp = Fd * Lmax**2 / 8
    V_Ed_sec_comp = Fd * Lmax / 2

    '''cross-section classification'''
    fy = np.where(np.maximum(sec_tf, sec_tw)<=16, fy, np.where(np.maximum(sec_tf, sec_tw)<=40, fy-10, fy-20))
    epsilon = np.sqrt(235/fy)
    c_tf_ratio = 0.5*(sec_b - sec_tw - (2*sec_r))/sec_tf #outstand compression flange
    classification_outstand = np.where(c_tf_ratio <= 9*epsilon, 1, np.where(c_tf_ratio <= 10*epsilon, 2, np.where(c_tf_ratio <= 14*epsilon, 3, 4)))
    c_tw_ratio = sec_d/sec_tw
    classification_internal = np.where(c_tw_ratio <= 72*epsilon, 1, np.where(c_tw_ratio <= 83*epsilon, 2, np.where(c_tw_ratio <= 124*epsilon, 3, 4)))
    classification = np.maximum(classification_internal, classification_outstand)

    '''design resistance'''
    #shear buckling
    eta_vbuck = 1
    gate003 = np.where(sec_hi/sec_tw*xi/epsilon/72<=1, 0, 1)

    #vertical shear resistance
    V_c_Rd = sec_Avz*fy/np.sqrt(3)/gammaM0
    V_uc = V_Ed_sec_comp / V_c_Rd
    gate004 = np.where(V_uc <= 1, 0, 1)

    #bending
    gate005 = np.where(V_uc <= 0.5, 0, 1) #low shear chk
    gate006 = np.where(classification <= 2, 0, 1) #plastic chk
    M_c_Rd = sec_Wply*fy/gammaM0/1000
    M_uc = M_Ed_sec_comp / M_c_Rd
    gate007 = np.where(M_uc <= 1, 0, 1)

    #shear connectors
    '''TO BE REVIEWED FOR ECONOMISE'''
    fu = 450
    stud_dia = 19
    hsc = 95

    gate008 = np.where(hsc/stud_dia >= 4, 0, 1)
    P_Rd_1 = 0.8*fu*np.pi*stud_dia**2 / 4 / gammaV /1000
    alpha_studs = 1
    P_Rd_2 = 0.29*alpha_studs*stud_dia**2 * np.sqrt(fck*Ecm*1000)/ gammaV /1000

    hp = 51 #deck depth
    hsc = 95 #stud height
    nr = 1 #to be developed
    b0 = 110 #concrete between re-entrants
    kt = 0.7*b0/np.sqrt(nr)/hp * ((hsc/hp)-1)
    kt_max = 0.85
    kt = np.minimum(kt, kt_max)
    P_Rd = np.minimum(P_Rd_1, P_Rd_2)*kt
    nrP_Rd = nr*P_Rd

    #degree of shear connection
    gate009 = np.where(hsc >= 4*stud_dia, 0, 1)
    gate010 = np.where(Lmax <= 25, 0, 1)
    eta_degree_min = np.maximum(0.4, 1-(355/fy)*(0.75-(0.03*Lmax/1000)))

    #effective width
    bei = Lmax/8
    b0 = 0
    beff = np.minimum(b0 + 2*bei, sec_spacing)

    #compressive resistance of concrete
    fcd = fck/gammaC
    hc = slab_h-hp
    Ncf = 0.85*fcd*beff*hc

    #tensile reesistance of steel member
    Npl_a = fy*sec_Aa

    #compressive force in flange
    Nc_comp = np.minimum(Ncf, Npl_a)

    #resistance of the shear connectors
    deck_pitch = 150
    no_studs_half = 1000*Lmax/2/deck_pitch
    Nc_studs = no_studs_half*P_Rd_2

    #degree of shear connection
    eta_degree = np.minimum(1, Nc_studs / Nc_comp)
    gate011 = np.where(eta_degree >= eta_degree_min, 0, 1)

    '''design resistance of the cross-section for the composite stage'''
    #vertical shear resistance
    V_uc = V_Ed_sec_cons / V_c_Rd
    gate012 = np.where(V_uc <= 1, 0, 1)

    #resistance in bending
    gate013 = np.where(V_uc/2 <= 1, 0, 1)
    xpl = (Npl_a-Nc_studs)/2/fy/sec_b*1000
    PNA_text = np.where(Npl_a <= Ncf, "PNA lies in top flange", np.where(xpl < sec_tf, "PNA lies in top flange", "PNA lies in web"))
    PNA_index = np.where(Npl_a <= Ncf, 0, np.where(xpl < sec_tf, 1,2))

    '''#PNA in the top flange
    F1a = Nc_studs
    F6b = 2*fy*sec_b*xpl
    F7b = Npl_a
    M_A = 0'''

    #PNA  in the concrete slab
    Mpl_Rd_c = (Npl_a*((sec_hs/2)+slab_h-(Npl_a/Ncf*hc/2)))/1000
    xc = Npl_a/Ncf*hc
    Mpl_Rd_c2 = sec_hs/2 + slab_h - xc/2 #potentially more accurate for PNA in conc

    #PNA  in the top flange
    hd = hp
    Mpl_Rd_f = ((Npl_a*sec_hs/2)+(Ncf*((hc/2)+hd)))/1000

    #PNA  in the web
    Mpl_a_Rd = M_c_Rd
    hw = sec_hs-(2*sec_tf)
    Nw = 0.95*fy*sec_tw*hw
    Mpl_Rd_w = (Mpl_a_Rd + Ncf*((hc+2*hd+sec_hs)/2) - (Ncf**2*sec_hs/Nw/4))/1000

    Mpl_Rd =np.where(PNA_index == 0, Mpl_Rd_c, np.where(PNA_index == 1, Mpl_Rd_f, Mpl_Rd_w))

    #PNA with partitial
    MRd = Mpl_a_Rd + (Mpl_Rd-Mpl_a_Rd)*eta_degree

    '''longitudinal shear resistance of the slab'''
    fyd = 500/gammaS
    hf = hc
    cotTheta = 2.00569
    delta_x = Lmax/0.002
    delta_Fd = Nc_studs/2
    v_Ed = 1000*delta_Fd/hf/delta_x
    At = 1000*v_Ed*hf/fyd/cotTheta
    At_prov = np.where(At<=142, 142, np.where(At<=193, 193, np.where(At<=252, 252, np.where(At<=393, 393, np.where(At<=503, 503, np.where(At<=785, 785,1131))))))
    At_prov = np.maximum(slab_mesh_span, At_prov)

    #concrete flange crushing
    v = 0.6*(1-(fck/250))
    alpha_cc = 0.85
    fcd = alpha_cc*fck/gammaC
    sinTheta = 0.446197813
    cosTheta = 0.894934361
    v_Ed_crush = v * fcd * sinTheta * cosTheta
    long_v_UC = v_Ed / v_Ed_crush
    gate014 = np.where(long_v_UC <=1, 0, 1)

    '''verification at SLS'''
    #short term
    Ea = 210000
    Ec = 38000
    n0 = Ea/Ecm
    psi_l = 1.1 #creep multiplyer
    phi_t = 3 #creep coef
    nL = n0 * (1+(psi_l*phi_t))/1000
    n = 0.333*nL + 0.666*n0
    nd = Ea/Ec #natural frequence
    n_perm = 2*Ea/Ec #natural frequence

    #composite second moment of area
    y_el_perm = ((sec_Aa*sec_hs/2) + (beff*(slab_h-hp)*((sec_hs+(slab_h+hp)/2))/2)/n_perm) / (sec_Aa+(beff*(slab_h-hp)/n_perm))
    Ieq_perm = 10000*sec_Iy + 1000*beff*((slab_h-hp)**3)/12/n_perm + 100*sec_Aa*((y_el_perm-(sec_hs/2))**2) + 1000*beff*(slab_h-hp)/n_perm*((sec_hs+((slab_h+hp)/2)-y_el_perm)**2)

    Ieq1 = 10000*sec_Iy
    Ieq2 = 1000*beff*((slab_h-hp)**3)/12/n_perm
    Ieq3 = 100*sec_Aa*((y_el_perm-(sec_hs/2))**2)
    Ieq4 = 1000*beff*(slab_h-hp)/n_perm*((sec_hs+((slab_h+hp)/2)-y_el_perm)**2)

    #long term
    # def_gk = 5/384 * gk*sec_spacing * (Lmax*1000)**4 / Ea / Ieq_perm
    def_qk = 5 / 384 * qk * sec_spacing * (Lmax * 1000) ** 4 / Ea / Ieq_perm
    def_comp = 5 / 384 * Fd * sec_spacing * (Lmax * 1000) ** 4 / Ea / Ieq_perm
    def_lim_qk_sec = Lmax*1000/360
    def_lim_sls_sec = Lmax * 1000 / 250
    def_qk_sec_UC = def_qk / def_lim_qk_sec
    def_sls_sec_UC = def_comp / def_lim_sls_sec
    gate015 = np.where(def_qk_sec_UC <=1, 0, 1)
    gate016 = np.where(def_sls_sec_UC <= 1, 0, 1)

    #short term / dynamic
    y_el_dyn = ((sec_Aa*sec_hs/2) + (beff*(slab_h-hp)*((sec_hs+(slab_h+hp)/2))/2)/nd) / (sec_Aa+(beff*(slab_h-hp)/nd))
    Ieq_dyn = 10000*sec_Iy + 1000*beff*((slab_h-hp)**3)/12/nd + 100*sec_Aa*((y_el_dyn-(sec_hs/2))**2) + 1000*beff*(slab_h-hp)/nd*((sec_hs+((slab_h+hp)/2)-y_el_dyn)**2)
    def_dyn = 5 / 384 * (g1+g2+0.1*q1) * sec_spacing * (Lmax * 1000) ** 4 / Ea / Ieq_dyn
    fn = 18.07/np.sqrt(def_dyn)
    fn_UC = 4.5 / fn
    gate017 = np.where(fn_UC <= 1, 0, 1)


    '''quants'''
    vol_steel = sec_G / sec_spacing / 7850
    vol_conc = slab_conc_vol
    vol_rebar = At_prov/1000000
    vol_galv = deck_crossSec_area/1000000
    vol_timb = 0*deck_index
    h = sec_hs+slab_h

    gate000 = gate001 + gate002 + gate003 + gate004 + gate005 + gate006 + gate007 + gate008 + gate009 + gate010 + gate011 + gate012 + gate013 + gate014 + gate015 + gate016 + gate017

    return gate000, h, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb


def detailed_composite_pri(Gk, Qk, lx, ly, no_sec, deck_type, conc_type, deck_t, slab_h, fire_t, bm_index):
    #(gk, qk, lx, ly, no_sec(range 0 to 3.0 ceiling'd), deck_type(0=51+), conc_type (range 0 to 2, <1=NW), deck_t (range 0 to 3, <1=0.9, <2=1.0, <3=1.2), slab_h (range 100 to 210, floored to 10mm inc), fire_t(60, 90, 120))
    '''paras'''
    fck = 30 #MPa
    fy = 355 #MPa
    Ecm = 31
    gammaG = 1.35
    gammaQ = 1.5
    gammaM0 = 1
    gammaM1 = 1
    gammaM2 = 1.1
    gammaV = 1.25 #shear studs
    gammaC = 1.5 #concrete
    gammaS = 1.15 #persistent adn transient design for reinforcements
    xi = 0.925
    psi_0_1_cons = 1
    psi_0_1_comp = 0.7

    '''bay-geometry'''
    no_sec = np.ceil(no_sec)
    Lmin = np.minimum(lx, ly)
    Lmax = np.maximum(lx, ly)
    sec_spacing = Lmin / no_sec


    deck_type=np.floor(deck_type)

    #slab_h = np.where(slab_t < 135, 1, np.where(slab_t < 145, 2, np.where(slab_t < 155, 3, np.where(slab_t < 165, 4, np.where(slab_t < 175, 5, np.where(slab_t < 185, 6, np.where(slab_t < 195, 7, np.where(slab_t < 225, 8, 250))))))))
    slab_h = np.floor(slab_h/10)*10 #slab_h to be in range 130 to 210mm
    slab_wt_wet = np.where(conc_type<1, 2.16+((slab_h-100)*0.025), 1.71+((slab_h-100)*0.02))
    slab_wt_dry = np.where(conc_type<1, 2.12+((slab_h-100)*0.024), 1.62+((slab_h-100)*0.019))
    slab_conc_vol = (slab_h-10)/1000
    deck_attributes = np.array([[0.9, 0.86, 1578, 0.13, 15.9], [1, 0.96, 1762, 0.14, 16.5], [1.2, 1.16, 2137, 0.17, 16.8]])


    deck_index = np.floor(deck_t)
    deck_index = deck_index.astype(int)
    deck_attributes = deck_attributes[deck_index]

    slab_index = np.where(slab_h == 100, 0,np.where(slab_h == 110, 1, np.where(slab_h == 120, 2, np.where(slab_h == 130, 3, np.where(slab_h == 140, 4, np.where(slab_h == 150, 5, np.where(slab_h == 160, 6, np.where(slab_h == 170, 7, np.where(slab_h == 180, 8, np.where(slab_h == 190, 9, 10))))))))))
    max_deck_span_60min_5kPa_1mm = [3.01, 2.94, 2.86, 2.79, 2.73, 2.67, 2.61, 2.56, 2.51, 2.46, 2.42]
    mesh_60min_5kPa_1mm = [142, 142, 142, 193, 193, 252, 252, 252, 393, 393, 393]

    slab_and_deck_wt_wet = slab_wt_wet + np.take(deck_attributes, 3)
    slab_and_deck_wt_dry = slab_wt_dry + np.take(deck_attributes, 3)
    deck_crossSec_area = deck_attributes[:, 2]
    slab_mesh_span = np.take(mesh_60min_5kPa_1mm, slab_index)

    max_deck_span = np.take(max_deck_span_60min_5kPa_1mm, slab_index)

    gate001 = np.where(sec_spacing<=max_deck_span, 0, 1)#deck span check max_deck_span_60min_5kPa_1mm[slab_index]
    gate002 = np.where(fire_t==60, 0, 1)#fire period check

    '''beam selection'''
    bm_index = np.floor(bm_index)
    bm_index = bm_index.astype(int)
    beam_identity = sl.UBnames(bm_index)
    beam_attributes = sl.UBsectionAgregator(bm_index)

    # naturalises beam properties
    pri_name = beam_identity
    pri_G = beam_attributes[0]
    pri_hs = beam_attributes[1]
    pri_b = beam_attributes[2]
    sec_tw = beam_attributes[3]
    sec_tf = beam_attributes[4]
    sec_r = beam_attributes[5]
    sec_hi = beam_attributes[6]
    sec_d = beam_attributes[7]
    sec_Iy = beam_attributes[8]
    sec_Wely = beam_attributes[9]
    sec_Wply = beam_attributes[10]
    sec_iy = beam_attributes[11]
    sec_Avz = beam_attributes[12]
    sec_Iz = beam_attributes[13]
    sec_Welz = beam_attributes[14]
    sec_Wplz = beam_attributes[15]
    sec_iz = beam_attributes[16]
    sec_Ss = beam_attributes[17]
    sec_It = beam_attributes[18]
    sec_Iw = beam_attributes[19]
    sec_Aa = beam_attributes[20]


    '''loads'''
    cll = 1.5 #kPa
    bm_wt = sec_G

    '''combinations'''
    Fd1 = gammaG*(sec_G/sec_spacing + np.take(deck_attributes, 3)) + gammaG*(slab_wt_wet+cll) #construction stage ULS
    Fd = xi*gammaG*(sec_G/sec_spacing + gk + slab_and_deck_wt_dry) + gammaQ*qk #composite stage ULS
    g1 = slab_and_deck_wt_dry + sec_G/sec_spacing #frame selfweight dry
    g2 = gk #superdead
    q1 = qk #live

    '''bm and v'''
    #construction
    M_Ed_sec_cons = Fd1 * Lmax**2 / 8
    V_Ed_sec_cons = Fd1 * Lmax / 2

    #composite
    M_Ed_sec_comp = Fd * Lmax**2 / 8
    V_Ed_sec_comp = Fd * Lmax / 2

    '''cross-section classification'''
    fy = np.where(np.maximum(sec_tf, sec_tw)<=16, fy, np.where(np.maximum(sec_tf, sec_tw)<=40, fy-10, fy-20))
    epsilon = np.sqrt(235/fy)
    c_tf_ratio = 0.5*(sec_b - sec_tw - (2*sec_r))/sec_tf #outstand compression flange
    classification_outstand = np.where(c_tf_ratio <= 9*epsilon, 1, np.where(c_tf_ratio <= 10*epsilon, 2, np.where(c_tf_ratio <= 14*epsilon, 3, 4)))
    c_tw_ratio = sec_d/sec_tw
    classification_internal = np.where(c_tw_ratio <= 72*epsilon, 1, np.where(c_tw_ratio <= 83*epsilon, 2, np.where(c_tw_ratio <= 124*epsilon, 3, 4)))
    classification = np.maximum(classification_internal, classification_outstand)

    '''design resistance'''
    #shear buckling
    eta_vbuck = 1
    gate003 = np.where(sec_hi/sec_tw*xi/epsilon/72<=1, 0, 1)

    #vertical shear resistance
    V_c_Rd = sec_Avz*fy/np.sqrt(3)/gammaM0
    V_uc = V_Ed_sec_comp / V_c_Rd
    gate004 = np.where(V_uc <= 1, 0, 1)

    #bending
    gate005 = np.where(V_uc <= 0.5, 0, 1) #low shear chk
    gate006 = np.where(classification <= 2, 0, 1) #plastic chk
    M_c_Rd = sec_Wply*fy/gammaM0/1000
    M_uc = M_Ed_sec_comp / M_c_Rd
    gate007 = np.where(M_uc <= 1, 0, 1)

    #shear connectors
    '''TO BE REVIEWED FOR ECONOMISE'''
    fu = 450
    stud_dia = 19
    hsc = 95

    gate008 = np.where(hsc/stud_dia >= 4, 0, 1)
    P_Rd_1 = 0.8*fu*np.pi*stud_dia**2 / 4 / gammaV /1000
    alpha_studs = 1
    P_Rd_2 = 0.29*alpha_studs*stud_dia**2 * np.sqrt(fck*Ecm*1000)/ gammaV /1000

    hp = 51 #deck depth
    hsc = 95 #stud height
    nr = 1 #to be developed
    b0 = 110 #concrete between re-entrants
    kt = 0.7*b0/np.sqrt(nr)/hp * ((hsc/hp)-1)
    kt_max = 0.85
    kt = np.minimum(kt, kt_max)
    P_Rd = np.minimum(P_Rd_1, P_Rd_2)*kt
    nrP_Rd = nr*P_Rd

    #degree of shear connection
    gate009 = np.where(hsc >= 4*stud_dia, 0, 1)
    gate010 = np.where(Lmax <= 25, 0, 1)
    eta_degree_min = np.maximum(0.4, 1-(355/fy)*(0.75-(0.03*Lmax/1000)))

    #effective width
    bei = Lmax/8
    b0 = 0
    beff = np.minimum(b0 + 2*bei, sec_spacing)

    #compressive resistance of concrete
    fcd = fck/gammaC
    hc = slab_h-hp
    Ncf = 0.85*fcd*beff*hc

    #tensile reesistance of steel member
    Npl_a = fy*sec_Aa

    #compressive force in flange
    Nc_comp = np.minimum(Ncf, Npl_a)

    #resistance of the shear connectors
    deck_pitch = 150
    no_studs_half = 1000*Lmax/2/deck_pitch
    Nc_studs = no_studs_half*P_Rd_2

    #degree of shear connection
    eta_degree = np.minimum(1, Nc_studs / Nc_comp)
    gate011 = np.where(eta_degree >= eta_degree_min, 0, 1)

    '''design resistance of the cross-section for the composite stage'''
    #vertical shear resistance
    V_uc = V_Ed_sec_cons / V_c_Rd
    gate012 = np.where(V_uc <= 1, 0, 1)

    #resistance in bending
    gate013 = np.where(V_uc/2 <= 1, 0, 1)
    xpl = (Npl_a-Nc_studs)/2/fy/sec_b*1000
    PNA_text = np.where(Npl_a <= Ncf, "PNA lies in top flange", np.where(xpl < sec_tf, "PNA lies in top flange", "PNA lies in web"))
    PNA_index = np.where(Npl_a <= Ncf, 0, np.where(xpl < sec_tf, 1,2))

    '''#PNA in the top flange
    F1a = Nc_studs
    F6b = 2*fy*sec_b*xpl
    F7b = Npl_a
    M_A = 0'''

    #PNA  in the concrete slab
    Mpl_Rd_c = (Npl_a*((sec_hs/2)+slab_h-(Npl_a/Ncf*hc/2)))/1000
    xc = Npl_a/Ncf*hc
    Mpl_Rd_c2 = sec_hs/2 + slab_h - xc/2 #potentially more accurate for PNA in conc

    #PNA  in the top flange
    hd = hp
    Mpl_Rd_f = ((Npl_a*sec_hs/2)+(Ncf*((hc/2)+hd)))/1000

    #PNA  in the web
    Mpl_a_Rd = M_c_Rd
    hw = sec_hs-(2*sec_tf)
    Nw = 0.95*fy*sec_tw*hw
    Mpl_Rd_w = (Mpl_a_Rd + Ncf*((hc+2*hd+sec_hs)/2) - (Ncf**2*sec_hs/Nw/4))/1000

    Mpl_Rd =np.where(PNA_index == 0, Mpl_Rd_c, np.where(PNA_index == 1, Mpl_Rd_f, Mpl_Rd_w))

    #PNA with partitial
    MRd = Mpl_a_Rd + (Mpl_Rd-Mpl_a_Rd)*eta_degree

    '''longitudinal shear resistance of the slab'''
    fyd = 500/gammaS
    hf = hc
    cotTheta = 2.00569
    delta_x = Lmax/0.002
    delta_Fd = Nc_studs/2
    v_Ed = 1000*delta_Fd/hf/delta_x
    At = 1000*v_Ed*hf/fyd/cotTheta
    At_prov = np.where(At<=142, 142, np.where(At<=193, 193, np.where(At<=252, 252, np.where(At<=393, 393, np.where(At<=503, 503, np.where(At<=785, 785,1131))))))
    At_prov = np.maximum(slab_mesh_span, At_prov)

    #concrete flange crushing
    v = 0.6*(1-(fck/250))
    alpha_cc = 0.85
    fcd = alpha_cc*fck/gammaC
    sinTheta = 0.446197813
    cosTheta = 0.894934361
    v_Ed_crush = v * fcd * sinTheta * cosTheta
    long_v_UC = v_Ed / v_Ed_crush
    gate014 = np.where(long_v_UC <=1, 0, 1)

    '''verification at SLS'''
    #short term
    Ea = 210000
    Ec = 38000
    n0 = Ea/Ecm
    psi_l = 1.1 #creep multiplyer
    phi_t = 3 #creep coef
    nL = n0 * (1+(psi_l*phi_t))/1000
    n = 0.333*nL + 0.666*n0
    nd = Ea/Ec #natural frequence
    n_perm = 2*Ea/Ec #natural frequence

    #composite second moment of area
    y_el_perm = ((sec_Aa*sec_hs/2) + (beff*(slab_h-hp)*((sec_hs+(slab_h+hp)/2))/2)/n_perm) / (sec_Aa+(beff*(slab_h-hp)/n_perm))
    Ieq_perm = 10000*sec_Iy + 1000*beff*((slab_h-hp)**3)/12/n_perm + 100*sec_Aa*((y_el_perm-(sec_hs/2))**2) + 1000*beff*(slab_h-hp)/n_perm*((sec_hs+((slab_h+hp)/2)-y_el_perm)**2)

    Ieq1 = 10000*sec_Iy
    Ieq2 = 1000*beff*((slab_h-hp)**3)/12/n_perm
    Ieq3 = 100*sec_Aa*((y_el_perm-(sec_hs/2))**2)
    Ieq4 = 1000*beff*(slab_h-hp)/n_perm*((sec_hs+((slab_h+hp)/2)-y_el_perm)**2)

    #long term
    # def_gk = 5/384 * gk*sec_spacing * (Lmax*1000)**4 / Ea / Ieq_perm
    def_qk = 5 / 384 * qk * sec_spacing * (Lmax * 1000) ** 4 / Ea / Ieq_perm
    def_comp = 5 / 384 * Fd * sec_spacing * (Lmax * 1000) ** 4 / Ea / Ieq_perm
    def_lim_qk_sec = Lmax*1000/360
    def_lim_sls_sec = Lmax * 1000 / 250
    def_qk_sec_UC = def_qk / def_lim_qk_sec
    def_sls_sec_UC = def_comp / def_lim_sls_sec
    gate015 = np.where(def_qk_sec_UC <=1, 0, 1)
    gate016 = np.where(def_sls_sec_UC <= 1, 0, 1)

    #short term / dynamic
    y_el_dyn = ((sec_Aa*sec_hs/2) + (beff*(slab_h-hp)*((sec_hs+(slab_h+hp)/2))/2)/nd) / (sec_Aa+(beff*(slab_h-hp)/nd))
    Ieq_dyn = 10000*sec_Iy + 1000*beff*((slab_h-hp)**3)/12/nd + 100*sec_Aa*((y_el_dyn-(sec_hs/2))**2) + 1000*beff*(slab_h-hp)/nd*((sec_hs+((slab_h+hp)/2)-y_el_dyn)**2)
    def_dyn = 5 / 384 * (g1+g2+0.1*q1) * sec_spacing * (Lmax * 1000) ** 4 / Ea / Ieq_dyn
    fn = 18.07/np.sqrt(def_dyn)
    fn_UC = 4.5 / fn
    gate017 = np.where(fn_UC <= 1, 0, 1)


    '''quants'''
    vol_steel = sec_G / sec_spacing / 7850
    vol_conc = slab_conc_vol
    vol_rebar = At_prov/1000000
    vol_galv = deck_crossSec_area/1000000
    vol_timb = 0*deck_index
    h = sec_hs+slab_h

    gate000 = gate001 + gate002 + gate003 + gate004 + gate005 + gate006 + gate007 + gate008 + gate009 + gate010 + gate011 + gate012 + gate013 + gate014 + gate015 + gate016 + gate017

    return gate000, h, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb

def detailed_fs(gk, qk, lx, ly):
    fck = 30  # MPa
    h_ratio = np.where(qk <= 2.5, 28, np.where(qk <= 5, 26, np.where(qk <= 7.5, 25, 23)))

    h = np.maximum(lx, ly) / h_ratio
    h = np.ceil(h / 0.025) * 0.025
    sw = h * 25
    UDL_uls = 1.35 * (sw + gk) + 1.5 * qk
    F = lx * ly * UDL_uls

    # long dim
    dir1_stripDim = np.minimum(lx, ly) / 2
    dir1_Mhog = -0.086 * np.maximum(lx, ly) * F  # first interior span (worst case)
    dir1_Msag = 0.086 * np.maximum(lx, ly) * F  # first bay (worst case)
    dir1_Vcol = 0.6 * np.maximum(lx, ly) * F  # first interior span (worst case)
    dir1_M_strip_hog_col = 0.7 * dir1_Mhog
    dir1_M_strip_hog_mid = 0.3 * dir1_Mhog
    dir1_M_strip_sag_col = 0.6 * dir1_Msag
    dir1_M_strip_sag_mid = 0.4 * dir1_Msag
    dir1_M_strip_hog_col_As = slab_section_design(dir1_M_strip_hog_col, dir1_stripDim, h, 25, fck)
    dir1_M_strip_hog_mid_As = slab_section_design(dir1_M_strip_hog_mid, dir1_stripDim, h, 25, fck)
    dir1_M_strip_sag_col_As = slab_section_design(dir1_M_strip_sag_col, dir1_stripDim, h, 25, fck)
    dir1_M_strip_sag_mid_As = slab_section_design(dir1_M_strip_sag_mid, dir1_stripDim, h, 25, fck)
    dir1_rebar_volume = (dir1_M_strip_hog_col_As + dir1_M_strip_hog_mid_As + dir1_M_strip_sag_col_As + dir1_M_strip_sag_mid_As) / 1000000

    # short dim
    dir2_stripDim = np.maximum(lx, ly) / 2
    dir2_Mhog = -0.086 * np.minimum(lx, ly) * F  # first internor span (worst case)
    dir2_Msag = 0.086 * np.minimum(lx, ly) * F  # first bay (worst case)
    dir2_Vcol = 0.6 * np.minimum(lx, ly) * F  # first internor span (worst case)
    dir2_M_strip_hog_col = 0.7 * dir2_Mhog
    dir2_M_strip_hog_mid = 0.3 * dir2_Mhog
    dir2_M_strip_sag_col = 0.6 * dir2_Msag
    dir2_M_strip_sag_mid = 0.4 * dir2_Msag
    dir2_M_strip_hog_col_As = slab_section_design(dir2_M_strip_hog_col, dir1_stripDim, h, 45, fck) #assumes 20mm DIA main bars
    dir2_M_strip_hog_mid_As = slab_section_design(dir2_M_strip_hog_mid, dir1_stripDim, h, 45, fck) #assumes 20mm DIA main bars
    dir2_M_strip_sag_col_As = slab_section_design(dir2_M_strip_sag_col, dir1_stripDim, h, 45, fck) #assumes 20mm DIA main bars
    dir2_M_strip_sag_mid_As = slab_section_design(dir2_M_strip_sag_mid, dir1_stripDim, h, 45, fck) #assumes 20mm DIA main bars
    dir2_rebar_volume = (dir2_M_strip_hog_col_As + dir2_M_strip_hog_mid_As + dir2_M_strip_sag_col_As + dir2_M_strip_sag_mid_As) / 1000000

    vol_steel = 0
    vol_conc = h  # volume of concrete per m2
    vol_rebar = dir1_rebar_volume + dir2_rebar_volume
    vol_galv = 0.000
    vol_timb = 0.0
    return h, vol_steel, vol_conc, vol_rebar, vol_galv, vol_timb



# TIMBER FUNCTIONS

def joist_h_des(gk, qk, L, b, h, cc):
    kh = 1
    kmod = 0.8
    ksys = 1.1
    kcrit = 1
    kcr = 0.67
    kdef = 0.6
    lambdaM = 1.3
    psi_2 = 0.3
    f_mk = 24
    f_vk = 4
    E_0mean = 11000
    ec_timber = 263  # kgC02e/kg

    # Loading
    UDL_uls = 1.35 * gk + 1.5 * qk
    UDL_sls = gk + qk

    # ULS
    w_uls = UDL_uls * cc
    Muls = w_uls * L ** 2 / 8
    Vuls = w_uls * L / 2

    Wply = b * h ** 2 / 6
    stress_myd = Muls / Wply

    Vd = 3 * Vuls / 2 / b / h / kcr

    f_myd = kh * kcrit * kmod * ksys * f_mk / lambdaM
    f_vd = kmod * ksys * f_vk / lambdaM

    # SLS
    Iyy = b * h ** 3 / 12
    def_inst = 5 * gk * cc / 1000 * (1000 * L) ** 4 / 384 / E_0mean / Iyy * (1 + kdef)
    def_fin = 5 * qk * cc / 1000 * (1000 * L) ** 4 / 384 / E_0mean / Iyy * (1 + (psi_2 * kdef))
    def_tot = (def_inst + def_fin) * 1.1
    def_lim = L * 1000 / 250

    # CHK
    bendingCHK = f_myd / f_mk
    shearCHK = f_vd / f_vk
    defCHK = def_tot / def_lim

    utiCHK = defCHK

    # ensures return value is as close to 1 as possible but does not exceed
    for d in range(len(utiCHK)):
        if utiCHK[d] > 1:
            utiCHK[d] = 0.
        else:
            utiCHK[d] = utiCHK[d]

    return utiCHK


def joist_carbon_inc_ver001(b, h, cc, beamIndex, t_board, l1, l2, m):
    # PLY INDEX SIZE CONVERTER
    t_board = np.where(t_board >= 0.5, 50, 25)

    # JOIST INDEX TO SIZE CONVERTER

    joist_breath_array = [38, 47, 50, 63, 75, 100]
    joist_depth_array = [97, 122, 140, 147, 170, 184, 195, 220, 235, 250, 300]
    b = np.take(joist_breath_array, [b])
    h = np.take(joist_depth_array, [h])
    b = b[0]
    h = h[0]

    ec_timber = 0.655  # kgC02e/kg
    massTimber = ((b / 1000 * h / 1000 * 1000 / cc) + (t_board/1000))* 500
    EC_Timber = ec_timber * massTimber

    mass_conc = np.where(m<=20, 0, 2.400*m)
    ec_conc = 0.183  # kgC02e/kg
    EC_Conc = ec_conc * mass_conc

    bI = [1299.0, 1202.0, 1086.0, 990.0, 900.0, 818.0, 744.0, 677.0, 634.0, 592.0, 551.0, 509.0, 467.0, 393.0, 340.0, 287.0, 235.0, 202.0, 177.0, 153.0, 129.0, 500.0, 545.0, 415.0, 375.0, 342.0, 313.0, 283.0, 240.0, 198.0, 158.0, 137.0, 118.0, 96.9, 167.0, 132.0, 107.0, 88.9, 73.1, 100.0, 86.1, 71.0, 60.0, 52.0, 46.1, 51.0, 44.0, 37.0, 30.0, 23.0]

    ec_steel = 1.303
    mass_steel = np.take(bI, [beamIndex])
    EC_steel = ec_steel * mass_steel / l1 * 1000

    EC_sum = EC_Timber + EC_steel + EC_Conc

    return EC_sum, EC_Timber, EC_steel, EC_Conc


def hybrid_floor_cost_ver001(b, h, cc, beamIndex, t_board, m, l1, l2):
    bI = [1299.0, 1202.0, 1086.0, 990.0, 900.0, 818.0, 744.0, 677.0, 634.0, 592.0, 551.0, 509.0, 467.0, 393.0, 340.0, 287.0, 235.0, 202.0, 177.0, 153.0, 129.0, 500.0, 545.0, 415.0, 375.0, 342.0, 313.0, 283.0, 240.0, 198.0, 158.0, 137.0, 118.0, 96.9, 167.0, 132.0, 107.0, 88.9, 73.1, 100.0, 86.1, 71.0, 60.0, 52.0, 46.1, 51.0, 44.0, 37.0, 30.0, 23.0]
    cost_board = np.where(t_board >= 0.5, 33.9, 8.12)

    # JOIST INDEX TO SIZE CONVERTER
    joist_breath_array = [38, 47, 50, 63, 75, 100]
    joist_depth_array = [97, 122, 140, 147, 170, 184, 195, 220, 235, 250, 300]
    b = np.take(joist_breath_array, [b])
    h = np.take(joist_depth_array, [h])
    b = b[0]
    h = h[0]

    unit_cost_steel = 3.5 #£/kg
    mass_steel = np.take(bI, [beamIndex])/(l1+l2)*2000
    unit_cost_timber = 1.2936 #646.8/500kg/m3
    unit_cost_conc = 135 #£135/m3
    vol_tim = b*h/cc/1000 #excluding board
    vol_conc = m/1000
    cost_tim = 500 * vol_tim * unit_cost_timber + cost_board
    cost_steel = mass_steel * unit_cost_steel
    cost_conc =vol_conc * unit_cost_conc
    cost_sum = cost_tim + cost_steel + cost_conc

    return cost_sum, cost_tim, cost_steel, cost_conc
'''
a = joist_carbon_inc(0, 8, 600, 151, .4, 7500, 4500)
print("a", a)
'''
'''X = np.array([[0, 0, 400, 5], [5, 10, 600.81850517, 100]])
f2 = joist_carbon_inc(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
print(f2)'''

def joist_beam_vibration_level_inc_ver001(b_floor, l_beam, l1, l2, b_st, h_st, cc, b_board, t_board, E0_mean_ST, E0_mean_fb, t_screed,
                                   beamIndex, a, b_BSEN, ke1j, ke1b, ke2b, sides_supp, fw):
    beamIndex = np.ceil(beamIndex)
    py = 345 #MPa

    #MASS OF FLOOR
    h_screed = np.where(t_screed<=20, 0, t_screed)
    m_screed = np.where(t_screed<=20, 0, t_screed/1000*2400)#if screed thickness is more than 20mm then calculates mass on basis on screed thickness
    mass_gypsum = 0.013*2360 #13mm plastboard
    mass_battons = 0.028*0.095*500/0.4 #28x95mm battons at 400mm c/c


    # PLY INDEX SIZE CONVERTER
    t_board = np.where(t_board >= 0.5, 50, 25)
    m_board = np.where(t_board <= 0.5, 12.5, 25) #adds ply weight

    # JOIST INDEX TO SIZE CONVERTER
    joist_breath_array = [38, 47, 50, 63, 75, 100]
    joist_depth_array = [97, 122, 140, 147, 170, 184, 195, 220, 235, 250, 300]
    b_st = np.take(joist_breath_array, [b_st])
    h_st = np.take(joist_depth_array, [h_st])
    b_st = b_st[0]
    h_st = h_st[0]
    m_joists = b_st*h_st/1000*500/cc #adds joist weight

    m = m_screed + m_board + m_joists + mass_gypsum + mass_battons  # adds plastboard and battons

    I_st = b_st * (h_st ** 3) / 12
    I_b = b_board * (t_board ** 3) / 12
    b_span = cc - b_st
    b_eff1 = b_span + b_st
    b_eff2 = b_st + (20 * t_board)
    b_eff3 = b_st + (0.2 * l1)

    # UB SECTION BEAMS

    # tables
    #UCnames = ["UC 356 x 406 x 1299", "UC 356 x 406 x 1202", "UC 356 x 406 x 1086", "UC 356 x 406 x 990", "UC 356 x 406 x 900", "UC 356 x 406 x 818", "UC 356 x 406 x 744", "UC 356 x 406 x 677", "UC 356 x 406 x 634", "UC 356 x 406 x 592", "UC 356 x 406 x 551", "UC 356 x 406 x 509", "UC 356 x 406 x 467", "UC 356 x 406 x 393", "UC 356 x 406 x 340", "UC 356 x 406 x 287", "UC 356 x 406 x 235", "UC 356 x 368 x 202", "UC 356 x 368 x 177", "UC 356 x 368 x 153", "UC 356 x 368 x 129", "UC 305 x 305 x 500", "UC 305 x 305 x 454", "UC 305 x 305 x 415", "UC 305 x 305 x 375", "UC 305 x 305 x 342", "UC 305 x 305 x 313", "UC 305 x 305 x 283", "UC 305 x 305 x 240", "UC 305 x 305 x 198", "UC 305 x 305 x 158", "UC 305 x 305 x 137", "UC 305 x 305 x 118", "UC 305 x 305 x 97", "UC 254 x 254 x 167", "UC 254 x 254 x 132", "UC 254 x 254 x 107", "UC 254 x 254 x 89", "UC 254 x 254 x 73", "UC 203 x 203 x 100", "UC 203 x 203 x 86", "UC 203 x 203 x 71", "UC 203 x 203 x 60", "UC 203 x 203 x 52", "UC 203 x 203 x 46", "UC 152 x 152 x 51", "UC 152 x 152 x 44", "UC 152 x 152 x 37", "UC 152 x 152 x 30", "UC 152 x 152 x 23"]
    UCprops = [["UC 356 x 406 x 1299", "UC 356 x 406 x 1202", "UC 356 x 406 x 1086", "UC 356 x 406 x 990", "UC 356 x 406 x 900", "UC 356 x 406 x 818", "UC 356 x 406 x 744", "UC 356 x 406 x 677", "UC 356 x 406 x 634", "UC 356 x 406 x 592", "UC 356 x 406 x 551", "UC 356 x 406 x 509", "UC 356 x 406 x 467", "UC 356 x 406 x 393", "UC 356 x 406 x 340", "UC 356 x 406 x 287", "UC 356 x 406 x 235", "UC 356 x 368 x 202", "UC 356 x 368 x 177", "UC 356 x 368 x 153", "UC 356 x 368 x 129", "UC 305 x 305 x 500", "UC 305 x 305 x 454", "UC 305 x 305 x 415", "UC 305 x 305 x 375", "UC 305 x 305 x 342", "UC 305 x 305 x 313", "UC 305 x 305 x 283", "UC 305 x 305 x 240", "UC 305 x 305 x 198", "UC 305 x 305 x 158", "UC 305 x 305 x 137", "UC 305 x 305 x 118", "UC 305 x 305 x 97", "UC 254 x 254 x 167", "UC 254 x 254 x 132", "UC 254 x 254 x 107", "UC 254 x 254 x 89", "UC 254 x 254 x 73", "UC 203 x 203 x 86", "UC 203 x 203 x 71", "UC 203 x 203 x 60", "UC 203 x 203 x 52", "UC 203 x 203 x 46", "UC 152 x 152 x 37", "UC 152 x 152 x 30", "UC 152 x 152 x 23"],
        [1299.0, 1202.0, 1086.0, 990.0, 900.0, 818.0, 744.0, 677.0, 634.0, 592.0, 551.0, 509.0, 467.0, 393.0, 340.0, 287.0, 235.0, 202.0, 177.0, 153.0, 129.0, 500.0, 545.0, 415.0, 375.0, 342.0, 313.0, 283.0, 240.0, 198.0, 158.0, 137.0, 118.0, 96.9, 167.0, 132.0, 107.0, 88.9, 73.1, 86.1, 71.0, 60.0, 52.0, 46.1, 37.0, 30.0, 23.0],
        [600.0, 580.0, 569.0, 550.0, 531.0, 514.0, 498.0, 483.0, 474.6, 465.0, 455.6, 446.0, 436.6, 419.0, 406.4, 393.6, 381.0, 374.6, 368.2, 362.0, 355.6, 427.0, 415.0, 403.0, 391.0, 382.0, 374.0, 365.3, 352.5, 339.9, 327.1, 320.5, 314.5, 307.9, 289.1, 276.3, 266.7, 260.3, 254.1, 222.2, 215.8, 209.6, 206.2, 203.2, 161.8, 157.6, 152.4],
        [476.0, 471.0, 454.0, 448.0, 442.0, 437.0, 432.0, 428.0, 424.0, 421.0, 418.5, 416.0, 412.2, 407.0, 403.0, 399.0, 394.8, 374.7, 372.6, 370.5, 368.6, 340.0, 336.0, 334.0, 330.0, 328.0, 325.0, 322.2, 318.4, 314.5, 311.2, 309.2, 307.4, 305.3, 265.2, 261.3, 258.8, 256.3, 254.6, 209.1, 206.4, 205.8, 204.3, 203.6, 154.4, 152.9, 152.2],
        [100.0, 95.0, 78.0, 71.9, 65.9, 60.5, 55.6, 51.2, 47.6, 45.0, 42.1, 39.1, 35.8, 30.6, 26.6, 22.6, 18.4, 16.5, 14.4, 12.3, 10.4, 45.1, 41.3, 38.9, 35.4, 32.6, 30.0, 26.8, 23.0, 19.1, 15.8, 13.8, 12.0, 9.9, 19.2, 15.3, 12.8, 10.3, 8.6, 12.7, 10.0, 9.4, 7.9, 7.2, 8.0, 6.5, 5.8],
        [140.0, 130.0, 125.0, 115.0, 106.0, 97.0, 88.9, 81.5, 77.0, 72.3, 67.5, 62.7, 58.0, 49.2, 42.9, 36.5, 30.2, 27.0, 23.8, 20.7, 17.5, 75.1, 68.7, 62.7, 57.2, 52.6, 48.3, 44.1, 37.7, 31.4, 25.0, 21.7, 18.7, 15.4, 31.7, 25.3, 20.5, 17.3, 14.2, 20.5, 17.3, 14.2, 12.5, 11.0, 11.5, 9.4, 6.8],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 13, 13, 13, 13, 13, 10, 10, 10, 10, 10, 8, 8, 8],
        [320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 320.0, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 277.1, 225.7, 225.7, 225.7, 225.7, 225.7, 181.2, 181.2, 181.2, 181.2, 181.2, 138.8, 138.8, 138.8],
        [280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 280.0, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 237.1, 199.7, 199.7, 199.7, 199.7, 199.7, 161.2, 161.2, 161.2, 161.2, 161.2, 122.8, 122.8, 122.8],
        [754950, 663970, 596070, 519260, 450550, 392540, 342470, 299820, 275190, 250510, 227280, 204880, 183340, 146960, 122880, 100220, 79430, 66600, 57460, 48930, 40590, 169080, 148200, 130180, 113040, 100770, 89820, 79120, 64450, 51150, 39000, 33060, 27920, 22500, 30000, 22530, 17510, 14270, 11410, 9446, 7615, 6121, 5256, 4565, 2212, 1750, 1252],
        [25160, 22890, 20950, 18880, 16970, 15270, 13750, 12410, 11590, 10770, 9977, 9187, 8398, 7014, 6047, 5092, 4169, 3556, 3121, 2703, 2282, 7919, 7142, 6460, 5782, 5276, 4803, 4332, 3657, 3010, 2384, 2063, 1775, 1461, 2075, 1631, 1313, 1096, 898.4, 850.2, 705.7, 584.1, 509.8, 449.3, 273.5, 222.1, 164.3],
        [33260, 30030, 27230, 24300, 21640, 19270, 17180, 15360, 14250, 13160, 12090, 11050, 10020, 8244, 7021, 5834, 4708, 3994, 3477, 2987, 2501, 9895, 8835, 7922, 7023, 6353, 5735, 5124, 4266, 3459, 2699, 2315, 1976, 1611, 2424, 1870, 1485, 1224, 992.7, 976.4, 798.4, 655.7, 567, 497.1, 309.1, 248, 182.3],
        [45.7, 45.5, 45.1, 45.2, 45, 44.6, 40.9, 40.4, 40.4, 40.1, 40.1, 40.3, 40.1, 39.9, 38.9, 37.9, 43.4, 43.1, 42.6,
         42.1, 42, 41.7, 41.8, 41.6, 41.4, 41.6, 41.4, 41.2, 40.5, 41.5, 42.3, 42, 41.6, 41.2, 40.5, 40.2, 39.9, 39.7,
         39.5, 39.3, 39.1, 39, 38.7, 38.5, 38.4, 38.7, 38.4, 38.3, 38.1, 37.8, 37.7, 37.1, 37, 36.9, 36.7, 36.5, 36.3,
         35.6, 37.1, 36.8, 36.7, 36.4, 36.3, 35.9, 35.7, 35.5, 34.8, 34.3, 33.6, 33.1, 34, 33.8, 33.6, 33.4, 33.1, 33,
         32.6, 32.5, 32.2, 31.4, 30.9, 30.5, 30, 29.7, 32.2, 31, 30.8, 30.6, 30.4, 30.2, 30, 29.9, 29.7, 29.3, 29.2, 29,
         28.5, 28, 27.8, 27.6, 27.2, 28.1, 27.9, 27.7, 27.4, 27.2, 27, 26.9, 26.7, 26.6, 26.4, 26.2, 25.9, 25.7, 25.5,
         26.2, 25.9, 25.7, 25.3, 25, 24.9, 24.6, 24.3, 23.4, 23.1, 22.1, 22.1, 21.9, 21.8, 21.6, 21.3, 21.2, 20.7, 20.4,
         19, 19.1, 18.9, 18.8, 18.7, 18.5, 18.7, 18.5, 18.3, 18.2, 17.9, 17, 16.9, 16.8, 16.7, 16.4, 16.4, 16.3, 15.8,
         15, 14.8, 14.7, 14.5, 14.2, 13.9, 13, 12.9, 12.8, 12.5, 12.3, 12.3, 12.4, 12.2, 11.8, 10.9, 10.8, 10.5, 10.5,
         10.3, 10, 8.7, 8.5, 8.4, 7.4, 6.4, 5.3],
        [376.4, 339.5, 313.7, 266.6, 242.1, 216.9, 403.2, 344.5, 299.9, 288.5, 271.2, 235.9, 213.4, 184.5, 180.7, 172.2,
         570.7, 516.4, 438.9, 379.6, 346.3, 328, 316.3, 282.7, 261.8, 235.9, 212.5, 184.5, 181.5, 812.9, 688.7, 647.9,
         583.9, 526.8, 425.5, 394, 355.4, 318.2, 290.4, 264.5, 243.9, 228.5, 215.2, 204.5, 194.1, 364.4, 331.2, 300.1,
         268.3, 242.3, 218.9, 206.6, 190.6, 179.7, 168.6, 160.4, 153.9, 144.8, 300.2, 275.3, 245.7, 226.2, 204.3, 193.3,
         179.8, 165.5, 156.5, 146.8, 132.5, 125.2, 297.6, 270.6, 247.8, 220.6, 199.8, 178.3, 165, 150.4, 137.9, 139,
         129.7, 117.9, 104.7, 97.77, 407.4, 277.4, 251.9, 230.8, 209.9, 193.4, 178.5, 162.9, 146.2, 139.8, 127.5, 116.5,
         119, 109.9, 99.81, 93.43, 87.33, 278.8, 251.8, 228.5, 209.3, 186.5, 171.5, 154.8, 143.9, 132.3, 123.9, 113.5,
         105.2, 95.28, 86.21, 127.2, 96.99, 81.25, 97.12, 90.23, 81.79, 75.87, 71.07, 69.44, 63.39, 84.82, 73.43, 66.73,
         62.05, 57.77, 54.33, 59.14, 54.98, 50.12, 61.47, 55.8, 51.19, 48.01, 43.58, 40.85, 51.56, 46.97, 43.72, 39.26,
         36.39, 47.96, 41.75, 38.48, 34.51, 33.19, 34.54, 29.75, 27.5, 35.63, 31.4, 28.57, 26.71, 25.61, 22.98, 26.6,
         22.57, 20.12, 29.94, 26.48, 23.47, 22.2, 19.95, 18.95, 20.35, 17.73, 16.49, 17.92, 16.82, 15.72, 14.7, 12.93,
         12.5, 9.968, 8.292, 6.541],
        [63470, 56400, 50000, 43420, 38490, 33130, 33430, 26820, 23350, 21700, 20490, 18460, 16230, 14000, 11750, 9545,
         118520, 104970, 85110, 70280, 64010, 59090, 57630, 50710, 45490, 43400, 38580, 33120, 28960, 206350, 189900,
         175050, 152760, 133870, 103310, 93210, 83050, 72770, 65560, 59010, 53980, 50070, 45270, 42120, 39010, 36520,
         32140, 28650, 25190, 21910, 19510, 17040, 15590, 14510, 13300, 12280, 11230, 9423, 67220, 60730, 53670, 48350,
         42960, 38900, 34870, 31190, 12900, 11360, 9067, 7800, 64430, 57760, 51660, 45510, 39930, 35460, 31570, 28000,
         25010, 9443, 8177, 6851, 5457, 4789, 87540, 54300, 48670, 43890, 39500, 35670, 31870, 29430, 25610, 23130,
         20630, 18510, 7645, 6633, 5786, 5185, 4385, 48410, 42590, 38090, 34300, 30170, 27090, 23950, 22060, 19850,
         18430, 16310, 14240, 12370, 10780, 15830, 11410, 9309, 5001, 4508, 3935, 3436, 2917, 1441, 1208, 3869, 3387,
         2942, 2692, 2389, 2007, 1263, 1041, 857.3, 2514, 2346, 2089, 1870, 1671, 1452, 1184, 1046, 912.5, 794.6, 644.9,
         1803, 1545, 1364, 1203, 1021, 634.5, 538, 409.7, 1362, 1108, 968.2, 811, 357.8, 280.2, 1062, 895.6, 764.3, 461,
         388.7, 336.1, 194.1, 155.3, 122.9, 677.3, 570.6, 447.5, 178.5, 148.7, 119.3, 384.6, 307.6, 163.8, 136.7, 89.76,
         55.75],
        [3096, 2771, 2469, 2160, 1924, 1656, 2129, 1736, 1531, 1428, 1352, 1222, 1082, 933.6, 783.6, 636.3, 5538, 4951,
         4082, 3411, 3130, 2896, 2832, 2510, 2263, 2159, 1929, 1656, 1448, 8725, 8238, 7660, 6774, 6003, 4728, 4295,
         3854, 3408, 3085, 2796, 2552, 2373, 2156, 2010, 1866, 2268, 2015, 1813, 1609, 1413, 1267, 1102, 1013, 945.8,
         870.8, 805.6, 739, 621.3, 3271, 2969, 2643, 2393, 2142, 1930, 1739, 1559, 883.7, 773.3, 620.2, 534.8, 3254,
         2939, 2649, 2352, 2074, 1856, 1644, 1466, 1313, 710, 610.2, 513.8, 411.5, 362.2, 4524, 2919, 2638, 2392, 2170,
         1970, 1771, 1640, 1438, 1292, 1159, 1043, 602, 518.6, 454.7, 408.7, 346.6, 2790, 2483, 2241, 2030, 1801, 1627,
         1452, 1341, 1214, 1120, 995, 871, 761.6, 665.7, 1017, 743.1, 610.8, 436.8, 391.7, 343.6, 301.2, 256.3, 161,
         135.8, 361.6, 319.7, 279.2, 256.3, 228.3, 192.2, 152.2, 125.4, 103.9, 259.2, 243.4, 217.7, 195.5, 175.5, 152.9,
         152.5, 135.5, 118.6, 103.9, 84.63, 199.2, 172.1, 152.6, 135.2, 114.9, 88.56, 75.67, 57.79, 157.2, 128.7, 112.9,
         94.8, 56.79, 44.69, 127.3, 108.1, 92.65, 73.59, 62.55, 54.48, 37.91, 30.52, 24.2, 91.97, 77.95, 61.26, 34.94,
         29.18, 23.49, 57.45, 46.19, 32.19, 27.02, 20.24, 14.67],
        [4886, 4358, 3879, 3370, 2995, 2575, 3474, 2818, 2462, 2297, 2167, 1940, 1712, 1469, 1244, 1020, 8838, 7873,
         6459, 5378, 4915, 4546, 4435, 3918, 3529, 3348, 2984, 2554, 2242, 14160, 13130, 12190, 10740, 9497, 7431, 6739,
         6027, 5314, 4799, 4339, 3953, 3671, 3334, 3108, 2884, 3658, 3239, 2901, 2562, 2243, 2003, 1748, 1601, 1491,
         1371, 1267, 1163, 982.4, 5101, 4621, 4100, 3706, 3310, 2984, 2687, 2406, 1383, 1212, 974.7, 842.8, 5082, 4579,
         4119, 3646, 3211, 2865, 2537, 2259, 2020, 1114, 960.1, 808.9, 648.5, 571.2, 7146, 4565, 4114, 3723, 3369, 3054,
         2742, 2532, 2217, 1994, 1786, 1605, 943, 813.3, 712, 639.9, 544.1, 4381, 3889, 3500, 3164, 2799, 2525, 2247,
         2073, 1874, 1728, 1533, 1342, 1172, 1024, 1575, 1145, 938.6, 684.9, 613.9, 537.7, 471.5, 402.4, 258.5, 218.2,
         568.9, 499.7, 435.8, 399.5, 355.6, 300.4, 241.8, 200.3, 166.1, 405.3, 378.8, 338.3, 303.8, 272, 237.2, 240.3,
         213, 186.6, 163, 133.2, 310, 266.9, 236.5, 209, 178.2, 138.9, 118.1, 90.82, 242.9, 198.7, 174.1, 146.5, 89.02,
         70.26, 195.6, 165.5, 141.7, 116, 98.42, 85.42, 60.07, 48.48, 38.83, 141.1, 119.4, 94.15, 54.88, 46.03, 37.3,
         88.25, 70.97, 49.78, 41.61, 31.2, 22.6],
        [9, 8.9, 8.8, 8.8, 8.7, 8.6, 6.7, 6.5, 6.4, 6.4, 6.4, 6.4, 6.3, 6.3, 6, 5.8, 9.7, 9.6, 9.4, 9.2, 9.2, 9.1, 9.1,
         9, 8.9, 9, 9, 9, 8.7, 10.8, 10.8, 10.7, 10.5, 10.3, 10.1, 10, 9.9, 9.8, 9.7, 9.7, 9.7, 9.6, 9.5, 9.4, 9.4, 7,
         6.9, 6.8, 6.8, 6.7, 6.6, 6.5, 6.5, 6.4, 6.4, 6.3, 6.2, 6, 9.5, 9.5, 9.4, 9.3, 9.2, 9.2, 9.1, 9, 6.3, 6.2, 6,
         5.8, 9.3, 9.2, 9.1, 9, 8.9, 8.9, 8.8, 8.7, 8.7, 5.7, 5.6, 5.5, 5.3, 5.2, 9.2, 8.8, 8.7, 8.6, 8.6, 8.5, 8.4,
         8.4, 8.3, 8.2, 8.1, 8.1, 5.5, 5.5, 5.4, 5.3, 5.2, 8.2, 8.1, 8, 8, 7.9, 7.8, 7.8, 7.7, 7.7, 7.7, 7.6, 7.5, 7.4,
         7.3, 7.2, 7, 6.9, 5, 5, 4.9, 4.8, 4.7, 3.5, 3.4, 4.6, 4.6, 4.6, 4.5, 4.5, 4.3, 3.4, 3.3, 3.1, 4.3, 4.3, 4.2,
         4.2, 4.2, 4.1, 3.3, 3.3, 3.2, 3.2, 3.1, 4, 4, 3.9, 3.9, 3.8, 3, 3, 2.8, 3.9, 3.9, 3.8, 3.7, 2.6, 2.5, 3.9, 3.9,
         3.8, 2.7, 2.6, 2.6, 2.1, 2, 1.9, 3.5, 3.4, 3.3, 2.2, 2.1, 2, 3.1, 3, 2.3, 2.3, 2, 1.8],
        [17.6, 16.3, 15.1, 13.7, 12.7, 11.5, 19.9, 17.4, 16, 15.3, 14.7, 13.6, 12.6, 11.3, 10.3, 9.3, 26.4, 24.4, 21.4,
         18.9, 17.7, 16.8, 16.5, 15.2, 14.2, 13.6, 12.6, 11.3, 10.5, 33.6, 32.3, 30.7, 28.2, 25.9, 21.7, 20.3, 18.7,
         17.2, 15.9, 14.9, 13.8, 13.1, 12.3, 11.8, 11.2, 18.9, 17.4, 16.1, 14.8, 13.5, 12.5, 11.3, 10.6, 10.1, 9.6, 9.1,
         8.7, 7.8, 17.1, 15.9, 14.5, 13.5, 12.5, 11.5, 10.7, 10, 10.2, 9.3, 8.1, 7.5, 18.1, 16.8, 15.6, 14.3, 13, 12,
         10.9, 10.1, 9.4, 9.9, 8.9, 8, 7.1, 6.6, 25.3, 18.4, 17.1, 15.9, 14.8, 13.8, 12.8, 12, 11, 10.2, 9.5, 8.8, 9.4,
         8.5, 7.8, 7.3, 6.7, 20, 18.4, 17.1, 15.9, 14.5, 13.5, 12.5, 11.8, 11, 10.3, 9.5, 8.7, 8, 7.4, 10.4, 8.4, 7.4,
         8.7, 8, 7.4, 6.9, 6.3, 5.6, 5, 7.7, 7, 6.4, 6, 5.6, 5.1, 5.8, 5.2, 4.6, 6.5, 6.2, 5.7, 5.3, 4.9, 4.5, 6, 5.5,
         5, 4.6, 4.1, 5.9, 5.3, 4.9, 4.5, 4.1, 4.5, 4, 3.5, 5.2, 4.5, 4.2, 3.8, 3.9, 3.4, 4.5, 4, 3.6, 4.7, 4.2, 3.9,
         3.7, 3.2, 2.9, 4.1, 3.7, 3.2, 3.5, 3.2, 2.8, 3.4, 3, 3.3, 2.9, 2.9, 2.8],
        [5789, 4344, 3253, 2229, 1649, 1104, 7153, 4395, 3181, 2703, 2329, 1718, 1263, 834.8, 581.9, 390, 24410, 18510,
         11550, 7381, 5895, 4834, 4529, 3306, 2543, 2144, 1588, 1033, 763.6, 60450, 52120, 44110, 33050, 24680, 13650,
         10750, 8098, 5923, 4542, 3520, 2692, 2208, 1741, 1452, 1198, 7153, 5358, 4118, 3040, 2206, 1656, 1167, 931.5,
         774.3, 629.9, 518.3, 425.5, 293.9, 6167, 4777, 3511, 2704, 2026, 1510, 1159, 866.2, 745.5, 522.3, 312.2, 226.7,
         7219, 5563, 4279, 3127, 2247, 1675, 1182, 875, 651.1, 620.5, 416.5, 276.8, 166.1, 125, 20610, 7065, 5459, 4228,
         3299, 2569, 1953, 1567, 1120, 833.7, 627.2, 471.5, 479.7, 321.6, 231.2, 178.7, 124.3, 8577, 6477, 5004, 3876,
         2850, 2195, 1645, 1328, 1027, 802.2, 586.1, 418.3, 297.5, 209.5, 802.9, 350.5, 207.6, 315.8, 233.4, 168.2, 123,
         86.67, 71.3, 49.07, 250.9, 178.8, 126.7, 101.3, 75.95, 51.75, 73.98, 48.2, 32.15, 145.5, 121, 90.52, 69.06,
         51.67, 37.03, 89.04, 65.77, 47.54, 33.73, 21.29, 92.42, 62.61, 45.98, 33.21, 23.04, 28.85, 18.94, 10.64, 55.53,
         33.27, 23.72, 15.77, 15.02, 8.736, 34.83, 22.23, 14.76, 31.82, 21.17, 14.79, 12.3, 7.478, 4.833, 24.02, 15.43,
         8.629, 9.665, 6.495, 4.203, 10.39, 6.032, 7.099, 4.471, 3.619, 2.906],
        [185250, 163220, 143400, 123500, 108680, 92700, 81240, 64010, 55290, 51080, 48080, 43010, 37530, 32070, 26620,
         21370, 304410, 265670, 210640, 170660, 154320, 141320, 137550, 119890, 106730, 101450, 89440, 76020, 65890,
         485320, 449320, 409540, 350470, 301870, 225650, 201000, 176960, 153180, 136510, 121870, 110360, 101890, 91390,
         84530, 78000, 77860, 67730, 59790, 52070, 44830, 39610, 34160, 31080, 28840, 26280, 24200, 22000, 18300,
         122460, 109340, 95560, 85320, 75110, 67260, 59910, 53220, 22040, 19230, 15160, 12940, 97850, 86670, 76520,
         66600, 57710, 50690, 44630, 39310, 34650, 13190, 11280, 9364, 7377, 6440, 117630, 67920, 60090, 53420, 47560,
         42470, 37530, 34350, 29550, 26360, 23370, 20760, 8656, 7402, 6407, 5708, 4786, 49560, 42840, 37760, 33460,
         28990, 25720, 22450, 20470, 18250, 16780, 14670, 12690, 10910, 9436, 14430, 10130, 8165, 4456, 3978, 3444,
         2984, 2512, 1239, 1033, 2660, 2311, 1989, 1810, 1596, 1328, 845.4, 688.5, 562.8, 1260, 1172, 1035, 920, 816.5,
         703.8, 589, 516.2, 446.2, 385.7, 309.6, 715.1, 607, 531.6, 465.1, 391, 245.1, 206.1, 154.9, 410.9, 329.2,
         285.1, 236.3, 104.7, 80.97, 233.6, 194.3, 164.1, 101.2, 84.31, 72.26, 44.03, 34.79, 27.18, 103, 85.61, 65.87,
         27.88, 22.92, 18.15, 37.34, 29.32, 15.36, 9.847, 4.687, 1.981],
        [777.4, 702.5, 639.5, 555.4, 501.3, 440.7, 743.7, 629.1, 556.3, 528.7, 500.2, 445.1, 400.4, 346.8, 316.8, 282.8,
         1243.0, 1125.0, 953.4, 817.6, 752.7, 705.8, 687.2, 615.1, 563.7, 525.1, 472.8, 408.8, 377.6, 1753.7, 1616.5,
         1521.5, 1371.5, 1236.6, 1002.0, 922.9, 835.3, 748.1, 682.5, 623.3, 571.4, 534.1, 494.3, 465.6, 437.2, 733.2,
         663.7, 603.9, 542.0, 485.9, 439.7, 398.4, 368.3, 346.1, 322.8, 303.5, 285.6, 255.9, 735.0, 672.0, 603.0, 552.0,
         499.0, 457.0, 419.0, 381.0, 319.3, 288.6, 246.8, 224.0, 742.0, 676.0, 617.0, 553.0, 495.0, 445.0, 400.0, 362.0,
         328.0, 280.7, 250.6, 220.4, 187.2, 170.6, 1021.0, 697.4, 636.0, 582.2, 532.4, 488.6, 445.7, 411.8, 366.8,
         337.0, 305.6, 277.4, 243.8, 216.8, 194.1, 178.4, 159.5, 702.1, 634.8, 578.6, 529.4, 474.3, 433.7, 391.5, 363.6,
         332.7, 308.0, 277.6, 249.3, 222.0, 197.3, 303.3, 228.1, 190.0, 195.7, 178.2, 159.3, 143.9, 128.9, 117.6, 104.4,
         176.2, 155.4, 138.9, 128.7, 117.4, 104.7, 107.9, 95.3, 83.8, 134.7, 125.3, 113.8, 104.5, 94.6, 85.5, 104.5,
         94.5, 85.6, 76.2, 66.6, 108.2, 94.5, 85.5, 76.5, 69.0, 68.0, 58.6, 49.7, 85.5, 72.6, 64.9, 57.3, 49.8, 42.1,
         68.8, 58.7, 51.3, 61.2, 53.4, 47.2, 41.8, 35.9, 31.6, 54.8, 47.2, 39.7, 36.1, 32.0, 28.0, 38.2, 32.0, 29.4,
         24.3, 20.3, 16.5]]
    # G[0], h[1], b[2], tw[3], tf[4], r[5], hi[6], d[7], Iy[8], Wel.y[9], Wpl.y[10], iy[11], Avz[12], Iz[13], Wel.z[14], Wpl.z[15], iz[16], Ss[17], It[18], Iw[19], Aa[20]
    # kg / m, mm, mm, mm, mm, mm, mm, mm, cm4, cm3, cm3, cm, cm2, cm4, cm3, cm3, cm, cm, cm4, cm6x103

    # naturalises beam properties
    sec_name = np.take(UCprops[0], [beamIndex])
    sec_G = np.take(UCprops[1], [beamIndex])
    sec_hs = np.take(UCprops[2], [beamIndex])
    sec_b = np.take(UCprops[3], [beamIndex])
    sec_tw = np.take(UCprops[4], [beamIndex])
    sec_tf = np.take(UCprops[5], [beamIndex])
    sec_r = np.take(UCprops[6], [beamIndex])
    sec_hi = np.take(UCprops[7], [beamIndex])
    sec_d = np.take(UCprops[8], [beamIndex])
    sec_Iy = np.take(UCprops[9], [beamIndex])
    sec_Wely = np.take(UCprops[10], [beamIndex])
    sec_Wply = np.take(UCprops[11], [beamIndex])
    sec_iy = np.take(UCprops[12], [beamIndex])
    sec_Avz = np.take(UCprops[13], [beamIndex])
    sec_Iz = np.take(UCprops[14], [beamIndex])
    sec_Welz = np.take(UCprops[15], [beamIndex])
    sec_Wplz = np.take(UCprops[16], [beamIndex])
    sec_iz = np.take(UCprops[17], [beamIndex])
    sec_Ss = np.take(UCprops[18], [beamIndex])
    sec_It = np.take(UCprops[19], [beamIndex])
    sec_Iw = np.take(UCprops[20], [beamIndex])
    sec_Aa = np.take(UCprops[21], [beamIndex])
    sec_Iy = sec_Iy[0]

    EIbeam = 0.210 * sec_Iy / 100

    # takes min value
    b_eff = np.minimum(b_eff1, b_eff2, b_eff3)

    e = (b_st * h_st * (h_st + t_board) / 2) / (b_st * h_st + b_eff * t_board)
    Ieff_beam = I_st + (b_st * h_st * (((h_st + t_board) / 2) - e) ** 2) + (E0_mean_fb / E0_mean_ST * I_b) + (
            E0_mean_fb / E0_mean_ST * b_eff * t_board * e ** 2)
    EI_l1 = Ieff_beam * E0_mean_ST / 1000000000000
    EI_l = (EI_l1 / cc * 1000)
    EI_t = E0_mean_fb * I_b / 1000000000000

    # EC5 CURRENT
    f1_joist = (math.pi / 2 / ((l1 / 1000) ** 2) * np.sqrt(EI_l / m * 1000000))
    F = 1  # kN
    a = 1.8  # mm/kn
    w = F * (l1 ** 3) / 48 / EI_l / 1000000000
    f1_joist_n40 = np.minimum(f1_joist, 40)  # WEIRD STUFF HERE
    n40 = ((((40 / f1_joist_n40) ** 2) - 1) * ((b_floor / l1) ** 4) * (EI_l / EI_t)) ** (1 / 4)
    v = 4 * (0.4 + (0.6 * n40)) / ((m * b_floor * l1 / 1000000) + 200) * 1000
    b = 160 - (a * 40)
    epsilon = 0.02
    v_lim = b ** (f1_joist * epsilon - 1) * 1000


    # EC5 PROPOSED
    lratio = l1 / l2
    ke1j = np.where(lratio>=0.95,1,np.where(lratio>=0.85,1.09,np.where(lratio>=0.75,1.16,np.where(lratio>=0.65,1.21,np.where(lratio>=0.55,1.25,np.where(lratio>=0.45,1.28,np.where(lratio>=0.35,1.32,np.where(lratio>=0.25,1.36,1.41))))))))
    ke2j = np.sqrt(1 + (((l2 / b_floor) ** 4) * EI_t / EI_l))
    f1_joist = ke1j * ke2j * math.pi / 2 / (l1 ** 2) * np.sqrt(EI_l / m) * 1000000000
    f1_beam = ke1b * ke2b * math.pi / 2 / (l1**2) * np.sqrt(EI_l/m)
    mass_beam = sec_G[0]/100
    def_beam = (5 / 384) * (((l1+l2) / 2000 * (m / 100) ) + mass_beam) * (l_beam ** 4) / EIbeam / 1000000000000
    def_joist = 5 / 384 * m / 100 * (l1 ** 4) / EI_l / 1000000000000
    def_sys_mid = def_beam + def_joist
    mid = def_sys_mid
    f1_sys = ke1j * ke2j * 18 / np.sqrt(def_sys_mid)
    # print(f1_sys)

    # calculate R
    # general
    sides_supported = 2
    M_star = m * b_floor * l1 / sides_supported / 1000000
    alpha = np.exp(-0.4 * f1_sys)
    F0 = 700
    a_rms = 0.4 * alpha * F0 / np.sqrt(2) / 2 / epsilon / M_star
    fw = 1.5
    I = 42 * (fw ** 1.43) / (f1_sys ** 1.3)
    k_red = 0.7
    V1_peak = k_red * I / M_star
    k_imp = (0.48 * b_floor / l1 * ((EI_l / EI_t) ** 0.25))
    k_imp = np.where(k_imp > 1, 1, k_imp)
    v_tot_peak = V1_peak * k_imp
    k_imp = np.where(k_imp > 1, 1, k_imp)
    eta = np.where(k_imp <= 1, 0.69, np.where(k_imp >= 1.5, 0.69, 1.52 - (0.55 * k_imp)))
    beta = (0.65 - (0.01 * f1_sys)) * (1.22 - (11 * epsilon)) * eta
    v_rms = beta * v_tot_peak

    R = np.where(f1_sys <= 4.5, 1000, np.where(f1_sys <= 8, a_rms / 0.005, v_rms / 0.0001))

    '''STATIC CHK'''
    #deflection
    qk = 2.5 # assume resi loading
    def_lim = l_beam/250
    w_sls = (((l1+l2) / 2000 * (m / 100 + qk)) + mass_beam)
    def_beam = (5 / 384) * w_sls * (l_beam ** 4) / EIbeam / 1000000000000
    R = np.where(def_beam/def_lim <= 1, R, 1000)

    #moment
    w_uls = (((l1 + l2) / 2000 * (1.35*m / 100 + 1.5*qk)) + 1.35*mass_beam)
    Mr = w_uls*(l_beam/1000)**2/8
    Mc = sec_Wely[0]*py/1000
    R = np.where(Mr / Mc <= 1, R, 1000)

    #shear
    Vc = sec_d[0]*sec_tw[0]*py/3
    Vr = w_uls*l_beam/2
    R = np.where(Vr / Vc <= 1, R, 1000)

    '''ACCEPTANCE CRITERIA'''
    acceptance_class_R = np.where(R >= 32, 7, np.where(R >= 24, 6, np.where(R >= 16, 5, np.where(R >= 12, 4,
                                                                                                 np.where(R >= 8, 3,
                                                                                                          np.where(
                                                                                                              R >= 4, 2,
                                                                                                              1))))))

    # STIFFNESS CRITERIA
    bef = np.minimum(l1 / 1.1 / 1000 * ((EI_t / EI_l) ** 0.25), b_board)
    W_1kN_joist = 1000 * (l1 ** 3) / 48 / EI_l / bef / 1000000000000
    W_1kN_beam = 500 * (l_beam ** 3) / 48 / EIbeam / 1000000000000
    W_1kN_sys = W_1kN_joist + W_1kN_beam

    acceptance_class_W = np.where(W_1kN_sys >= 1.6, 7, np.where(W_1kN_sys >= 1.2, 6, np.where(W_1kN_sys >= 0.8, 5,
                                                                                              np.where(W_1kN_sys >= 0.5,
                                                                                                       4, np.where(
                                                                                                      W_1kN_sys >= 0.25,
                                                                                                      3, 2)))))

    acceptance_class = np.maximum(acceptance_class_R, acceptance_class_W)

    max_h = t_board+np.maximum(h_st, sec_hs)+h_screed
    mass = m + (sec_G[0]/(l1+l2)*2000)
    return R, acceptance_class, sec_G[0], max_h[0], mass

'''X = np.array([[4.8, 8.7, 350, 28.8, 0.32, 21.7], [4.8, 8.7, 350, 28.8, 0.32, 21.7]])
f1 = joist_beam_vibration_level_inc_ver001(5500, 7500, 3000, 3000, X[:, 0], X[:, 1], X[:, 2], 1000, X[:, 4], 11000, 9000, X[:, 5], X[:, 3], 1.8, 88, 1.0, 1, 1, 2, 1.5)
f2 = hybrid_floor_cost_ver001(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], 3000, 3000)
f3 = joist_carbon_inc_ver001(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], 3000, 3000, X[:, 5])
#def joist_beam_vibration_level(b_floor, l_beam, l1, l2, b_st, h_st, cc, b_board, t_board, E0_mean_ST, E0_mean_fb, m, beamIndex, a, b_BSEN, ke1j, ke1b, ke2b, sides_supp, fw):
print("f1 = ", f1)
print("f2 = ", f2)'''



def tables(i):
    breath_i = i[0]
    depth_i = i[1]
    UBnames_i = i[2]

    joist_breath_array = [38, 47, 50, 63, 75, 100]
    joist_depth_array = [97, 122, 140, 147, 170, 184, 195, 220, 235, 250, 300]
    UBnames = ["UB 1100 x 400 x 607", "UB 1100 x 400 x 548", "UB 1100 x 400 x 499", "UB 1100 x 400 x 433",
               "UB 1100 x 400 x 390", "UB 1100 x 400 x 343", "UB 1016 x 305 x 584", "UB 1016 x 305 x 494""",
               "UB 1016 x 305 x 438", "UB 1016 x 305 x 415", "UB 1016 x 305 x 393", "UB 1016 x 305 x 350",
               "UB 1016 x 305 x 314", "UB 1016 x 305 x 272", "UB 1016 x 305 x 249", "UB 1016 x 305 x 222",
               "UB 1000 x 400 x 976", "UB 1000 x 400 x 883", "UB 1000 x 400 x 748", "UB 1000 x 400 x 642",
               "UB 1000 x 400 x 591", "UB 1000 x 400 x 554", "UB 1000 x 400 x 539", "UB 1000 x 400 x 483",
               "UB 1000 x 400 x 443", "UB 1000 x 400 x 412", "UB 1000 x 400 x 371", "UB 1000 x 400 x 321",
               "UB 1000 x 400 x 296", "UB 920 x 420 x 1377", "UB 920 x 420 x 1269", "UB 920 x 420 x 1194",
               "UB 920 x 420 x 1077", "UB 920 x 420 x 970", "UB 920 x 420 x 787", "UB 920 x 420 x 725",
               "UB 920 x 420 x 656", "UB 920 x 420 x 588", "UB 920 x 420 x 537", "UB 920 x 420 x 491",
               "UB 920 x 420 x 449", "UB 920 x 420 x 420", "UB 920 x 420 x 390", "UB 920 x 420 x 368",
               "UB 920 x 420 x 344", "UB 914 x 305 x 576", "UB 914 x 305 x 521", "UB 914 x 305 x 474",
               "UB 914 x 305 x 425", "UB 914 x 305 x 381", "UB 914 x 305 x 345", "UB 914 x 305 x 313",
               "UB 914 x 305 x 289", "UB 914 x 305 x 271", "UB 914 x 305 x 253", "UB 914 x 305 x 238",
               "UB 914 x 305 x 224", "UB 914 x 305 x 201", "UB 840 x 400 x 576", "UB 840 x 400 x 527",
               "UB 840 x 400 x 473", "UB 840 x 400 x 433", "UB 840 x 400 x 392", "UB 840 x 400 x 359",
               "UB 840 x 400 x 329", "UB 840 x 400 x 299", "UB 838 x 292 x 251", "UB 838 x 292 x 226",
               "UB 838 x 292 x 194", "UB 838 x 292 x 176", "UB 760 x 380 x 582", "UB 760 x 380 x 531",
               "UB 760 x 380 x 484", "UB 760 x 380 x 434", "UB 760 x 380 x 389", "UB 760 x 380 x 350",
               "UB 760 x 380 x 314", "UB 760 x 380 x 284", "UB 760 x 380 x 257", "UB 762 x 267 x 220",
               "UB 762 x 267 x 197", "UB 762 x 267 x 173", "UB 762 x 267 x 147", "UB 762 x 267 x 134",
               "UB 690 x 360 x 802", "UB 690 x 360 x 548", "UB 690 x 360 x 500", "UB 690 x 360 x 457",
               "UB 690 x 360 x 419", "UB 690 x 360 x 384", "UB 690 x 360 x 350", "UB 690 x 360 x 323",
               "UB 690 x 360 x 289", "UB 690 x 360 x 265", "UB 690 x 360 x 240", "UB 690 x 360 x 217",
               "UB 686 x 254 x 192", "UB 686 x 254 x 170", "UB 686 x 254 x 152", "UB 686 x 254 x 140",
               "UB 686 x 254 x 125", "UB 610 x 325 x 551", "UB 610 x 325 x 498", "UB 610 x 325 x 455",
               "UB 610 x 325 x 415", "UB 610 x 325 x 372", "UB 610 x 325 x 341", "UB 610 x 325 x 307",
               "UB 610 x 325 x 285", "UB 610 x 325 x 262", "UB 610 x 325 x 241", "UB 610 x 325 x 217",
               "UB 610 x 325 x 195", "UB 610 x 325 x 174", "UB 610 x 325 x 155", "UB 610 x 305 x 238",
               "UB 610 x 305 x 179", "UB 610 x 305 x 149", "UB 610 x 229 x 153", "UB 610 x 229 x 140",
               "UB 610 x 229 x 125", "UB 610 x 229 x 113", "UB 610 x 229 x 101", "UB 610 x 178 x 92",
               "UB 610 x 178 x 82", "UB 533 x 210 x 138", "UB 533 x 210 x 122", "UB 533 x 210 x 109",
               "UB 533 x 210 x 101", "UB 533 x 210 x 92", "UB 533 x 210 x 82", "UB 533 x 165 x 85", "UB 533 x 165 x 74",
               "UB 533 x 165 x 66", "UB 457 x 191 x 106", "UB 457 x 191 x 98", "UB 457 x 191 x 89", "UB 457 x 191 x 82",
               "UB 457 x 191 x 74", "UB 457 x 191 x 67", "UB 457 x 152 x 82", "UB 457 x 152 x 74", "UB 457 x 152 x 67",
               "UB 457 x 152 x 60", "UB 457 x 152 x 52", "UB 406 x 178 x 85", "UB 406 x 178 x 74", "UB 406 x 178 x 67",
               "UB 406 x 178 x 60", "UB 406 x 178 x 54", "UB 406 x 140 x 53", "UB 406 x 140 x 46", "UB 406 x 140 x 39",
               "UB 356 x 171 x 67", "UB 356 x 171 x 57", "UB 356 x 171 x 51", "UB 356 x 171 x 45", "UB 356 x 127 x 39",
               "UB 356 x 127 x 33", "UB 305 x 165 x 54", "UB 305 x 165 x 46", "UB 305 x 165 x 40", "UB 305 x 127 x 48",
               "UB 305 x 127 x 42", "UB 305 x 127 x 37", "UB 305 x 102 x 33", "UB 305 x 102 x 28", "UB 305 x 102 x 25",
               "UB 254 x 146 x 43", "UB 254 x 146 x 37", "UB 254 x 146 x 31", "UB 254 x 102 x 28", "UB 254 x 102 x 25",
               "UB 254 x 102 x 22", "UB 203 x 133 x 30", "UB 203 x 133 x 25", "UB 203 x 102 x 23", "UB 178 x 102 x 19",
               "UB 152 x 89 x 16", "UB 127 x 76 x 13"]
    UBprops = [
        [607.0, 548.0, 499.0, 433.0, 390.0, 343.0, 584.0, 494.0, 438.0, 415.0, 393.0, 350.0, 314.0, 272.0, 249.0, 222.0,
         976.0, 883.0, 748.0, 642.0, 591.0, 554.0, 539.0, 483.0, 443.0, 412.0, 371.0, 321.0, 296.0, 1377.0, 1269.0,
         1194.0, 1077.0, 970.0, 787.0, 725.0, 656.0, 588.0, 537.0, 491.0, 449.0, 420.0, 390.0, 368.0, 344.0, 576.0,
         521.0, 474.0, 425.0, 381.0, 345.0, 313.0, 289.0, 271.0, 253.0, 238.0, 224.0, 201.0, 576.0, 527.0, 473.0, 433.0,
         392.0, 359.0, 329.0, 299.0, 251.0, 226.0, 194.0, 176.0, 582.0, 531.0, 484.0, 434.0, 389.0, 350.0, 314.0, 284.0,
         257.0, 220.0, 197.0, 173.0, 147.0, 134.0, 802.0, 548.0, 500.0, 457.0, 419.0, 384.0, 350.0, 323.0, 289.0, 265.0,
         240.0, 217.0, 192.0, 170.0, 152.0, 140.0, 125.0, 551, 498, 455, 415, 372, 341, 307, 285, 262, 241, 217, 195,
         174, 155, 238.0, 179.0, 149.0, 153.0, 140.0, 125.0, 113.0, 101.0, 92.0, 82.0, 138.0, 122.0, 109.0, 101.0, 92.1,
         82.2, 85.0, 74.0, 66.0, 106.0, 98.3, 89.3, 82.0, 74.3, 67.1, 82.1, 74.2, 67.2, 59.8, 52.3, 85.0, 74.2, 67.1,
         60.1, 54.1, 53.3, 46.0, 39.0, 67.1, 57.0, 51.0, 45.0, 39.1, 33.1, 54.0, 46.1, 40.3, 48.1, 41.9, 37.0, 32.8,
         28.2, 24.8, 43.0, 37.0, 31.1, 28.3, 25.2, 22.0, 30.0, 25.1, 23.1, 19.0, 16.0, 13.0],
        [1138.0, 1128.0, 1118.0, 1108.0, 1100.0, 1090.0, 1056.0, 1036.0, 1026.0, 1020.0, 1016.0, 1008.0, 1000.0, 990.0,
         980.0, 970.0, 1108.0, 1092.0, 1068.0, 1048.0, 1040.0, 1032.0, 1030.0, 1020.0, 1012.0, 1008.0, 1000.0, 990.0,
         982.0, 1093.0, 1093.0, 1081.0, 1061.0, 1043.0, 1011.0, 999.0, 987.0, 975.0, 965.0, 957.0, 948.0, 943.0, 936.0,
         931.0, 927.0, 993.0, 981.0, 971.0, 961.0, 951.0, 943.0, 932.0, 926.6, 923.0, 918.4, 915.0, 910.4, 903.0, 913.0,
         903.0, 893.0, 885.0, 877.0, 868.0, 862.0, 856.0, 859.0, 850.9, 840.7, 834.9, 843.0, 833.0, 823.0, 813.0, 803.0,
         795.0, 786.0, 780.0, 772.0, 779.0, 769.8, 762.2, 754.0, 750.0, 826.0, 772.0, 762.0, 752.0, 744.0, 736.0, 728.0,
         722.0, 714.0, 706.0, 701.0, 695.0, 702.0, 692.9, 687.5, 683.5, 677.9, 711.0, 699.0, 689.0, 679.0, 669.0, 661.0,
         653.0, 647.0, 641.0, 635.0, 628.0, 622.0, 616.0, 611.0, 635.8, 620.2, 612.4, 623.0, 617.2, 612.2, 607.6, 602.6,
         603.0, 599.0, 549.0, 544.5, 539.5, 536.7, 533.1, 528.3, 535.0, 529.0, 525.0, 469.0, 467.2, 463.4, 460.0, 457.0,
         453.4, 465.8, 462.0, 458.0, 454.6, 449.8, 417.0, 412.8, 409.4, 406.4, 402.6, 406.6, 403.2, 398.0, 363.4, 358.0,
         355.0, 351.4, 353.4, 349.0, 310.4, 306.6, 303.4, 311.0, 307.2, 304.4, 312.7, 308.7, 305.1, 259.6, 256.0, 251.4,
         260.4, 257.2, 254.0, 206.8, 203.2, 203.2, 177.8, 152.4, 127.0],
        [410.0, 407.0, 405.0, 402.0, 400.0, 400.0, 314.0, 309.0, 305.0, 304.0, 303.0, 302.0, 300.0, 300.0, 300.0, 300.0,
         428.0, 424.0, 417.0, 412.0, 409.0, 408.0, 407.0, 404.0, 402.0, 402.0, 400.0, 400.0, 400.0, 473.0, 461.0, 457.0,
         451.0, 446.0, 437.0, 434.0, 431.0, 427.0, 425.0, 422.0, 423.0, 422.0, 420.0, 419.0, 418.0, 322.0, 319.0, 316.0,
         313.0, 310.0, 308.0, 309.0, 307.7, 307.0, 305.5, 305.0, 304.1, 303.3, 411.0, 409.0, 406.0, 404.0, 401.0, 403.0,
         401.0, 400.0, 292.0, 293.8, 292.4, 291.7, 396.0, 393.0, 390.0, 387.0, 385.0, 382.0, 384.0, 382.0, 381.0, 266.0,
         268.0, 266.7, 265.2, 264.4, 387.0, 372.0, 369.0, 367.0, 364.0, 362.0, 360.0, 359.0, 356.0, 358.0, 356.0, 355.0,
         254.0, 255.8, 254.5, 253.7, 253.0, 347.0, 343.0, 340.0, 338.0, 335.0, 333.0, 330.0, 329.0, 327.0, 329.0, 328.0,
         327.0, 325.0, 324.0, 311.4, 307.1, 304.8, 229.0, 230.2, 229.0, 228.2, 227.6, 179.0, 178.0, 214.0, 211.9, 210.8,
         210.0, 209.3, 208.8, 166.0, 166.0, 165.0, 194.0, 192.8, 191.9, 191.3, 190.4, 189.9, 155.3, 154.4, 153.8, 152.9,
         152.4, 181.0, 179.5, 178.8, 177.9, 177.7, 143.3, 142.2, 141.8, 173.2, 172.2, 171.5, 171.1, 126.0, 125.4, 166.9,
         165.7, 165.0, 125.3, 124.3, 123.4, 102.4, 101.8, 101.6, 147.3, 146.4, 146.1, 102.2, 101.9, 101.6, 133.9, 133.2,
         101.8, 101.2, 88.7, 76.0],
        [31.0, 28.0, 26.0, 22.0, 20.0, 18.0, 36.0, 31.0, 26.9, 26.0, 24.4, 21.1, 19.1, 16.5, 16.5, 16.0, 50.0, 45.5,
         39.0, 34.0, 31.0, 29.5, 28.4, 25.4, 23.6, 21.1, 19.0, 16.5, 16.5, 76.7, 64.0, 60.5, 55.0, 50.0, 40.9, 38.1,
         34.5, 31.0, 28.4, 25.9, 24.0, 22.5, 21.3, 20.3, 19.3, 36.1, 33.0, 30.0, 26.9, 24.4, 22.1, 21.1, 19.5, 18.4,
         17.3, 16.5, 15.9, 15.1, 32.0, 29.5, 26.4, 24.4, 22.1, 21.1, 19.7, 18.2, 17.0, 16.1, 14.7, 14.0, 34.5, 31.5,
         29.0, 25.9, 23.6, 21.1, 19.7, 18.0, 16.6, 16.5, 15.6, 14.3, 12.8, 12.0, 50.0, 35.1, 32.0, 29.5, 26.9, 24.9,
         23.1, 21.1, 19.0, 18.4, 16.8, 15.4, 15.5, 14.5, 13.2, 12.4, 11.7, 38.6, 35.1, 32.0, 29.5, 26.4, 24.4, 22.1,
         20.6, 19.0, 17.9, 16.5, 15.4, 14.0, 12.7, 18.4, 14.1, 11.8, 14.0, 13.1, 11.9, 11.1, 10.5, 10.9, 10.0, 14.7,
         12.7, 11.6, 10.8, 10.1, 9.6, 10.3, 9.7, 8.9, 12.6, 11.4, 10.5, 9.9, 9.0, 8.5, 10.5, 9.6, 9.0, 8.1, 7.6, 10.9,
         9.5, 8.8, 7.9, 7.7, 7.9, 6.8, 6.4, 9.1, 8.1, 7.4, 7.0, 6.6, 6.0, 7.9, 6.7, 6.0, 9.0, 8.0, 7.1, 6.6, 6.0, 5.8,
         7.2, 6.3, 6.0, 6.3, 6.0, 5.7, 6.4, 5.7, 5.4, 4.8, 4.5, 4.0],
        [55.0, 50.0, 45.0, 40.0, 36.0, 31.0, 64.0, 54.0, 49.0, 46.0, 43.9, 40.0, 35.9, 31.0, 26.0, 21.1, 89.9, 82.0,
         70.0, 60.0, 55.9, 52.0, 51.1, 46.0, 41.9, 40.0, 36.1, 31.0, 27.1, 115.1, 115.1, 109.0, 99.1, 89.9, 73.9, 68.1,
         62.0, 55.9, 51.1, 47.0, 42.7, 39.9, 36.6, 34.3, 32.0, 65.0, 58.9, 54.1, 49.0, 43.9, 39.9, 34.5, 32.0, 30.0,
         27.9, 25.9, 23.9, 20.2, 57.9, 53.1, 48.0, 43.9, 39.9, 35.6, 32.4, 29.2, 31.0, 26.8, 21.7, 18.8, 62.0, 56.9,
         52.1, 47.0, 41.9, 38.1, 33.4, 30.1, 27.1, 30.0, 25.4, 21.6, 17.5, 15.5, 89.9, 63.0, 57.9, 53.1, 49.0, 45.0,
         40.9, 38.1, 34.0, 30.2, 27.4, 24.8, 27.9, 23.7, 21.0, 19.0, 16.2, 69.1, 63.0, 57.9, 53.1, 48.0, 43.9, 39.9,
         37.1, 34.0, 31.0, 27.7, 24.4, 21.6, 19.0, 31.4, 23.6, 19.7, 24.9, 22.1, 19.6, 17.3, 14.8, 15.0, 12.8, 23.6,
         21.3, 18.8, 17.4, 15.6, 13.2, 16.5, 13.6, 11.4, 20.6, 19.6, 17.7, 16.0, 14.5, 12.7, 18.9, 17.0, 15.0, 13.3,
         10.9, 18.2, 16.0, 14.3, 12.8, 10.9, 12.9, 11.2, 8.6, 15.7, 13.0, 11.5, 9.7, 10.7, 8.5, 13.7, 11.8, 10.2, 14.0,
         12.1, 10.7, 10.8, 8.8, 7.0, 12.7, 10.9, 8.6, 10.0, 8.4, 6.8, 9.6, 7.8, 9.3, 7.9, 7.7, 7.6],
        [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
         30, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
         20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
         20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
         20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8,
         8, 8, 8, 8, 8, 8, 8, 8, 8],
        [1028.0, 1028.0, 1028.0, 1028.0, 1028.0, 1028.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0,
         928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 928.0, 862.8, 862.8,
         862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8,
         862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 862.8, 797.1, 797.1, 797.1, 797.1, 797.1,
         797.1, 797.1, 797.1, 797.3, 797.3, 797.3, 797.3, 719.1, 719.1, 719.1, 719.1, 719.1, 719.1, 719.1, 719.1, 719.1,
         719.0, 719.0, 719.0, 719.0, 719.0, 645.9, 645.9, 645.9, 645.9, 645.9, 645.9, 645.9, 645.9, 645.9, 645.9, 645.9,
         645.9, 645.5, 645.5, 645.5, 645.5, 645.5, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0,
         573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 573.0, 501.9, 501.9,
         501.9, 501.9, 501.9, 501.9, 502.0, 502.0, 502.0, 428.0, 428.0, 428.0, 428.0, 428.0, 428.0, 428.0, 428.0, 428.0,
         428.0, 428.0, 380.8, 380.8, 380.8, 380.8, 380.8, 380.8, 380.8, 380.8, 332.0, 332.0, 332.0, 332.0, 332.0, 332.0,
         283.0, 283.0, 283.0, 283.0, 283.0, 283.0, 291.1, 291.1, 291.1, 234.2, 234.2, 234.2, 240.4, 240.4, 240.4, 187.6,
         187.6, 184.6, 162.0, 137.0, 111.8],
        [968.0, 968.0, 968.0, 968.0, 968.0, 968.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0,
         868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 868.0, 812.8, 812.8, 812.8,
         812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 812.8, 822.8, 822.8, 822.8,
         822.8, 822.8, 822.8, 822.8, 822.8, 822.8, 822.8, 822.8, 822.8, 822.8, 757.1, 757.1, 757.1, 757.1, 757.1, 757.1,
         757.1, 757.1, 757.3, 757.3, 757.3, 757.3, 679.1, 679.1, 679.1, 679.1, 679.1, 679.1, 679.1, 679.1, 679.1, 679.0,
         679.0, 679.0, 679.0, 679.0, 605.9, 605.9, 605.9, 605.9, 605.9, 605.9, 605.9, 605.9, 605.9, 605.9, 605.9, 605.9,
         605.5, 605.5, 605.5, 605.5, 605.5, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0,
         533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 533.0, 547.0, 547.0, 475.9, 475.9, 475.9,
         475.9, 475.9, 475.9, 476.0, 476.0, 476.0, 408.0, 408.0, 408.0, 408.0, 408.0, 408.0, 408.0, 408.0, 408.0, 408.0,
         408.0, 360.8, 360.8, 360.8, 360.8, 360.8, 360.8, 360.8, 360.8, 312.0, 312.0, 312.0, 312.0, 312.0, 312.0, 265.0,
         265.0, 265.0, 265.0, 265.0, 265.0, 275.1, 275.1, 275.1, 218.2, 218.2, 218.2, 224.4, 224.4, 224.4, 171.6, 171.6,
         168.6, 146.0, 121.0, 95.8],
        [1624100, 1456640, 1305020, 1136540, 1016360, 878350, 1246070, 1027950, 909170, 853120, 807680, 722960, 644210,
         553840, 481070, 407660, 2348680, 2096410, 1731940, 1450580, 1331030, 1232370, 1202530, 1067480, 966510, 910470,
         813730, 696440, 620310, 3035400, 2901080, 2696760, 2379090, 2104260, 1649860, 1496530, 1339430, 1185310,
         1069610, 970410, 878790, 817410, 745810, 696290, 649060, 1102320, 983010, 886280, 788770, 697370, 626190,
         548840, 504730, 472170, 436840, 407010, 376950, 325790, 1011770, 915080, 813200, 736270, 659650, 591620,
         535820, 481680, 387490, 340800, 280270, 247110, 861550, 776640, 698760, 618840, 545190, 486880, 428860, 383750,
         342050, 279390, 241320, 206650, 169870, 152060, 1063170, 672930, 606230, 546550, 495390, 448880, 403400,
         371040, 326240, 291790, 262680, 235790, 199440, 171780, 151810, 137720, 119440, 559070, 496220, 446080, 401320,
         354790, 319840, 285230, 262280, 237550, 216990, 192360, 169450, 148720, 130550, 210330, 153880, 126730, 126750,
         113390, 100220, 88930, 77390, 64680, 56030, 86120, 76080, 66860, 61550, 55260, 47570, 48580, 41090, 35100,
         48800, 45710, 40990, 37030, 33300, 29360, 36570, 32650, 28910, 25480, 21350, 31520, 27290, 24310, 21580, 18710,
         18270, 15670, 12490, 19450, 16020, 14120, 12050, 10160, 8240, 11690, 9901, 8505, 9577, 8198, 7173, 6512, 5376,
         4466, 6550, 5543, 4420, 4012, 3421, 2848, 2900, 2344, 2109, 1359, 836.5, 474.9],
        [28540, 25820, 23340, 20510, 18470, 16110, 23590, 19840, 17720, 16720, 15890, 14340, 12880, 11180, 9817, 8405,
         42390, 38390, 32430, 27680, 25590, 23880, 23350, 20930, 19100, 18060, 16270, 14060, 12630, 55540, 53080, 49890,
         44840, 40350, 32630, 29960, 27140, 24310, 22160, 20280, 18540, 17330, 15930, 14950, 14000, 22200, 20040, 18250,
         16410, 14660, 13280, 11770, 10890, 10230, 9513, 8896, 8281, 7215, 22160, 20260, 18210, 16630, 15040, 13630,
         12430, 11250, 9021, 8010, 6667, 5919, 20440, 18640, 16980, 15220, 13570, 12240, 10910, 9839, 8861, 7173, 6269,
         5422, 4505, 4054, 25740, 17430, 15910, 14530, 13310, 12190, 11080, 10270, 9138, 8266, 7494, 6785, 5682, 4958,
         4416, 4029, 3524, 15720, 14190, 12940, 11820, 10600, 9677, 8736, 8107, 7411, 6834, 6126, 5448, 4828, 4273,
         6616, 4962, 4139, 4069, 3674, 3274, 2927, 2568, 2145, 1870, 3137, 2794, 2478, 2293, 2073, 1801, 1816, 1553,
         1337, 2081, 1956, 1769, 1610, 1457, 1295, 1570, 1413, 1262, 1121, 949.4, 1512, 1322, 1188, 1062, 929.4, 898.7,
         777.4, 627.9, 1070, 895.4, 795.8, 686.2, 575.1, 472.2, 753.7, 645.8, 560.6, 615.9, 533.7, 471.3, 416.5, 348.3,
         292.7, 504.6, 433.1, 351.6, 308.1, 266, 224.2, 280.4, 230.7, 207.5, 152.8, 109.7, 74.79],
        [33000, 29720, 26810, 23370, 20990, 18270, 28030, 23410, 20740, 19570, 18530, 16580, 14850, 12820, 11340, 9803,
         50290, 45260, 37880, 32090, 29520, 27490, 26820, 23920, 21770, 20460, 18360, 15790, 14250, 67740, 64020, 59910,
         53450, 47750, 38110, 34830, 31360, 27940, 25360, 23090, 21040, 19620, 18010, 16880, 15790, 26290, 23610, 21400,
         19140, 17030, 15360, 13640, 12580, 11790, 10950, 10240, 9547, 8364, 25560, 23270, 20790, 18920, 17040, 15420,
         14040, 12680, 10320, 9182, 7668, 6835, 23750, 21550, 19530, 17400, 15450, 13860, 12320, 11070, 9951, 8231,
         7205, 6236, 5194, 4682, 30930, 20380, 18490, 16800, 15310, 13960, 12630, 11660, 10320, 9333, 8433, 7613, 6504,
         5676, 5046, 4604, 4040, 18650, 16720, 15140, 13750, 12240, 11120, 9984, 9230, 8405, 7726, 6903, 6129, 5417,
         4783, 7516, 5578, 4624, 4657, 4199, 3733, 3338, 2938, 2514, 2198, 3614, 3197, 2830, 2613, 2361, 2060, 2105,
         1810, 1563, 2386, 2231, 2012, 1830, 1651, 1470, 1810, 1625, 1452, 1286, 1095, 1724, 1500, 1345, 1198, 1053,
         1030, 886.9, 723, 1210, 1009, 895.4, 774, 657.9, 542.3, 846.2, 720.2, 623.2, 710.8, 613.7, 539.6, 481.5, 403.6,
         342.7, 566.9, 483.8, 393.6, 353.4, 306.1, 259.6, 314.8, 258.2, 234.5, 171.6, 123.6, 84.43],
        [45.7, 45.5, 45.1, 45.2, 45, 44.6, 40.9, 40.4, 40.4, 40.1, 40.1, 40.3, 40.1, 39.9, 38.9, 37.9, 43.4, 43.1, 42.6,
         42.1, 42, 41.7, 41.8, 41.6, 41.4, 41.6, 41.4, 41.2, 40.5, 41.5, 42.3, 42, 41.6, 41.2, 40.5, 40.2, 39.9, 39.7,
         39.5, 39.3, 39.1, 39, 38.7, 38.5, 38.4, 38.7, 38.4, 38.3, 38.1, 37.8, 37.7, 37.1, 37, 36.9, 36.7, 36.5, 36.3,
         35.6, 37.1, 36.8, 36.7, 36.4, 36.3, 35.9, 35.7, 35.5, 34.8, 34.3, 33.6, 33.1, 34, 33.8, 33.6, 33.4, 33.1, 33,
         32.6, 32.5, 32.2, 31.4, 30.9, 30.5, 30, 29.7, 32.2, 31, 30.8, 30.6, 30.4, 30.2, 30, 29.9, 29.7, 29.3, 29.2, 29,
         28.5, 28, 27.8, 27.6, 27.2, 28.1, 27.9, 27.7, 27.4, 27.2, 27, 26.9, 26.7, 26.6, 26.4, 26.2, 25.9, 25.7, 25.5,
         26.2, 25.9, 25.7, 25.3, 25, 24.9, 24.6, 24.3, 23.4, 23.1, 22.1, 22.1, 21.9, 21.8, 21.6, 21.3, 21.2, 20.7, 20.4,
         19, 19.1, 18.9, 18.8, 18.7, 18.5, 18.7, 18.5, 18.3, 18.2, 17.9, 17, 16.9, 16.8, 16.7, 16.4, 16.4, 16.3, 15.8,
         15, 14.8, 14.7, 14.5, 14.2, 13.9, 13, 12.9, 12.8, 12.5, 12.3, 12.3, 12.4, 12.2, 11.8, 10.9, 10.8, 10.5, 10.5,
         10.3, 10, 8.7, 8.5, 8.4, 7.4, 6.4, 5.3],
        [376.4, 339.5, 313.7, 266.6, 242.1, 216.9, 403.2, 344.5, 299.9, 288.5, 271.2, 235.9, 213.4, 184.5, 180.7, 172.2,
         570.7, 516.4, 438.9, 379.6, 346.3, 328, 316.3, 282.7, 261.8, 235.9, 212.5, 184.5, 181.5, 812.9, 688.7, 647.9,
         583.9, 526.8, 425.5, 394, 355.4, 318.2, 290.4, 264.5, 243.9, 228.5, 215.2, 204.5, 194.1, 364.4, 331.2, 300.1,
         268.3, 242.3, 218.9, 206.6, 190.6, 179.7, 168.6, 160.4, 153.9, 144.8, 300.2, 275.3, 245.7, 226.2, 204.3, 193.3,
         179.8, 165.5, 156.5, 146.8, 132.5, 125.2, 297.6, 270.6, 247.8, 220.6, 199.8, 178.3, 165, 150.4, 137.9, 139,
         129.7, 117.9, 104.7, 97.77, 407.4, 277.4, 251.9, 230.8, 209.9, 193.4, 178.5, 162.9, 146.2, 139.8, 127.5, 116.5,
         119, 109.9, 99.81, 93.43, 87.33, 278.8, 251.8, 228.5, 209.3, 186.5, 171.5, 154.8, 143.9, 132.3, 123.9, 113.5,
         105.2, 95.28, 86.21, 127.2, 96.99, 81.25, 97.12, 90.23, 81.79, 75.87, 71.07, 69.44, 63.39, 84.82, 73.43, 66.73,
         62.05, 57.77, 54.33, 59.14, 54.98, 50.12, 61.47, 55.8, 51.19, 48.01, 43.58, 40.85, 51.56, 46.97, 43.72, 39.26,
         36.39, 47.96, 41.75, 38.48, 34.51, 33.19, 34.54, 29.75, 27.5, 35.63, 31.4, 28.57, 26.71, 25.61, 22.98, 26.6,
         22.57, 20.12, 29.94, 26.48, 23.47, 22.2, 19.95, 18.95, 20.35, 17.73, 16.49, 17.92, 16.82, 15.72, 14.7, 12.93,
         12.5, 9.968, 8.292, 6.541],
        [63470, 56400, 50000, 43420, 38490, 33130, 33430, 26820, 23350, 21700, 20490, 18460, 16230, 14000, 11750, 9545,
         118520, 104970, 85110, 70280, 64010, 59090, 57630, 50710, 45490, 43400, 38580, 33120, 28960, 206350, 189900,
         175050, 152760, 133870, 103310, 93210, 83050, 72770, 65560, 59010, 53980, 50070, 45270, 42120, 39010, 36520,
         32140, 28650, 25190, 21910, 19510, 17040, 15590, 14510, 13300, 12280, 11230, 9423, 67220, 60730, 53670, 48350,
         42960, 38900, 34870, 31190, 12900, 11360, 9067, 7800, 64430, 57760, 51660, 45510, 39930, 35460, 31570, 28000,
         25010, 9443, 8177, 6851, 5457, 4789, 87540, 54300, 48670, 43890, 39500, 35670, 31870, 29430, 25610, 23130,
         20630, 18510, 7645, 6633, 5786, 5185, 4385, 48410, 42590, 38090, 34300, 30170, 27090, 23950, 22060, 19850,
         18430, 16310, 14240, 12370, 10780, 15830, 11410, 9309, 5001, 4508, 3935, 3436, 2917, 1441, 1208, 3869, 3387,
         2942, 2692, 2389, 2007, 1263, 1041, 857.3, 2514, 2346, 2089, 1870, 1671, 1452, 1184, 1046, 912.5, 794.6, 644.9,
         1803, 1545, 1364, 1203, 1021, 634.5, 538, 409.7, 1362, 1108, 968.2, 811, 357.8, 280.2, 1062, 895.6, 764.3, 461,
         388.7, 336.1, 194.1, 155.3, 122.9, 677.3, 570.6, 447.5, 178.5, 148.7, 119.3, 384.6, 307.6, 163.8, 136.7, 89.76,
         55.75],
        [3096, 2771, 2469, 2160, 1924, 1656, 2129, 1736, 1531, 1428, 1352, 1222, 1082, 933.6, 783.6, 636.3, 5538, 4951,
         4082, 3411, 3130, 2896, 2832, 2510, 2263, 2159, 1929, 1656, 1448, 8725, 8238, 7660, 6774, 6003, 4728, 4295,
         3854, 3408, 3085, 2796, 2552, 2373, 2156, 2010, 1866, 2268, 2015, 1813, 1609, 1413, 1267, 1102, 1013, 945.8,
         870.8, 805.6, 739, 621.3, 3271, 2969, 2643, 2393, 2142, 1930, 1739, 1559, 883.7, 773.3, 620.2, 534.8, 3254,
         2939, 2649, 2352, 2074, 1856, 1644, 1466, 1313, 710, 610.2, 513.8, 411.5, 362.2, 4524, 2919, 2638, 2392, 2170,
         1970, 1771, 1640, 1438, 1292, 1159, 1043, 602, 518.6, 454.7, 408.7, 346.6, 2790, 2483, 2241, 2030, 1801, 1627,
         1452, 1341, 1214, 1120, 995, 871, 761.6, 665.7, 1017, 743.1, 610.8, 436.8, 391.7, 343.6, 301.2, 256.3, 161,
         135.8, 361.6, 319.7, 279.2, 256.3, 228.3, 192.2, 152.2, 125.4, 103.9, 259.2, 243.4, 217.7, 195.5, 175.5, 152.9,
         152.5, 135.5, 118.6, 103.9, 84.63, 199.2, 172.1, 152.6, 135.2, 114.9, 88.56, 75.67, 57.79, 157.2, 128.7, 112.9,
         94.8, 56.79, 44.69, 127.3, 108.1, 92.65, 73.59, 62.55, 54.48, 37.91, 30.52, 24.2, 91.97, 77.95, 61.26, 34.94,
         29.18, 23.49, 57.45, 46.19, 32.19, 27.02, 20.24, 14.67],
        [4886, 4358, 3879, 3370, 2995, 2575, 3474, 2818, 2462, 2297, 2167, 1940, 1712, 1469, 1244, 1020, 8838, 7873,
         6459, 5378, 4915, 4546, 4435, 3918, 3529, 3348, 2984, 2554, 2242, 14160, 13130, 12190, 10740, 9497, 7431, 6739,
         6027, 5314, 4799, 4339, 3953, 3671, 3334, 3108, 2884, 3658, 3239, 2901, 2562, 2243, 2003, 1748, 1601, 1491,
         1371, 1267, 1163, 982.4, 5101, 4621, 4100, 3706, 3310, 2984, 2687, 2406, 1383, 1212, 974.7, 842.8, 5082, 4579,
         4119, 3646, 3211, 2865, 2537, 2259, 2020, 1114, 960.1, 808.9, 648.5, 571.2, 7146, 4565, 4114, 3723, 3369, 3054,
         2742, 2532, 2217, 1994, 1786, 1605, 943, 813.3, 712, 639.9, 544.1, 4381, 3889, 3500, 3164, 2799, 2525, 2247,
         2073, 1874, 1728, 1533, 1342, 1172, 1024, 1575, 1145, 938.6, 684.9, 613.9, 537.7, 471.5, 402.4, 258.5, 218.2,
         568.9, 499.7, 435.8, 399.5, 355.6, 300.4, 241.8, 200.3, 166.1, 405.3, 378.8, 338.3, 303.8, 272, 237.2, 240.3,
         213, 186.6, 163, 133.2, 310, 266.9, 236.5, 209, 178.2, 138.9, 118.1, 90.82, 242.9, 198.7, 174.1, 146.5, 89.02,
         70.26, 195.6, 165.5, 141.7, 116, 98.42, 85.42, 60.07, 48.48, 38.83, 141.1, 119.4, 94.15, 54.88, 46.03, 37.3,
         88.25, 70.97, 49.78, 41.61, 31.2, 22.6],
        [9, 8.9, 8.8, 8.8, 8.7, 8.6, 6.7, 6.5, 6.4, 6.4, 6.4, 6.4, 6.3, 6.3, 6, 5.8, 9.7, 9.6, 9.4, 9.2, 9.2, 9.1, 9.1,
         9, 8.9, 9, 9, 9, 8.7, 10.8, 10.8, 10.7, 10.5, 10.3, 10.1, 10, 9.9, 9.8, 9.7, 9.7, 9.7, 9.6, 9.5, 9.4, 9.4, 7,
         6.9, 6.8, 6.8, 6.7, 6.6, 6.5, 6.5, 6.4, 6.4, 6.3, 6.2, 6, 9.5, 9.5, 9.4, 9.3, 9.2, 9.2, 9.1, 9, 6.3, 6.2, 6,
         5.8, 9.3, 9.2, 9.1, 9, 8.9, 8.9, 8.8, 8.7, 8.7, 5.7, 5.6, 5.5, 5.3, 5.2, 9.2, 8.8, 8.7, 8.6, 8.6, 8.5, 8.4,
         8.4, 8.3, 8.2, 8.1, 8.1, 5.5, 5.5, 5.4, 5.3, 5.2, 8.2, 8.1, 8, 8, 7.9, 7.8, 7.8, 7.7, 7.7, 7.7, 7.6, 7.5, 7.4,
         7.3, 7.2, 7, 6.9, 5, 5, 4.9, 4.8, 4.7, 3.5, 3.4, 4.6, 4.6, 4.6, 4.5, 4.5, 4.3, 3.4, 3.3, 3.1, 4.3, 4.3, 4.2,
         4.2, 4.2, 4.1, 3.3, 3.3, 3.2, 3.2, 3.1, 4, 4, 3.9, 3.9, 3.8, 3, 3, 2.8, 3.9, 3.9, 3.8, 3.7, 2.6, 2.5, 3.9, 3.9,
         3.8, 2.7, 2.6, 2.6, 2.1, 2, 1.9, 3.5, 3.4, 3.3, 2.2, 2.1, 2, 3.1, 3, 2.3, 2.3, 2, 1.8],
        [17.6, 16.3, 15.1, 13.7, 12.7, 11.5, 19.9, 17.4, 16, 15.3, 14.7, 13.6, 12.6, 11.3, 10.3, 9.3, 26.4, 24.4, 21.4,
         18.9, 17.7, 16.8, 16.5, 15.2, 14.2, 13.6, 12.6, 11.3, 10.5, 33.6, 32.3, 30.7, 28.2, 25.9, 21.7, 20.3, 18.7,
         17.2, 15.9, 14.9, 13.8, 13.1, 12.3, 11.8, 11.2, 18.9, 17.4, 16.1, 14.8, 13.5, 12.5, 11.3, 10.6, 10.1, 9.6, 9.1,
         8.7, 7.8, 17.1, 15.9, 14.5, 13.5, 12.5, 11.5, 10.7, 10, 10.2, 9.3, 8.1, 7.5, 18.1, 16.8, 15.6, 14.3, 13, 12,
         10.9, 10.1, 9.4, 9.9, 8.9, 8, 7.1, 6.6, 25.3, 18.4, 17.1, 15.9, 14.8, 13.8, 12.8, 12, 11, 10.2, 9.5, 8.8, 9.4,
         8.5, 7.8, 7.3, 6.7, 20, 18.4, 17.1, 15.9, 14.5, 13.5, 12.5, 11.8, 11, 10.3, 9.5, 8.7, 8, 7.4, 10.4, 8.4, 7.4,
         8.7, 8, 7.4, 6.9, 6.3, 5.6, 5, 7.7, 7, 6.4, 6, 5.6, 5.1, 5.8, 5.2, 4.6, 6.5, 6.2, 5.7, 5.3, 4.9, 4.5, 6, 5.5,
         5, 4.6, 4.1, 5.9, 5.3, 4.9, 4.5, 4.1, 4.5, 4, 3.5, 5.2, 4.5, 4.2, 3.8, 3.9, 3.4, 4.5, 4, 3.6, 4.7, 4.2, 3.9,
         3.7, 3.2, 2.9, 4.1, 3.7, 3.2, 3.5, 3.2, 2.8, 3.4, 3, 3.3, 2.9, 2.9, 2.8],
        [5789, 4344, 3253, 2229, 1649, 1104, 7153, 4395, 3181, 2703, 2329, 1718, 1263, 834.8, 581.9, 390, 24410, 18510,
         11550, 7381, 5895, 4834, 4529, 3306, 2543, 2144, 1588, 1033, 763.6, 60450, 52120, 44110, 33050, 24680, 13650,
         10750, 8098, 5923, 4542, 3520, 2692, 2208, 1741, 1452, 1198, 7153, 5358, 4118, 3040, 2206, 1656, 1167, 931.5,
         774.3, 629.9, 518.3, 425.5, 293.9, 6167, 4777, 3511, 2704, 2026, 1510, 1159, 866.2, 745.5, 522.3, 312.2, 226.7,
         7219, 5563, 4279, 3127, 2247, 1675, 1182, 875, 651.1, 620.5, 416.5, 276.8, 166.1, 125, 20610, 7065, 5459, 4228,
         3299, 2569, 1953, 1567, 1120, 833.7, 627.2, 471.5, 479.7, 321.6, 231.2, 178.7, 124.3, 8577, 6477, 5004, 3876,
         2850, 2195, 1645, 1328, 1027, 802.2, 586.1, 418.3, 297.5, 209.5, 802.9, 350.5, 207.6, 315.8, 233.4, 168.2, 123,
         86.67, 71.3, 49.07, 250.9, 178.8, 126.7, 101.3, 75.95, 51.75, 73.98, 48.2, 32.15, 145.5, 121, 90.52, 69.06,
         51.67, 37.03, 89.04, 65.77, 47.54, 33.73, 21.29, 92.42, 62.61, 45.98, 33.21, 23.04, 28.85, 18.94, 10.64, 55.53,
         33.27, 23.72, 15.77, 15.02, 8.736, 34.83, 22.23, 14.76, 31.82, 21.17, 14.79, 12.3, 7.478, 4.833, 24.02, 15.43,
         8.629, 9.665, 6.495, 4.203, 10.39, 6.032, 7.099, 4.471, 3.619, 2.906],
        [185250, 163220, 143400, 123500, 108680, 92700, 81240, 64010, 55290, 51080, 48080, 43010, 37530, 32070, 26620,
         21370, 304410, 265670, 210640, 170660, 154320, 141320, 137550, 119890, 106730, 101450, 89440, 76020, 65890,
         485320, 449320, 409540, 350470, 301870, 225650, 201000, 176960, 153180, 136510, 121870, 110360, 101890, 91390,
         84530, 78000, 77860, 67730, 59790, 52070, 44830, 39610, 34160, 31080, 28840, 26280, 24200, 22000, 18300,
         122460, 109340, 95560, 85320, 75110, 67260, 59910, 53220, 22040, 19230, 15160, 12940, 97850, 86670, 76520,
         66600, 57710, 50690, 44630, 39310, 34650, 13190, 11280, 9364, 7377, 6440, 117630, 67920, 60090, 53420, 47560,
         42470, 37530, 34350, 29550, 26360, 23370, 20760, 8656, 7402, 6407, 5708, 4786, 49560, 42840, 37760, 33460,
         28990, 25720, 22450, 20470, 18250, 16780, 14670, 12690, 10910, 9436, 14430, 10130, 8165, 4456, 3978, 3444,
         2984, 2512, 1239, 1033, 2660, 2311, 1989, 1810, 1596, 1328, 845.4, 688.5, 562.8, 1260, 1172, 1035, 920, 816.5,
         703.8, 589, 516.2, 446.2, 385.7, 309.6, 715.1, 607, 531.6, 465.1, 391, 245.1, 206.1, 154.9, 410.9, 329.2,
         285.1, 236.3, 104.7, 80.97, 233.6, 194.3, 164.1, 101.2, 84.31, 72.26, 44.03, 34.79, 27.18, 103, 85.61, 65.87,
         27.88, 22.92, 18.15, 37.34, 29.32, 15.36, 9.847, 4.687, 1.981],
        [777.4, 702.5, 639.5, 555.4, 501.3, 440.7, 743.7, 629.1, 556.3, 528.7, 500.2, 445.1, 400.4, 346.8, 316.8, 282.8,
         1243.0, 1125.0, 953.4, 817.6, 752.7, 705.8, 687.2, 615.1, 563.7, 525.1, 472.8, 408.8, 377.6, 1753.7, 1616.5,
         1521.5, 1371.5, 1236.6, 1002.0, 922.9, 835.3, 748.1, 682.5, 623.3, 571.4, 534.1, 494.3, 465.6, 437.2, 733.2,
         663.7, 603.9, 542.0, 485.9, 439.7, 398.4, 368.3, 346.1, 322.8, 303.5, 285.6, 255.9, 735.0, 672.0, 603.0, 552.0,
         499.0, 457.0, 419.0, 381.0, 319.3, 288.6, 246.8, 224.0, 742.0, 676.0, 617.0, 553.0, 495.0, 445.0, 400.0, 362.0,
         328.0, 280.7, 250.6, 220.4, 187.2, 170.6, 1021.0, 697.4, 636.0, 582.2, 532.4, 488.6, 445.7, 411.8, 366.8,
         337.0, 305.6, 277.4, 243.8, 216.8, 194.1, 178.4, 159.5, 702.1, 634.8, 578.6, 529.4, 474.3, 433.7, 391.5, 363.6,
         332.7, 308.0, 277.6, 249.3, 222.0, 197.3, 303.3, 228.1, 190.0, 195.7, 178.2, 159.3, 143.9, 128.9, 117.6, 104.4,
         176.2, 155.4, 138.9, 128.7, 117.4, 104.7, 107.9, 95.3, 83.8, 134.7, 125.3, 113.8, 104.5, 94.6, 85.5, 104.5,
         94.5, 85.6, 76.2, 66.6, 108.2, 94.5, 85.5, 76.5, 69.0, 68.0, 58.6, 49.7, 85.5, 72.6, 64.9, 57.3, 49.8, 42.1,
         68.8, 58.7, 51.3, 61.2, 53.4, 47.2, 41.8, 35.9, 31.6, 54.8, 47.2, 39.7, 36.1, 32.0, 28.0, 38.2, 32.0, 29.4,
         24.3, 20.3, 16.5]]
    # G[0], h[1], b[2], tw[3], tf[4], r[5], hi[6], d[7], Iy[8], Wel.y[9], Wpl.y[10], iy[11], Avz[12], Iz[13], Wel.z[14], Wpl.z[15], iz[16], Ss[17], It[18], Iw[19], Aa[20]
    # kg / m, mm, mm, mm, mm, mm, mm, mm, cm4, cm3, cm3, cm, cm2, cm4, cm3, cm3, cm, cm, cm4, cm6x103

    joist_breath = joist_breath_array[breath_i]
    joist_depth = joist_depth_array[depth_i]
    UBnames = UBnames[UBnames_i]

    return joist_breath, joist_depth, UBnames


