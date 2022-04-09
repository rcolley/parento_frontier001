import numpy as np
import math as math
import sectionLibrary as sl

def detailed_composite(gk, qk, lx, ly, no_sec, deck_type, conc_type, deck_t, slab_h, fire_t, bm_index, secOrPri, R_Gk, R_Qk):
    #(gk, qk, lx, ly, no_sec(range 0 to 3.0 ceiling'd), deck_type(0=51+), conc_type (range 0 to 2, <1=NW), deck_t (range 0 to 3, <1=0.9, <2=1.0, <3=1.2), slab_h (range 100 to 210, floored to 10mm inc), fire_t(60, 90, 120))
    #paras
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

    #bay-geometry
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
    #deck_crossSec_area = np.take(deck_attributes, 2)
    deck_crossSec_area = deck_attributes[:, 2]
    slab_mesh_span = np.take(mesh_60min_5kPa_1mm, slab_index)

    max_deck_span = np.take(max_deck_span_60min_5kPa_1mm, slab_index)

    gate001 = np.where(sec_spacing<=max_deck_span, 0, 1)#deck span check max_deck_span_60min_5kPa_1mm[slab_index]
    gate002 = np.where(fire_t==60, 0, 1)#fire period check

    #beam selection
    bm_index = np.floor(bm_index)
    bm_index = bm_index.astype(int)
    beam_identity = sl.UBnames(bm_index)
    beam_attributes = sl.UBsectionAgregator(bm_index)

    # naturalises beam properties
    sec_name = beam_identity
    bm_G = beam_attributes[0]
    bm_hs = beam_attributes[1]
    bm_b = beam_attributes[2]
    bm_tw = beam_attributes[3]
    bm_tf = beam_attributes[4]
    bm_r = beam_attributes[5]
    bm_hi = beam_attributes[6]
    bm_d = beam_attributes[7]
    bm_Iy = beam_attributes[8]
    bm_Wely = beam_attributes[9]
    bm_Wply = beam_attributes[10]
    bm_iy = beam_attributes[11]
    bm_Avz = beam_attributes[12]
    bm_Iz = beam_attributes[13]
    bm_Welz = beam_attributes[14]
    bm_Wplz = beam_attributes[15]
    bm_iz = beam_attributes[16]
    bm_Ss = beam_attributes[17]
    bm_It = beam_attributes[18]
    bm_Iw = beam_attributes[19]
    bm_Aa = beam_attributes[20]


    #loads
    cll = 1.5 #kPa
    bm_wt = bm_G

    #combinations
    Fd1 = gammaG*(bm_G/sec_spacing + np.take(deck_attributes, 3)) + gammaG*(slab_wt_wet+cll) #construction stage ULS
    Fd = xi*gammaG*(bm_G/sec_spacing + gk + slab_and_deck_wt_dry) + gammaQ*qk #composite stage ULS
    g1 = slab_and_deck_wt_dry + bm_G/sec_spacing #frame selfweight dry
    g2 = gk #superdead
    q1 = qk #live

    #bm and v
    #construction
    R_cons = 0
    M_Ed_sec_cons = np.where(secOrPri == 0, Fd1 * Lmax**2 / 8, np.where(no_sec==1, 0, np.where(no_sec==2, R_cons*Lmin/4, np.where(no_sec==3, R_cons*Lmin/3, np.where(no_sec==4, R_cons*3/Lmin * Lmax**2 / 8, R_cons*4/Lmin * Lmax**2 / 8)))))
    V_Ed_sec_cons = Fd1 * Lmax / 2

    #composite
    M_Ed_sec_comp = Fd * Lmax**2 / 8
    V_Ed_sec_comp = Fd * Lmax / 2

    #cross-section classification
    fy = np.where(np.maximum(bm_tf, bm_tw)<=16, fy, np.where(np.maximum(bm_tf, bm_tw)<=40, fy-10, fy-20))
    epsilon = np.sqrt(235/fy)
    c_tf_ratio = 0.5*(bm_b - bm_tw - (2*bm_r))/bm_tf #outstand compression flange
    classification_outstand = np.where(c_tf_ratio <= 9*epsilon, 1, np.where(c_tf_ratio <= 10*epsilon, 2, np.where(c_tf_ratio <= 14*epsilon, 3, 4)))
    c_tw_ratio = bm_d/bm_tw
    classification_internal = np.where(c_tw_ratio <= 72*epsilon, 1, np.where(c_tw_ratio <= 83*epsilon, 2, np.where(c_tw_ratio <= 124*epsilon, 3, 4)))
    classification = np.maximum(classification_internal, classification_outstand)

    #design resistance
    #shear buckling
    eta_vbuck = 1
    gate003 = np.where(bm_hi/bm_tw*xi/epsilon/72<=1, 0, 1)

    #vertical shear resistance
    V_c_Rd = bm_Avz*fy/np.sqrt(3)/gammaM0
    V_uc = V_Ed_sec_comp / V_c_Rd
    gate004 = np.where(V_uc <= 1, 0, 1)

    #bending
    gate005 = np.where(V_uc <= 0.5, 0, 1) #low shear chk
    gate006 = np.where(classification <= 2, 0, 1) #plastic chk
    M_c_Rd = bm_Wply*fy/gammaM0/1000
    M_uc = M_Ed_sec_comp / M_c_Rd
    gate007 = np.where(M_uc <= 1, 0, 1)

    #shear connectors
    #TO BE REVIEWED FOR ECONOMISE
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
    Npl_a = fy*bm_Aa

    #compressive force in flange
    Nc_comp = np.minimum(Ncf, Npl_a)

    #resistance of the shear connectors
    deck_pitch = 150
    no_studs_half = 1000*Lmax/2/deck_pitch
    Nc_studs = no_studs_half*P_Rd_2

    #degree of shear connection
    eta_degree = np.minimum(1, Nc_studs / Nc_comp)
    gate011 = np.where(eta_degree >= eta_degree_min, 0, 1)

    #design resistance of the cross-section for the composite stage
    #vertical shear resistance
    V_uc = V_Ed_sec_cons / V_c_Rd
    gate012 = np.where(V_uc <= 1, 0, 1)

    #resistance in bending
    gate013 = np.where(V_uc/2 <= 1, 0, 1)
    xpl = (Npl_a-Nc_studs)/2/fy/bm_b*1000
    PNA_text = np.where(Npl_a <= Ncf, "PNA lies in top flange", np.where(xpl < bm_tf, "PNA lies in top flange", "PNA lies in web"))
    PNA_index = np.where(Npl_a <= Ncf, 0, np.where(xpl < bm_tf, 1,2))
    print(PNA_text)

    #PNA in the top flange
    F1a = Nc_studs
    F6b = 2*fy*bm_b*xpl
    F7b = Npl_a
    M_A = 0

    #PNA  in the concrete slab
    Mpl_Rd_c = (Npl_a*((bm_hs/2)+slab_h-(Npl_a/Ncf*hc/2)))/1000
    xc = Npl_a/Ncf*hc
    Mpl_Rd_c2 = bm_hs/2 + slab_h - xc/2 #potentially more accurate for PNA in conc

    #PNA  in the top flange
    hd = hp
    Mpl_Rd_f = ((Npl_a*bm_hs/2)+(Ncf*((hc/2)+hd)))/1000

    #PNA  in the web
    Mpl_a_Rd = M_c_Rd
    hw = bm_hs-(2*bm_tf)
    Nw = 0.95*fy*bm_tw*hw
    Mpl_Rd_w = (Mpl_a_Rd + Ncf*((hc+2*hd+bm_hs)/2) - (Ncf**2*bm_hs/Nw/4))/1000

    Mpl_Rd =np.where(PNA_index == 0, Mpl_Rd_c, np.where(PNA_index == 1, Mpl_Rd_f, Mpl_Rd_w))

    #PNA with partitial
    MRd = Mpl_a_Rd + (Mpl_Rd-Mpl_a_Rd)*eta_degree

    #longitudinal shear resistance of the slab
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

    #verification at SLS
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
    y_el_perm = ((bm_Aa*bm_hs/2) + (beff*(slab_h-hp)*((bm_hs+(slab_h+hp)/2))/2)/n_perm) / (bm_Aa+(beff*(slab_h-hp)/n_perm))
    Ieq_perm = 10000*bm_Iy + 1000*beff*((slab_h-hp)**3)/12/n_perm + 100*bm_Aa*((y_el_perm-(bm_hs/2))**2) + 1000*beff*(slab_h-hp)/n_perm*((bm_hs+((slab_h+hp)/2)-y_el_perm)**2)

    Ieq1 = 10000*bm_Iy
    Ieq2 = 1000*beff*((slab_h-hp)**3)/12/n_perm
    Ieq3 = 100*bm_Aa*((y_el_perm-(bm_hs/2))**2)
    Ieq4 = 1000*beff*(slab_h-hp)/n_perm*((bm_hs+((slab_h+hp)/2)-y_el_perm)**2)

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
    y_el_dyn = ((bm_Aa*bm_hs/2) + (beff*(slab_h-hp)*((bm_hs+(slab_h+hp)/2))/2)/nd) / (bm_Aa+(beff*(slab_h-hp)/nd))
    Ieq_dyn = 10000*bm_Iy + 1000*beff*((slab_h-hp)**3)/12/nd + 100*bm_Aa*((y_el_dyn-(bm_hs/2))**2) + 1000*beff*(slab_h-hp)/nd*((bm_hs+((slab_h+hp)/2)-y_el_dyn)**2)
    def_dyn = 5 / 384 * (g1+g2+0.1*q1) * sec_spacing * (Lmax * 1000) ** 4 / Ea / Ieq_dyn
    fn = 18.07/np.sqrt(def_dyn)
    fn_UC = 4.5 / fn
    gate017 = np.where(fn_UC <= 1, 0, 1)


    #quants
    vol_steel = bm_G / sec_spacing / 7850
    vol_conc = slab_conc_vol
    vol_rebar = At_prov/1000000
    vol_galv = deck_crossSec_area/1000000
    vol_timb = 0*deck_index
    h = bm_hs+slab_h

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
    pri_tw = beam_attributes[3]
    pri_tf = beam_attributes[4]
    pri_r = beam_attributes[5]
    pri_hi = beam_attributes[6]
    pri_d = beam_attributes[7]
    pri_Iy = beam_attributes[8]
    pri_Wely = beam_attributes[9]
    pri_Wply = beam_attributes[10]
    pri_iy = beam_attributes[11]
    pri_Avz = beam_attributes[12]
    pri_Iz = beam_attributes[13]
    pri_Welz = beam_attributes[14]
    pri_Wplz = beam_attributes[15]
    pri_iz = beam_attributes[16]
    pri_Ss = beam_attributes[17]
    pri_It = beam_attributes[18]
    pri_Iw = beam_attributes[19]
    pri_Aa = beam_attributes[20]


    '''loads'''
    cll = 1.5 #kPa
    bm_wt = pri_G

    '''combinations'''
    Fd1 = gammaG*(pri_G/sec_spacing + np.take(deck_attributes, 3)) + gammaG*(slab_wt_wet+cll) #construction stage ULS
    Fd = xi*gammaG*(pri_G/sec_spacing + gk + slab_and_deck_wt_dry) + gammaQ*qk #composite stage ULS
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
    fy = np.where(np.maximum(pri_tf, pri_tw)<=16, fy, np.where(np.maximum(pri_tf, pri_tw)<=40, fy-10, fy-20))
    epsilon = np.sqrt(235/fy)
    c_tf_ratio = 0.5*(pri_b - pri_tw - (2*pri_r))/pri_tf #outstand compression flange
    classification_outstand = np.where(c_tf_ratio <= 9*epsilon, 1, np.where(c_tf_ratio <= 10*epsilon, 2, np.where(c_tf_ratio <= 14*epsilon, 3, 4)))
    c_tw_ratio = pri_d/pri_tw
    classification_internal = np.where(c_tw_ratio <= 72*epsilon, 1, np.where(c_tw_ratio <= 83*epsilon, 2, np.where(c_tw_ratio <= 124*epsilon, 3, 4)))
    classification = np.maximum(classification_internal, classification_outstand)

    '''design resistance'''
    #shear buckling
    eta_vbuck = 1
    gate003 = np.where(pri_hi/pri_tw*xi/epsilon/72<=1, 0, 1)

    #vertical shear resistance
    V_c_Rd = pri_Avz*fy/np.sqrt(3)/gammaM0
    V_uc = V_Ed_sec_comp / V_c_Rd
    gate004 = np.where(V_uc <= 1, 0, 1)

    #bending
    gate005 = np.where(V_uc <= 0.5, 0, 1) #low shear chk
    gate006 = np.where(classification <= 2, 0, 1) #plastic chk
    M_c_Rd = pri_Wply*fy/gammaM0/1000
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

X = np.array([[2.5, 0.0, 0.0, 0.5, 135, 0], [1.5, 0.0, 0.0, 0.5, 135, 179]])
f1 = detailed_composite_pri(0.85, 3.5, 7.5, 9, X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], 60, X[:,5])
#                      (gk, qk, lx, ly, no_sec, deck_type, conc_type, deck_t, slab_h, fire_t, bm_index)
# (gk, qk, lx, ly,
# no_sec(range 0 to 5.0 ceiling'd), deck_type(0=51+), conc_type (range 0 to 2, <1=NW), deck_t (range 0 to 3, <1=0.9, <2=1.0, <3=1.2), slab_h (range 100 to 210, floored to 10mm inc), fire_t(60, 90, 120), bm_index(0-179))
print("f1 = ", f1)