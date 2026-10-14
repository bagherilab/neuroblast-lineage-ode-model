"""Model parameters shared by all figures."""
import numpy as np

# -------- Time grid --------
t_span = [0, 48]                             # simulation start/end time [hours]
t_eval = np.linspace(t_span[0], t_span[1], 500)  # output sample times [hours]

# -------- Core volumes/thresholds/floors (Models 1 & 2) --------
nb_vol       = 905.0                         # reference NB volume unit (per cell) [volume]
V0           = nb_vol                        # initial/target average NB volume [volume]
n            = 3                             # Hill / sat-power exponent for division (dimensionless)
k_max_GMC    = 1.0/8.0                       # maximum GMC division rate [1/hour]
V_thresh_NB  = 1.25 * nb_vol                 # NB division threshold volume [volume]
V_thresh_GMC = (nb_vol * 1.25 * 0.2) * 2     # GMC division threshold volume [volume]
V_floor_NB   = 0.25 * nb_vol                 # NB minimum volume required to divide [volume]
V_floor_GMC  = 0.25 * (nb_vol * 1.25 * 0.2)  # GMC minimum volume required to divide [volume]

# -------- Rates --------
k_Neuron = 1.0 / 48.0                        # maturation rate: Im → Mat neurons [1/hour]
g_GMC    = (nb_vol * 1.25 * 0.2) / 9.0       # GMC growth rate per GMC cell [volume/hour/cell]

# -------- WT calibration at V0 --------
k_star   = 1.0 / 1.5                         # desired WT NB division rate at V0 (CCD=1.5 h) [1/hour]
core0    = min(1.0, (V0 / V_thresh_NB) ** n) # unsaturated sat-power value at V0 (dimensionless)
k_max_NB = k_star / core0                    # calibrated max NB division rate [1/hour]
g_NB_WT  = 0.2 * k_star * V0                 # NB growth per NB cell to hold Vavg≈V0 [volume/hour/cell]
                                            # 0.2*Vavg removed per asymmetric NB division in model → balance

# -------- Model 2 dynamic threshold bits --------
V_thresh_base = V_thresh_NB                   # baseline NB threshold used by Model 2 [volume]
delta_thresh  = 0.03 * V_thresh_base          # threshold drop per symmetric NB division [volume/division]
V_thresh_min  = V_floor_NB                    # lower bound on the NB threshold [volume]
g_NB_base_WT  = g_NB_WT                       # base NB growth used by Model 2 WT [volume/hour/cell]

# -------- Initial conditions used by other wrappers --------
y0_8 = [1.0, V0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]               # 8-state models: [N_NB, V_NB, N_GMC, V_GMC, N_Im, V_Im, N_Mat, V_Mat]
                                                             # counts in cells, volumes in [volume]
y0_9 = [1.0, V0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, V_thresh_base] # 9-state models add Vth_eff [volume]

# -------- Model 3 and 4 volume-sensitive growth default--
alpha_value = 3.0

# -------- Model 5 (NB self-repression) defaults --------
K_self   = 2.0                               # half-saturation NB count for self-repression [cells]
beta_value = 4.0                                 # β: exponent/shape parameter for Model 5 (dimensionless)

# -------- Base model (supplemental figure) --------
base_k_NB     = 1/1.5
base_k_GMC    = 1/9.0
base_k_Neuron = 1/48
base_y0       = [1, 0, 0, 0]                 # [N_NB, N_GMC, N_ImNeuron, N_MatNeuron]

# -------- Genotypes: (label, sym_frac, g_nb_base_scale) --------
GENOS = [
    ("WT",         0.0,  1.0),
    ("mud mutant", 0.15, 1.0),
    ("nanobody",   0.15, 0.8),
]
