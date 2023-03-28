DL_test = 4.0
DT_test = 9.75

threshold_rel = 0.1

shift_max = 2.0
shift_step = 0.1

angle_sim_step = 0.1

timeTickSF = 0.4
driftVel = 0.1571 # DATA
#driftVel = 0.157565 # MC (at 493.8 V/cm)
wirePitch = 0.3
DL_actual = 4.0
DT_actual = 8.8

N_wires = 11
N_wires_fit = 7
N_wires_fine = 25*N_wires
N_ticks = 401
N_ticks_fit = 321
N_ticks_fine = 5*N_ticks

offset_distance = 14.5
AC_distance = 148.275
ticks_drift_A = offset_distance/driftVel/timeTickSF
ticks_drift_C = (AC_distance - offset_distance)/driftVel/timeTickSF
