DL_test = 4.0
DT_test = 9.75
DL_actual = 4.0
DT_actual = 8.8

threshold_rel = 0.1

#shift_max = 2.0
shift_max = 1.5
#shift_max = 1.0
#shift_max = 0.5
shift_step = 0.1

angle_sim_step = 0.1

timeTickSF = 0.4
driftVel_data = 0.1571 # DATA
#driftVel_MC = 0.157565 # MC (at 493.8 V/cm)
driftVel_MC = 0.157560  # Value from simparams.jsonnet
#driftVel_MC = 0.1565 # Value in wcls-multitpc-sim-drift-simchannel-omit-noise.fcl
#driftVel_MC = 0.1544 # MC down by 2% (just to see)
#driftVel_MC = 0.1560 # MC down by 1% (just to see)
wirePitch = 0.3

N_wires = 11
N_wires_fit = 7
N_wires_fine = 25*N_wires
N_ticks = 401
N_ticks_fit = 321
N_ticks_fine = 5*N_ticks

# Determine loop ranges that avoid wires/ticks with artifacts
N_wires_start = ((N_wires - 1) // 2) - ((N_wires_fit - 1) // 2)
N_wires_end   = ((N_wires - 1) // 2) + ((N_wires_fit - 1) // 2) + 1
N_ticks_start = ((N_ticks - 1) // 2) - ((N_ticks_fit - 1) // 2)
N_ticks_end   = ((N_ticks - 1) // 2) + ((N_ticks_fit - 1) // 2) + 1

offset_distance = 14.5
AC_distance = 148.275

driftVel = None
ticks_drift_A = None
ticks_drift_C = None

def set_drift_params(isdata):
    driftVel = driftVel_data if isdata else driftVel_MC
    # These are the only two constants that depend on drift velocity
    ticks_drift_A = offset_distance / driftVel / timeTickSF
    ticks_drift_C = (AC_distance - offset_distance) / driftVel / timeTickSF

    return driftVel, ticks_drift_A, ticks_drift_C
