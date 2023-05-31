import numpy as np
from .test_statistics import calc_chisq, calc_test_statistic
from .consts import *

def diffusion_grid_scan(DL_min, DL_max, DL_step, DT_min, DT_max, DT_step, 
                        num_angle_bins, input_signal, 
                        anode_hist, anode_uncert_hist, cathode_hist, cathode_uncert_hist,
                        test_statistic='chi2',
                        verbose=True):

    min_test_stat = np.inf
    min_numvals = 0.0
    all_shifts_result = np.zeros((num_angle_bins, N_wires))
    all_shifts_actual = np.zeros((num_angle_bins, N_wires))
    all_shifts_test   = np.zeros((num_angle_bins, N_wires))

    test_stat_values = np.zeros((int((DL_max-DL_min)/DL_step+1), int((DT_max-DT_min)/DT_step+1)))
    num_values   = np.zeros((int((DL_max-DL_min)/DL_step+1), int((DT_max-DT_min)/DT_step+1)))
    row = 0

    if verbose:
        print('[GRID_SCAN] Calculating test statistic', test_statistic)

    for DL in np.arange(DL_min, DL_max+DL_step, DL_step):
        col = 0
        for DT in np.arange(DT_min, DT_max+DT_step, DT_step):
            test_stat = 0.0
            numvals = 0.0
            all_shifts = np.zeros((num_angle_bins, N_wires))
            for k in range(0, num_angle_bins):
                temp_test_stat, temp_numvals, shift_vec = calc_test_statistic(
                    input_signal[k], 
                    anode_hist[k], anode_uncert_hist[k], 
                    cathode_hist[k], cathode_uncert_hist[k], 
                    DL, DT,
                    test_statistic
                )
                print('[GRID_SCAN] temp_test_stat type:', type(temp_test_stat))
                print('[GRID_SCAN] shift_vec:', shift_vec)
                test_stat += temp_test_stat
                numvals += temp_numvals
                all_shifts[k, :] = shift_vec
                if verbose:
                    print('test_stat', temp_test_stat)
                    print('    %d %.2f' % (k, temp_test_stat))
                    with np.printoptions(precision=1, sign=' ', floatmode='fixed', suppress=True):
                        print('   ', k, shift_vec)
            test_stat_values[row, col] = test_stat
            num_values[row, col] = numvals
            if test_stat < min_test_stat:
                min_test_stat = test_stat
                min_numvals = numvals
                all_shifts_result = all_shifts
            if abs(DL-DL_actual) < DL_step and abs(DT-DT_actual) < DT_step:
                all_shifts_actual = all_shifts
            if abs(DL-DL_test) < DL_step and abs(DT-DT_test) < DT_step:
                all_shifts_test = all_shifts
            print('%.2f %.2f %.2f' % (DL, DT, test_stat))
            col += 1
        row += 1

    return test_stat_values, min_test_stat, min_numvals, all_shifts_result, all_shifts_actual
