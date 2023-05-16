import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fftpack
from scipy import stats
from scipy import ndimage
from numba import jit
import uproot

from lardiff.input_checks import check_input_filename, check_input_range
from lardiff.waveform_functions import create_signal
from lardiff.consts import *
from lardiff.grid_scan import diffusion_grid_scan

def measure_diffusion(input_filename, angle_range, dl_range, dt_range, is_data=False):
    input_file = uproot.open(input_filename)

    # angle_range is of the form [min, max, step]
    angle_bins = [[x, x + angle_range[2]] for x in range(angle_range[0], angle_range[1], angle_range[2])]
    print('angle_bins', angle_bins)
    num_angle_bins = len(angle_bins)
    angle_sim_step = 0.1

    input_signal = np.zeros((num_angle_bins, N_ticks_fine, N_wires_fine))
    anode_hist        = np.zeros((num_angle_bins, N_ticks, N_wires))
    anode_uncert_hist = np.zeros((num_angle_bins, N_ticks, N_wires))
    cathode_hist        = np.zeros((num_angle_bins, N_ticks, N_wires))
    cathode_uncert_hist = np.zeros((num_angle_bins, N_ticks, N_wires))

    #for bin_idx, angles in enumerate(range(num_angle_bins)):
    for bin_idx, angles in enumerate(angle_bins):
        print('bin_idx, angles', bin_idx, angles)
        # Generate a sequence of angles at the center of each bin
        #start_angle = angle_bins[bin_idx][0]# + angle_sim_step/2
        #end_angle = angle_bins[bin_idx][1]# - angle_sim_step/2
        start_angle = angles[0]
        end_angle   = angles[1]

        # Each angle bin is finely split into sub-bins with step size of angle_sim_step
        num_angles = int((end_angle - start_angle) / angle_sim_step) + 1
        signal_angles = np.linspace(start_angle + angle_sim_step/2, end_angle - angle_sim_step/2, num_angles)

        # Calculate the signal for this set of angles and store it in the input_sig array
        input_signal[bin_idx] = create_signal(signal_angles)
        anode_hist[bin_idx]        = np.swapaxes(input_file['AnodeTrackHist2D_%dto%d'       % (start_angle, end_angle)].values(), 0, 1)
        anode_uncert_hist[bin_idx] = np.swapaxes(input_file['AnodeTrackUncertHist2D_%dto%d' % (start_angle, end_angle)].values(), 0, 1)
        cathode_hist[bin_idx]        = np.swapaxes(input_file['CathodeTrackHist2D_%dto%d'       % (start_angle, end_angle)].values(), 0, 1)
        cathode_uncert_hist[bin_idx] = np.swapaxes(input_file['CathodeTrackUncertHist2D_%dto%d' % (start_angle, end_angle)].values(), 0, 1)

        anode_hist_nonzero_indices = np.nonzero(anode_hist)
        anode_uncert_hist_nonzero_indices = np.nonzero(anode_uncert_hist)
        anode_hist_nonzero = anode_hist[anode_hist_nonzero_indices]
        anode_uncert_hist_nonzero = anode_uncert_hist[anode_uncert_hist_nonzero_indices]
        cathode_hist_nonzero_indices = np.nonzero(cathode_hist)
        cathode_uncert_hist_nonzero_indices = np.nonzero(cathode_uncert_hist)
        cathode_hist_nonzero = cathode_hist[cathode_hist_nonzero_indices]
        cathode_uncert_hist_nonzero = cathode_uncert_hist[cathode_uncert_hist_nonzero_indices]
        print('anode_hist_nonzero', anode_hist_nonzero)
        print('anode_uncert_hist_nonzero', anode_uncert_hist_nonzero)
        print('cathode_hist_nonzero', cathode_hist_nonzero)
        print('cathode_uncert_hist_nonzero', cathode_uncert_hist_nonzero)

    DL_min  = dl_range[0]
    DL_max  = dl_range[1]
    DL_step = dl_range[2]

    DT_min  = dt_range[0]
    DT_max  = dt_range[1]
    DT_step = dt_range[2]

    #min_chisq = np.inf
    #min_numvals = 0.0
    #all_shifts_result = np.zeros((num_angle_bins, N_wires))
    #all_shifts_actual = np.zeros((num_angle_bins, N_wires))
    #all_shifts_test   = np.zeros((num_angle_bins, N_wires))

    #chisq_values = np.zeros((int((DL_max-DL_min)/DL_step+1), int((DT_max-DT_min)/DT_step+1)))
    #num_values   = np.zeros((int((DL_max-DL_min)/DL_step+1), int((DT_max-DT_min)/DT_step+1)))
    #row = 0

    chisq_values, min_numvals, all_shifts_result = diffusion_grid_scan(
        DL_min, DL_max, DL_step, DT_min, DT_max, DT_step, num_angle_bins,
        input_signal, anode_hist, anode_uncert_hist, cathode_hist, cathode_uncert_hist
    )

    zoom_factor = 100
    delta_chisq_values = ndimage.zoom(chisq_values - np.amin(chisq_values), zoom_factor)
    point_y, point_x = np.unravel_index(np.argmin(delta_chisq_values), delta_chisq_values.shape)
    DL_result = (DL_step*point_y/zoom_factor) + DL_min-DL_step/2.0
    DT_result = (DT_step*point_x/zoom_factor) + DT_min-DT_step/2.0

    print('DL, DT:', DL_result, DT_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", type=str, #nargs=1,
                        help="Input .root file from WaveformStudy")
    parser.add_argument("--angle_range", 
                        type=int, nargs=3, default=[26,80,2], help='''\
                        Three ints corresponding to (min, max, step) of angle scan range in degrees
                        Default values: [26,80,2]''')
    parser.add_argument("--dl_range", 
                        type=float, nargs=3, default=[3.0,5.0,0.25], help='''\
                        Three floats corresponding to (min, max, step) DL scan range in cm^2/s
                        Default values: [3.0,5.0,0.25]''')
    parser.add_argument("--dt_range", 
                        type=float, nargs=3, default=[7.0,10.0,0.25], help='''\
                        Three floats corresponding to (min, max, step) DT scan range in cm^2/s
                        Default values: [7.0,10.0,0.25]''')
    parser.add_argument("--test_statistic", 
                        type=str, nargs=1, default='chi2', help='''\
                        Which test statistic to use for evaluating statistical uncertainties
                        Default value: chi2''')
    parser.add_argument("--is_data", 
                        type=bool, nargs=1, default=False, help='''\
                        Bool for determinig whether to use data parameters, particularly drift velocity. 
                        Default value: False.''')
    args = parser.parse_args()

    # Input checking 
    try:
        args.input_filename = check_input_filename(args.input_filename)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    
    try:
        check_input_range(args.angle_range)
        check_input_range(args.dl_range)
        check_input_range(args.dt_range)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    measure_diffusion(args.input_filename, args.angle_range, args.dl_range, args.dt_range, args.is_data)














