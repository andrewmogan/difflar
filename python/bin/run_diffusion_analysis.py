import argparse
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fftpack
from scipy import stats
from scipy import ndimage
from numba import jit
import uproot
import time
import yaml
import pickle

from lardiff.waveform_functions import create_signal
from lardiff.consts import *
from lardiff.grid_scan import diffusion_grid_scan
from lardiff.plotting_functions import make_test_statistic_plot

# Set project root directory two directories up
LARDIFF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = "{:s}/config/default.yaml".format(LARDIFF_DIR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", type=str, 
                        help="Input .root file from WaveformStudy")
    parser.add_argument('-c', '--config', type=str, 
                        default='{}/config/default_mc.yaml'.format(LARDIFF_DIR),
                        help='Path to YAML config file')
    
    args = parser.parse_args()
    return args 

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_config(config):

    required_params = ['DL_min', 'DL_max', 'DL_step',
                       'DT_min', 'DT_max', 'DT_step',
                       'angle_min', 'angle_max', 'angle_step',
                       'test_statistic', 'interpolation', 'isdata']

    missing_params = [param for param in required_params if param not in config]

    if missing_params:
        raise ValueError(f"Missing required config parameters: {', '.join(missing_params)}")
    if config['DL_min'] >= config['DL_max']:
        raise ValueError('DL_min must be less than DL_max')
    if config['DT_min'] >= config['DT_max']:
        raise ValueError('DT_min must be less than DT_max')
    if config['angle_min'] >= config['angle_max']:
        raise ValueError('angle_min must be less than angle_max')
    if not isinstance(config['angle_step'], int):
        raise ValueError('angle_step must be an integer')

def save_outputs(delta_test_statistic_values, all_shifts_actual, all_shifts_result, config):

    current_time = datetime.datetime.now().strftime('%m%d%Y%H%M%S')
    #test_statistic_file_name = '{}/plots/diffusion_{}_{}.png'.format(LARDIFF_DIR, config['test_statistic'], current_time)
    test_statistic = config['test_statistic']
    isdata = config['isdata']
    data_or_mc = 'data' if isdata==True else 'mc'

    results_dict = {}
    test_statistic_file_name = '{}/plots/diffusion_{}_{}_{}.png'.format(LARDIFF_DIR, test_statistic, data_or_mc, current_time)
    make_test_statistic_plot(
        delta_test_statistic_values, config,
        filename=test_statistic_file_name,
        results_dict=results_dict
    )
    print('RUN results_dict', results_dict)
    print('Test statistic grid scan plot saved to', test_statistic_file_name)

    output_data_filename = '{}/output_data/diffusion_results_{}_{}.pkl'.format(LARDIFF_DIR, data_or_mc, current_time)
    with open(output_data_filename, 'wb') as fout:
        pickle.dump((results_dict, all_shifts_actual, all_shifts_result, delta_test_statistic_values, config), fout)

    print('Output data and config saved to', output_data_filename)

def measure_diffusion(input_filename, config):

    input_file = uproot.open(input_filename)

    isdata = config['isdata']

    angle_min = config['angle_min']
    angle_max = config['angle_max']
    angle_step = config['angle_step']
    angle_bins = [[x, x + angle_step] for x in range(angle_min, angle_max, angle_step)]
    num_angle_bins = len(angle_bins)

    input_signal        = np.zeros((num_angle_bins, N_ticks_fine, N_wires_fine))
    anode_hist          = np.zeros((num_angle_bins, N_ticks, N_wires))
    anode_uncert_hist   = np.zeros((num_angle_bins, N_ticks, N_wires))
    cathode_hist        = np.zeros((num_angle_bins, N_ticks, N_wires))
    cathode_uncert_hist = np.zeros((num_angle_bins, N_ticks, N_wires))

    # Generate initial histograms
    for bin_idx, angles in enumerate(angle_bins):
        print('Generating hists for bin_idx, angles', bin_idx, angles)
        start_angle = angles[0]
        end_angle   = angles[1]

        # Each angle bin is finely split into sub-bins with step size of angle_sim_step
        num_angles = int((end_angle - start_angle) / angle_sim_step) + 1
        signal_angles = np.linspace(start_angle + angle_sim_step/2, end_angle - angle_sim_step/2, num_angles)

        # Calculate the signal for this set of angles and store it in the input_sig array
        input_signal[bin_idx] = create_signal(signal_angles, isdata)
        anode_hist[bin_idx]        = np.swapaxes(input_file['AnodeTrackHist2D_%dto%d'       % (start_angle, end_angle)].values(), 0, 1)
        anode_uncert_hist[bin_idx] = np.swapaxes(input_file['AnodeTrackUncertHist2D_%dto%d' % (start_angle, end_angle)].values(), 0, 1)
        cathode_hist[bin_idx]        = np.swapaxes(input_file['CathodeTrackHist2D_%dto%d'       % (start_angle, end_angle)].values(), 0, 1)
        cathode_uncert_hist[bin_idx] = np.swapaxes(input_file['CathodeTrackUncertHist2D_%dto%d' % (start_angle, end_angle)].values(), 0, 1)

    DL_min  = config['DL_min']
    DL_max  = config['DL_max']
    DL_step = config['DL_step']

    DT_min  = config['DT_min']
    DT_max  = config['DT_max']
    DT_step = config['DT_step']

    test_statistic = config['test_statistic']
    interpolation = config['interpolation']

    #test_statistic_values, min_test_statistic, min_numvals, all_shifts_result, all_shifts_actual = diffusion_grid_scan(
    test_statistic_values, min_numvals, all_shifts_result, all_shifts_actual = diffusion_grid_scan(
        DL_min, DL_max, DL_step, DT_min, DT_max, DT_step, num_angle_bins,
        input_signal, anode_hist, anode_uncert_hist, cathode_hist, cathode_uncert_hist,
        test_statistic=test_statistic,
        interpolation=interpolation,
        isdata=isdata
    )
    min_test_statistic = np.amin(test_statistic_values)
    debug_filename = '{}/output_data/debug_full.pkl'.format(LARDIFF_DIR)
    with open(debug_filename, 'wb') as fout:
        pickle.dump((test_statistic_values), fout)

    zoom_factor = 100
    test_statistic_values = ndimage.zoom(test_statistic_values, zoom_factor)
    if test_statistic == "chi2":
        test_statistic_values = test_statistic_values - np.amin(test_statistic_values)

    point_y, point_x = np.unravel_index(np.argmin(test_statistic_values), test_statistic_values.shape)
    DL_result = (DL_step * point_y / zoom_factor) + DL_min - DL_step / 2.0
    DT_result = (DT_step * point_x / zoom_factor) + DT_min - DT_step / 2.0

    ndof = 2 # Two parameter measurement
    print('Minimum %s:  %.5f' % (test_statistic, min_test_statistic))
    if test_statistic == "chi2":
        print('Minimum %s (Reduced):  %.2f' % (test_statistic, (min_test_statistic / (min_numvals - ndof))))

    save_outputs(test_statistic_values, all_shifts_actual, all_shifts_result, config)

def main():

    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    print('*******Running with config********:\n', yaml.dump(config))
    print('**********************************')

    start_time = time.time()
    measure_diffusion(args.input_filename, config)
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time, "seconds")

if __name__ == "__main__":
    main()













