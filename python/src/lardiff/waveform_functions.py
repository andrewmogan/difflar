import numpy as np
import scipy.interpolate as interp
from scipy import fftpack
from scipy import stats
from numba import jit
from numba import float64
#from lardiff.consts import driftVel, N_ticks, N_ticks_fine, N_wires, N_wires_fine, \
#                           wirePitch, timeTickSF, set_drift_velocity, ticks_drift_A, \
#                           ticks_drift_C
from lardiff.consts import *

# Convolve two 1D or 2D distributions (order does not matter)
def convolve(input1, input2):
    input1_fft = fftpack.fftshift(fftpack.fftn(input1))
    input2_fft = fftpack.fftshift(fftpack.fftn(input2))
    result = fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(input1_fft*input2_fft)))
    for dim in range(0,result.ndim):
        result = np.roll(result,1,dim)
    return result

# Deconvolve one 1D or 2D distribution by another (input1 deconvolved by or 'divided by' input2)
def deconvolve(input1, input2):
    input1_fft = fftpack.fftshift(fftpack.fftn(input1))
    input2_fft = fftpack.fftshift(fftpack.fftn(input2))
    result = fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(input1_fft/input2_fft)))
    return result

# Create hypothesis track-like signal distribution used to calculate diffusion kernel
@jit(nopython=True)
def create_signal(angles, isdata):
    """
    Computes the track-like signal distribution for a given set of incident angles.

    Parameters:
    -----------
    angles : numpy.ndarray
        Array of incident angles in degrees.

    Returns:
    --------
    result : numpy.ndarray
        Array of shape (N_ticks_fine, N_wires_fine) containing the track-like signal distribution.
    """

    # These can't be set using consts.set_drift_params while using numba, so
    # just set them here
    driftVel = driftVel_data if isdata else driftVel_MC
    ticks_drift_A = offset_distance / driftVel / timeTickSF
    ticks_drift_C = (AC_distance - offset_distance) / driftVel / timeTickSF

    result = np.zeros((N_ticks_fine, N_wires_fine))
    for angle in angles:
        for row in range(N_ticks_fine):
            track_col = (
                (N_wires_fine - 1.0)/2.0 + (1.0 * N_wires_fine/N_wires) * 
                (driftVel/wirePitch)*(1.0 * N_ticks/N_ticks_fine) * timeTickSF * 
                (row - ((N_ticks_fine - 1.0)/2.0))/np.tan(angle * np.pi / 180.0)
            )
            for col in range(0, N_wires_fine):
                if col == 0 and col >= track_col and col < track_col+1:
                    result[row,0] += 1.0 - (col-track_col)
                elif col == N_wires_fine - 1 and col >= track_col - 1 and col < track_col:
                    result[row,N_wires_fine-1] += 1.0-(track_col-col)
                elif col < N_wires_fine - 1 and col >= track_col - 1 and col < track_col:
                    result[row,col] += 1.0-(track_col-col)
                    result[row,col + 1] += track_col-col
    result = result/result.sum()
    return result

# Smear input 2D distribution to simulate action of diffusion (both longitudinal and transverse)
def smear_signal(input, ticks_drift, DL_hyp, DT_hyp, isdata):
    driftVel, ticks_drift_A, ticks_drift_C = set_drift_params(isdata)
    sigmaT = np.sqrt(2.0 * DL_hyp * timeTickSF * ticks_drift) / 1000.0 / driftVel;
    sigmaW = np.sqrt(2.0 * DT_hyp * timeTickSF * ticks_drift) / 1000.0 / wirePitch;
    X_fine, Y_fine = np.ogrid[0:N_ticks_fine, 0:N_wires_fine]
    gauss = stats.norm.pdf(X_fine, (N_ticks_fine-1.0)/2.0, (sigmaT/timeTickSF)*(N_ticks_fine/N_ticks)) * stats.norm.pdf(Y_fine, (N_wires_fine-1.0)/2.0, sigmaW*(N_wires_fine/N_wires))
    gauss = gauss/gauss.sum()
    result = convolve(input, gauss)
    return result

# Coarsen input 2D distribution to reflect actual granularity of TPC in time and wire number
def coarsen_signal(input):
    result = input.reshape([N_ticks, N_ticks_fine//N_ticks, N_wires, N_wires_fine//N_wires]).mean(3).mean(1)
    result = (input.sum()/result.sum())*result
    return result

# Shift 1D distribution in time
def shift_signal_1D(input, shift_val, interpolation):
    if interpolation == 'scipy':
        input_interp = interp.interp1d(np.arange(input.size), input, fill_value='extrapolate', kind='cubic')
        result = input_interp(np.arange(input.size)-shift_val)
        return result
    elif interpolation == 'numpy':
        result = shift_signal_1D_fast(input, shift_val)
        return result
    else:
        raise ValueError('Invalid interpolate argument {} provided to shift_signal_1D'.format(interpolation))

@jit(nopython=True)
def shift_signal_1D_fast(input, shift_val):
    size = input.size
    input_interp = np.interp(np.arange(size), np.arange(size), input)
    result = np.interp(np.arange(size) - shift_val, np.arange(size), input_interp)
    return result

# Normalize 1D distribution and associated uncertainty (use is experimental)
def normalize_signal_wires(input, input_uncert):
    result = np.zeros((N_ticks, N_wires))
    result_uncert = np.zeros((N_ticks, N_wires))
    for i in range(0, N_wires):
        this_wire_sum = input.sum(axis=0)[i]
        result[:, i] = input[:, i] / this_wire_sum
        result_uncert[:, i] = input_uncert[:, i] / this_wire_sum
    return result, result_uncert

# Fix baseline of individual wires in input 2D distribution to match that of reference 2D distribution
def fix_baseline(input, ref):
    result = np.zeros((N_ticks, N_wires))
    for col in range(0, N_wires):
        sum_input = 0.0
        sum_ref = 0.0
        num = 0.0
        for row in range(0, N_ticks):
            if (col < (N_wires-1)//2 and row > 3*(N_ticks-1)//4) or (col > (N_wires-1)//2 and row < (N_ticks-1)//4) or (col == (N_wires-1)//2 and (row > 3*(N_ticks-1)//4 or row < (N_ticks-1)//4)):
                sum_input += input[row, col]
                sum_ref += ref[row, col]
                num += 1.0
        avg = (sum_ref - sum_input) / num
        for row in range(0, N_ticks):
            result[row, col] = input[row, col] + avg
    return result

# For each angle bin, construct the cathode prediction from the input diffusion parameters
def get_cathode_prediction(input_sig, anode_hist, anode_uncert_hist, DL_hyp, DT_hyp, isdata):
    driftVel, ticks_drift_A, ticks_drift_C = set_drift_params(isdata)
    sig_A = smear_signal(input_sig, ticks_drift_A, DL_hyp, DT_hyp, isdata)
    sig_C = smear_signal(input_sig, ticks_drift_C, DL_hyp, DT_hyp, isdata)
    sig_A_coarse = coarsen_signal(sig_A)
    sig_C_coarse = coarsen_signal(sig_C)

    pred_hist = np.zeros((N_ticks, N_wires))
    pred_uncert_hist = np.zeros((N_ticks, N_wires))
    for col in range(N_wires):
        sig_A_slice = sig_A_coarse[:, col]
        sig_A_slice = sig_A_slice / sig_A_slice.sum()
        sig_C_slice = sig_C_coarse[:, col]
        sig_C_slice = sig_C_slice / np.real(sig_C_slice).sum()
        diffusion_kernel = deconvolve(sig_C_slice, sig_A_slice)

        anode_slice = anode_hist[:, col]
        anode_uncert_slice = anode_uncert_hist[:, col]

        pred_slice = convolve(anode_slice, diffusion_kernel)
        pred_uncert_slice = convolve(anode_uncert_slice, diffusion_kernel)

        pred_hist[:, col] = np.real(pred_slice)
        pred_uncert_hist[:, col] = np.real(pred_uncert_slice)

    pred_hist = fix_baseline(pred_hist, anode_hist)
    pred_uncert_hist = fix_baseline(pred_uncert_hist, anode_uncert_hist)

    return pred_hist, pred_uncert_hist




















