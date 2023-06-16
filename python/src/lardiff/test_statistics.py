import numpy as np
from scipy import fftpack
from scipy import stats
from numba import jit

from functools import lru_cache
from scipy.stats import rv_continuous, chi2, ttest_ind, chisquare
from scipy.special import erf, erfinv
from scipy.optimize import root_scalar

from .consts import *
from .waveform_functions import smear_signal, convolve, \
     deconvolve, coarsen_signal, fix_baseline, shift_signal_1D

#def calc_test_statistic(input_sig, 
def calc_test_statistic(anode_hist, anode_uncert_hist, 
                        cathode_hist, cathode_uncert_hist, 
                        pred_hist, pred_uncert_hist,
                        test_statistic='chi2',
                        interpolation='scipy'):

    temp_test_stat  = None 
    temp_num_values = None 
    shift_vector    = None

    #sig_A = smear_signal(input_sig, ticks_drift_A, DL_hyp, DT_hyp)
    #sig_C = smear_signal(input_sig, ticks_drift_C, DL_hyp, DT_hyp)
    #sig_A_coarse = coarsen_signal(sig_A)
    #sig_C_coarse = coarsen_signal(sig_C)

    #pred_hist = np.zeros((N_ticks, N_wires))
    #pred_uncert_hist = np.zeros((N_ticks, N_wires))
    #for col in range(N_wires):
    #    sig_A_slice = sig_A_coarse[:, col]
    #    sig_A_slice = sig_A_slice / sig_A_slice.sum()
    #    sig_C_slice = sig_C_coarse[:, col]
    #    sig_C_slice = sig_C_slice / np.real(sig_C_slice).sum()
    #    diffusion_kernel = deconvolve(sig_C_slice, sig_A_slice)

    #    anode_slice = anode_hist[:, col]
    #    anode_uncert_slice = anode_uncert_hist[:, col]

    #    pred_slice = convolve(anode_slice, diffusion_kernel)
    #    pred_uncert_slice = convolve(anode_uncert_slice, diffusion_kernel)

    #    pred_hist[:, col] = np.real(pred_slice)
    #    pred_uncert_hist[:, col] = np.real(pred_uncert_slice)

    #pred_hist = fix_baseline(pred_hist, anode_hist)
    #pred_uncert_hist = fix_baseline(pred_uncert_hist, anode_uncert_hist)

    cathode_max = 0.0
    for col in range(N_wires_start, N_wires_end):
        for row in range(N_ticks_start, N_ticks_end):
            if col == (N_wires - 1) // 2: continue
            if cathode_hist[row, col] < cathode_max: continue
            cathode_max = cathode_hist[row, col]

    ### TODO Slicing would be faster, but how to ignore the central wire while slicing?
    ### Implement a range of indices instead of N_wires_start etc.?
    #cathode_max = np.amax(cathode_hist[N_ticks_start:N_ticks_end, N_wires_start:N_wires_end])

    if test_statistic == "chi2":
        #print('Calc chi2')
        temp_test_stat, temp_num_values, shift_vector = calc_chisq(
            pred_hist, pred_uncert_hist,
            cathode_hist, cathode_uncert_hist, cathode_max,
            interpolation
        )
    elif test_statistic == "invariant3":
        #print('Calc invar3')
        temp_test_stat, temp_num_values, shift_vector = calc_invariant3(
            pred_hist, pred_uncert_hist,
            cathode_hist, cathode_uncert_hist, cathode_max,
            interpolation
        )
    elif test_statistic == "invariant3_alt":
        #print('Calc invar3 alt')
        temp_test_stat, temp_num_values, shift_vector = calc_invariant3_alt(
            pred_hist, cathode_hist, cathode_max,
            interpolation
        )
    else:
        raise ValueError('Invalid test_statistic argument provided')

    return temp_test_stat, temp_num_values, shift_vector

############# Chi^2 test ################
# Calculate one chi-squared point given value of DL and DT and 2D distributions associated with specific track data angle bin
#@profile
#@jit(nopython=True)
def calc_chisq(pred_hist, pred_uncert_hist, cathode_hist, cathode_uncert_hist, cathode_max, interpolation='scipy'):
    chisq = 0.0
    numvals = 0.0
    # TODO Should shift_vec be of length N_wires or range(N_wires_start, N_wires_end)?
    shift_vec = np.zeros((N_wires))
    for col in range(N_wires_start, N_wires_end):

        # Skip central wire to avoid bias (I think?)
        if col == (N_wires-1)//2: continue

        min_chisq = np.inf
        min_numvals = 0.0
        chisq_count = 0
        for shift_val in np.arange(-1.0*shift_max, shift_max+shift_step, shift_step):
            #anode_norm = 0
            pred_norm = 0
            cathode_norm = 0
            pred_hist_1D_shifted = shift_signal_1D(pred_hist[:, col], shift_val, interpolation)
            pred_uncert_hist_1D_shifted = shift_signal_1D(pred_uncert_hist[:, col], shift_val, interpolation)
            for row in range(N_ticks_start, N_ticks_end):
                # Exclude values below threshold
                if cathode_hist[row, col] < threshold_rel * cathode_max: continue

                #anode_norm += anode_hist[row,col]
                pred_norm += pred_hist_1D_shifted[row]
                cathode_norm += cathode_hist[row, col]

            chisq_temp = 0.0
            numvals_temp = 0.0
            for row in range(N_ticks_start, N_ticks_end):
                if cathode_hist[row, col] < threshold_rel * cathode_max: continue

                chisq_temp += ((pred_hist_1D_shifted[row]/pred_norm - cathode_hist[row,col]/cathode_norm)**2) / \
                              ((pred_uncert_hist_1D_shifted[row]/pred_norm)**2 + (cathode_uncert_hist[row,col]/cathode_norm)**2)
                chisq_count += 1
                numvals_temp += 1.0
            if chisq_temp < min_chisq:
                min_chisq = chisq_temp
                min_numvals = numvals_temp
                shift_vec[col] = shift_val

        chisq += min_chisq
        numvals += min_numvals
                
    return chisq, numvals, shift_vec

############### Invariant3 test ####################
def calc_invariant3(pred_hist, pred_uncert_hist, cathode_hist, cathode_uncert_hist, cathode_max, interpolation='scipy'):
    invar3 = 0.0
    numvals = 0.0
    # TODO Should shift_vec be of length N_wires or range(N_wires_start, N_wires_end)?
    shift_vec = np.zeros((N_wires))
    z_scores = np.array([])
    for col in range(N_wires_start, N_wires_end):

        # Skip central wire to avoid bias (I think?)
        if col == (N_wires-1)//2: continue

        min_invar3 = np.inf
        min_numvals = 0.0
        invar3_count = 0
        #for shift_val in np.arange(-1.0*shift_max, shift_max+shift_step, shift_step):
        for shift_val in np.arange(0, 1):
            pred_norm = 0
            cathode_norm = 0
            pred_hist_1D_shifted = shift_signal_1D(pred_hist[:, col], shift_val, interpolation)
            pred_uncert_hist_1D_shifted = shift_signal_1D(pred_uncert_hist[:, col], shift_val, interpolation)
            above_threshold_count = 0
            for row in range(N_ticks_start, N_ticks_end):
                # Exclude values below threshold
                if cathode_hist[row, col] < threshold_rel * cathode_max: continue

                pred_norm += pred_hist_1D_shifted[row]
                cathode_norm += cathode_hist[row, col]

                above_threshold_count += 1

            invar3_temp = 0.0
            numvals_temp = 0.0
            z_scores = np.zeros(above_threshold_count)
            above_threshold_count = 0
            for row in range(N_ticks_start, N_ticks_end):
                if cathode_hist[row, col] < threshold_rel * cathode_max: continue

                sigma_1 = pred_uncert_hist_1D_shifted[row]/pred_norm
                sigma_2 = cathode_uncert_hist[row, col]/cathode_norm
                #sigma_1 = np.std(pred_uncert_hist_1D_shifted)
                #sigma_2 = np.std(cathode_uncert_hist[:, col])
                #sigma_1 = pred_uncert_hist_1D_shifted[row]
                #sigma_2 = cathode_uncert_hist[row, col]
                sigma = np.sqrt(sigma_1*sigma_1 + sigma_2*sigma_2)
                print('sigma', sigma)

                z_score = (pred_hist_1D_shifted[row]/pred_norm - cathode_hist[row, col]/cathode_norm) / sigma
                #z_score = (pred_hist_1D_shifted[row] - cathode_hist[row, col]) / sigma
                print('z_score', z_score)
                z_scores[above_threshold_count] = z_score
                above_threshold_count += 1


    print('z_scores shape:', z_scores.shape)
    dist = lambda x: invariant3(x, alpha=2/3, fast=False)
    invar3 = dist(z_scores)
    print('invar3:', invar3)
    return invar3, numvals, shift_vec

def calc_invariant3_alt(pred_hist, cathode_hist, cathode_max, interpolation='scipy'):
    invar3 = 0.0
    numvals = 0.0
    # TODO Should shift_vec be of length N_wires or range(N_wires_start, N_wires_end)?
    shift_vec = np.zeros((N_wires))
    z_scores = np.array([])

    # Relative wire indices, where 5 is the central wire
    # Exclude outer wires due to FFT artifacts 
    wire_indices = [2, 3, 4, 6, 7, 8]
    num_wires = len(wire_indices)
    num_ticks = len(pred_hist)
    percent_of_max = 0.2
    for iw, wire in enumerate(wire_indices):
        pred = pred_hist[:, wire]
        meas = cathode_hist[:, wire]

        wire_max = np.max(meas)
        threshold = percent_of_max * np.max(meas)

        meas_roi = meas[meas > threshold]
        meas_roi_indices = np.where(meas > threshold)
        pred_roi = pred[meas_roi_indices]

        sigma_1 = np.std(pred_roi)
        sigma_2 = np.std(meas_roi)
        sigma = np.sqrt(sigma_1*sigma_1 + sigma_2*sigma_2)

        wire_z_scores = (pred_roi - meas_roi) / sigma
        z_scores = np.concatenate((z_scores, wire_z_scores))
        
    print('z_scores:', z_scores)
    print('z scores shape:', z_scores.shape)
    dist = lambda x: invariant3(x, alpha=2/3, fast=False)
    distances = dist(z_scores)
    print('invariant3 shape', distances.shape)
    print('invariant3', distances)
    return invar3, numvals, shift_vec

############### Invariant3 Functions from the paper #################
# https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.113008 #
class Bee(rv_continuous):
    def _cdf(self, x, df):
        return erf(x/np.sqrt(2))**df
    def _pdf(self, x, df):
        ret = df*(erf(x/np.sqrt(2)))**(df-1)
        return ret * np.sqrt(2/np.pi)*np.exp(-x**2/2)
    def ppf(self, x, df):
        return erfinv((x)**(1/df)) * np.sqrt(2)
    
class BeeSquared(rv_continuous):
    def _cdf(self, x, df):
        b = np.sqrt(x)
        ret = bee.cdf(b, df)
        return ret
    def _pdf(self, x, df):
        ret = df*(erf(np.sqrt(x/2)))**(df-1)
        return ret / np.sqrt(2*np.pi*x) * np.exp(-x/2)
    def ppf(self, x, df):
        b = bee.ppf(x, df)
        return b**2

@np.vectorize
@lru_cache(10000)
def _yfrommax(b, df = 2, alpha = 0.5):
    """(1—diagonal coordinate) from (1—max) of accepted region"""
    # A=(1-b)**df—((y-b)**df)/((1-alpha+alpha*y)**(df-1))=1—y
    beta = 1 - alpha
    q=(1.0-b) ** df-1
    dfm=df-1
    def f(y):
        return q-((y-b) ** df) / ((beta + alpha * y) ** (dfm)) + y
    if b <=0:
        return 0.0
    if b >= 1:
        return 1.0
    else:
        return root_scalar(f, x0=b, x1 = b * 1.001).root

def yfrommax(b, df = 2, alpha = 0.5):
    """Buffer and interpolate values to speed things up."""
    step = 0.0001
    b_= np.floor(b / step, dtype=float) * step
    b__= b_ + step
    delta = (b-b_) / step
    x_  = _yfrommax(b_,  df = df, alpha = alpha)
    x__ = _yfrommax(b__, df = df, alpha = alpha)
    return x_+(x__ - x_) * delta

def invariant3(x, alpha = 0.5, fast = False):
    """Return test statistic given vector of normalized values."""
    if fast:
        sf = 1 - chi2.cdf(x ** 2, df = 1) # Faster, but less accurate
    else:
        sf = chi2.sf(x ** 2, df = 1)
    # Get possible diagonal coordinate from maximum CDF value (= minimum SF)
    a = np.min(sf, axis=-1)
    b = np.max(sf, axis=-1)
    yfm = yfrommax(a, df=x.shape[-1], alpha = alpha)
    # Get possible diagonal coordinate from center surface
    yfc = (alpha * (a - b) + b) / (1.0 + alpha * (a - b))
    y = np.minimum(yfc, yfm)
    y = np.maximum(y, 0) # Cap in case of rounding or root finding errors
    return chi2.isf(y, df = 1)



