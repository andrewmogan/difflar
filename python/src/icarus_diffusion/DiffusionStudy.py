import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fftpack
from scipy import stats
from scipy import ndimage
import scipy.interpolate as interp
from numba import jit
import uproot

###########################
# DEFINE GLOBAL CONSTANTS #
###########################

#print_details = True
print_details = False

DL_test = 4.0
DT_test = 9.75

#threshold_rel = -1000.0
threshold_rel = 0.1
#threshold_rel = 0.2
#threshold_rel = 0.05

#shift_max = 0.0
#shift_max = 1.0
shift_max = 2.0
shift_step = 0.1

#angle_bins = [[x, x+2] for x in range(26, 60, 2)]
#angle_bins = [[x, x+2] for x in range(26, 72, 2)]
#angle_bins = [[x, x+2] for x in range(26, 76, 2)]
angle_bins = [[x, x+2] for x in range(26, 80, 2)]
#angle_bins = [[x, x+2] for x in range(50, 80, 2)]
#angle_bins = [[30, 32], [54, 56], [78, 80]]
#angle_bins = [[30, 32]

angle_sim_step = 0.1
#angle_sim_step = 2.0

timeTickSF = 0.4
driftVel = 0.1571 # DATA
#driftVel = 0.157565 # MC (at 493.8 V/cm)
wirePitch = 0.3
DL_actual = 4.0
DT_actual = 8.8

N_wires = 11
N_wires_fit = 7
#N_wires_fine = 25*N_wires
N_wires_fine = 25*N_wires
N_ticks = 401
N_ticks_fit = 321
#N_ticks_fine = 5*N_ticks
N_ticks_fine = 5*N_ticks

offset_distance = 14.5
AC_distance = 148.275
ticks_drift_A = offset_distance/driftVel/timeTickSF
ticks_drift_C = (AC_distance - offset_distance)/driftVel/timeTickSF

###########################
# DEFINE HELPER FUNCTIONS #
###########################

# Prevent warning when creating too many figures
plt.rcParams.update({'figure.max_open_warning': 0})

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
def create_signal(angles):
    result = np.zeros((N_ticks_fine, N_wires_fine))
    for angle in angles:
        for row in range(0, N_ticks_fine):
            track_col = (N_wires_fine-1.0)/2.0 + (1.0*N_wires_fine/N_wires)*(driftVel/wirePitch)*(1.0*N_ticks/N_ticks_fine)*timeTickSF*(row - ((N_ticks_fine-1.0)/2.0))/np.tan(angle*np.pi/180.0)
            for col in range(0, N_wires_fine):
                if col == 0 and col >= track_col and col < track_col+1:
                    result[row,0] += 1.0-(col-track_col)
                elif col == N_wires_fine-1 and col >= track_col-1 and col < track_col:
                    result[row,N_wires_fine-1] += 1.0-(track_col-col)
                elif col < N_wires_fine-1 and col >= track_col-1 and col < track_col:
                    result[row,col] += 1.0-(track_col-col)
                    result[row,col+1] += track_col-col
    result = result/result.sum()
    return result

# Smear input 2D distribution to simulate action of diffusion (both longitudinal and transverse)
def smear_signal(input, ticks_drift, DL_hyp, DT_hyp):
    sigmaT = np.sqrt(2.0*DL_hyp*timeTickSF*ticks_drift)/1000.0/driftVel;
    sigmaW = np.sqrt(2.0*DT_hyp*timeTickSF*ticks_drift)/1000.0/wirePitch;
    X_fine, Y_fine = np.ogrid[0:N_ticks_fine, 0:N_wires_fine]
    gauss = stats.norm.pdf(X_fine, (N_ticks_fine-1.0)/2.0, (sigmaT/timeTickSF)*(N_ticks_fine/N_ticks))*stats.norm.pdf(Y_fine, (N_wires_fine-1.0)/2.0, sigmaW*(N_wires_fine/N_wires))
    gauss = gauss/gauss.sum()
    result = convolve(input, gauss)
    return result

# Coarsen input 2D distribution to reflect actual granularity of TPC in time and wire number
def coarsen_signal(input):
    result = input.reshape([N_ticks, N_ticks_fine//N_ticks, N_wires, N_wires_fine//N_wires]).mean(3).mean(1)
    result = (input.sum()/result.sum())*result
    return result

# Shift 1D distribution in time
def shift_signal_1D(input, shift_val):
    input_interp = interp.interp1d(np.arange(input.size), input, fill_value='extrapolate', kind='cubic')
    result = input_interp(np.arange(input.size)-shift_val)
    return result

# Normalize 1D distribution and associated uncertainty (use is experimental)
def normalize_signal_wires(input, input_uncert):
    result = np.zeros((N_ticks, N_wires))
    result_uncert = np.zeros((N_ticks, N_wires))
    for i in range(0, N_wires):
        this_wire_sum = input.sum(axis=0)[i]
        result[:,i] = input[:,i]/this_wire_sum
        result_uncert[:,i] = input_uncert[:,i]/this_wire_sum
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
                sum_input += input[row,col]
                sum_ref += ref[row,col]
                num += 1.0
        avg = (sum_ref-sum_input)/num
        for row in range(0, N_ticks):
            result[row,col] = input[row,col] + avg
    return result

# Calculate one chi-squared point given value of DL and DT and 2D distributions associated with specific track data angle bin
def calc_chisq(input_sig, anode_hist, anode_uncert_hist, cathode_hist, cathode_uncert_hist, DL_hyp, DT_hyp):
    sig_A = smear_signal(input_sig, ticks_drift_A, DL_hyp, DT_hyp)
    sig_C = smear_signal(input_sig, ticks_drift_C, DL_hyp, DT_hyp)
    sig_A_coarse = coarsen_signal(sig_A)
    sig_C_coarse = coarsen_signal(sig_C)

    pred_hist = np.zeros((N_ticks, N_wires))
    pred_uncert_hist = np.zeros((N_ticks, N_wires))
    for col in range(0, N_wires):
        sig_A_slice = sig_A_coarse[:,col]
        sig_A_slice = sig_A_slice/sig_A_slice.sum()
        sig_C_slice = sig_C_coarse[:,col]
        sig_C_slice = sig_C_slice/np.real(sig_C_slice).sum()
        diffusion_kernel = deconvolve(sig_C_slice, sig_A_slice)
        anode_slice = anode_hist[:,col]
        anode_uncert_slice = anode_uncert_hist[:,col]
        pred_slice = convolve(anode_slice, diffusion_kernel)
        pred_uncert_slice = convolve(anode_uncert_slice, diffusion_kernel)
        pred_hist[:,col] = np.real(pred_slice)
        pred_uncert_hist[:,col] = np.real(pred_uncert_slice)

    pred_hist = fix_baseline(pred_hist, anode_hist)
    pred_uncert_hist = fix_baseline(pred_uncert_hist, anode_uncert_hist)

    cathode_max = 0.0
    for col in range(((N_wires-1)//2)-((N_wires_fit-1)//2), ((N_wires-1)//2)+((N_wires_fit-1)//2)+1):
        for row in range(((N_ticks-1)//2)-((N_ticks_fit-1)//2), ((N_ticks-1)//2)+((N_ticks_fit-1)//2)+1):
            if col != (N_wires-1)//2:
                if cathode_hist[row,col] > cathode_max:
                    cathode_max = cathode_hist[row,col]

    chisq = 0.0
    numvals = 0.0
    shift_vec = np.zeros((N_wires))
    for col in range(((N_wires-1)//2)-((N_wires_fit-1)//2), ((N_wires-1)//2)+((N_wires_fit-1)//2)+1):
        if col != (N_wires-1)//2:
            min_chisq = 99999999.0
            min_numvals = 0.0
            for shift_val in np.arange(-1.0*shift_max, shift_max+shift_step, shift_step):
                anode_norm = 0
                pred_norm = 0
                cathode_norm = 0
                pred_hist_1D_shifted = shift_signal_1D(pred_hist[:,col], shift_val)
                pred_uncert_hist_1D_shifted = shift_signal_1D(pred_uncert_hist[:,col], shift_val)
                for row in range(((N_ticks-1)//2)-((N_ticks_fit-1)//2), ((N_ticks-1)//2)+((N_ticks_fit-1)//2)+1):
                    if cathode_hist[row,col] > threshold_rel*cathode_max:
                        anode_norm += anode_hist[row,col]
                        pred_norm += pred_hist_1D_shifted[row]
                        cathode_norm += cathode_hist[row,col]
                chisq_temp = 0.0
                numvals_temp = 0.0
                for row in range(((N_ticks-1)//2)-((N_ticks_fit-1)//2), ((N_ticks-1)//2)+((N_ticks_fit-1)//2)+1):
                    if cathode_hist[row,col] > threshold_rel*cathode_max:
                        chisq_temp += ((pred_hist_1D_shifted[row]/pred_norm - cathode_hist[row,col]/cathode_norm)**2)/((pred_uncert_hist_1D_shifted[row]/pred_norm)**2 + (cathode_uncert_hist[row,col]/cathode_norm)**2)
                        numvals_temp += 1.0
                if chisq_temp < min_chisq:
                    min_chisq = chisq_temp
                    min_numvals = numvals_temp
                    shift_vec[col] = shift_val
            chisq += min_chisq
            numvals += min_numvals
                
    return chisq, numvals, shift_vec

# Create plots illustrating signal distributions in 1D and 2D
def make_signal_plots(input_sig, anode_hist, anode_uncert_hist, cathode_hist, cathode_uncert_hist, DL_hyp, DT_hyp, shift_vec, filename2D, filename1D):
    sig_A = smear_signal(input_sig, ticks_drift_A, DL_hyp, DT_hyp)
    sig_C = smear_signal(input_sig, ticks_drift_C, DL_hyp, DT_hyp)
    sig_A_coarse = coarsen_signal(sig_A)
    sig_C_coarse = coarsen_signal(sig_C)

    pred_hist = np.zeros((N_ticks, N_wires))
    pred_uncert_hist = np.zeros((N_ticks, N_wires))
    diffusion_kernel_2D = np.zeros((N_ticks, N_wires))
    for col in range(0, N_wires):
        sig_A_slice = sig_A_coarse[:,col]
        sig_A_slice = sig_A_slice/sig_A_slice.sum()
        sig_C_slice = sig_C_coarse[:,col]
        sig_C_slice = sig_C_slice/np.real(sig_C_slice).sum()
        diffusion_kernel = deconvolve(sig_C_slice, sig_A_slice)
        diffusion_kernel_2D[:,col] = np.real(diffusion_kernel)
        anode_slice = anode_hist[:,col]
        anode_uncert_slice = anode_uncert_hist[:,col]
        pred_slice = convolve(anode_slice, diffusion_kernel)
        pred_uncert_slice = convolve(anode_uncert_slice, diffusion_kernel)
        pred_hist[:,col] = np.real(pred_slice)
        pred_uncert_hist[:,col] = np.real(pred_uncert_slice)

    pred_hist = fix_baseline(pred_hist, anode_hist)
    pred_uncert_hist = fix_baseline(pred_uncert_hist, anode_uncert_hist)

    pred_hist_shifted = np.zeros((N_ticks, N_wires))
    pred_uncert_hist_shifted = np.zeros((N_ticks, N_wires))
    for i in range(0, N_wires):
        pred_hist_shifted[:,i] = shift_signal_1D(pred_hist[:,i], shift_vec[i])
        pred_uncert_hist_shifted[:,i] = shift_signal_1D(pred_uncert_hist[:,i], shift_vec[i])
    
    anode_norm = np.zeros((N_wires))
    pred_norm = np.zeros((N_wires))
    cathode_norm = np.zeros((N_wires))
    cathode_max = 0.0
    for row in range(((N_ticks-1)//2)-((N_ticks_fit-1)//2), ((N_ticks-1)//2)+((N_ticks_fit-1)//2)+1):
        for col in range(((N_wires-1)//2)-((N_wires_fit-1)//2), ((N_wires-1)//2)+((N_wires_fit-1)//2)+1):
            if col != (N_wires-1)//2:
                if cathode_hist[row,col] > cathode_max:
                    cathode_max = cathode_hist[row,col]
                
    for row in range(((N_ticks-1)//2)-((N_ticks_fit-1)//2), ((N_ticks-1)//2)+((N_ticks_fit-1)//2)+1):
        for col in range(((N_wires-1)//2)-((N_wires_fit-1)//2), ((N_wires-1)//2)+((N_wires_fit-1)//2)+1):
            if col != (N_wires-1)//2:
                if cathode_hist[row,col] > threshold_rel*cathode_max:
                    anode_norm[col] += anode_hist[row,col]
                    pred_norm[col] += pred_hist_shifted[row,col]
                    cathode_norm[col] += cathode_hist[row,col]

    chisq_hist = np.zeros((N_ticks, N_wires))
    for row in range(((N_ticks-1)//2)-((N_ticks_fit-1)//2), ((N_ticks-1)//2)+((N_ticks_fit-1)//2)+1):
        for col in range(((N_wires-1)//2)-((N_wires_fit-1)//2), ((N_wires-1)//2)+((N_wires_fit-1)//2)+1):
            if col != (N_wires-1)//2 and cathode_hist[row,col] > threshold_rel*cathode_max:
                chisq_hist[row,col] += ((pred_hist_shifted[row,col]/pred_norm[col] - cathode_hist[row,col]/cathode_norm[col])**2)/((pred_uncert_hist_shifted[row,col]/pred_norm[col])**2 + (cathode_uncert_hist[row,col]/cathode_norm[col])**2)

    tickformat_time_fine = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((1.0*N_ticks/N_ticks_fine)*(x-(N_ticks_fine-1.0)/2.0)*timeTickSF))
    tickformat_wire_fine = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((1.0*N_wires/N_wires_fine)*(x-(N_wires_fine-1.0)/2.0)))
    tickformat_time = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((x-(N_ticks-1.0)/2.0)*timeTickSF))
    tickformat_wire = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x-(N_wires-1.0)/2.0))    
    labels_time_fine = []
    labels_time = []
    labels_wire_fine = []
    labels_wire = []
    for i in range(-4, 5):
        labels_time_fine.append((N_ticks_fine-1.0)/2.0 + (i/4.0)*((N_ticks_fine-1.0)/2.0 - ((N_ticks_fine/N_ticks)-1.0)/2.0))
        labels_time.append((N_ticks-1.0)/2.0 + (i/4.0)*((N_ticks-1.0)/2.0))
    for i in range(-1, 2):
        labels_wire_fine.append((N_wires_fine-1.0)/2.0 + i*((N_wires_fine-1.0)/2.0 - ((N_wires_fine/N_wires)-1.0)/2.0))
        labels_wire.append((N_wires-1.0)/2.0 + i*((N_wires-1.0)/2.0))
        
    f, axes = plt.subplots(1, 7, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 1]})
    #axes[0].imshow(anode_hist, aspect='auto', interpolation='none')
    #axes[1].imshow(anode_uncert_hist, aspect='auto', interpolation='none')
    #axes[2].imshow(pred_hist_shifted, aspect='auto', interpolation='none')
    #axes[3].imshow(pred_uncert_hist_shifted, aspect='auto', interpolation='none')
    #axes[4].imshow(cathode_hist, aspect='auto', interpolation='none')
    #axes[5].imshow(cathode_uncert_hist, aspect='auto', interpolation='none')
    #axes[6].imshow(chisq_hist, aspect='auto', interpolation='none')
    axes[0].imshow(np.real(sig_A), aspect='auto', interpolation='none')
    axes[1].imshow(np.real(sig_C), aspect='auto', interpolation='none')
    axes[2].imshow(np.real(sig_A_coarse), aspect='auto', interpolation='none')
    axes[3].imshow(np.real(sig_C_coarse), aspect='auto', interpolation='none')
    axes[4].imshow(anode_hist, aspect='auto', interpolation='none')
    axes[5].imshow(pred_hist_shifted, aspect='auto', interpolation='none')
    axes[6].imshow(cathode_hist, aspect='auto', interpolation='none')
    for i in range(0, 7):
        axes[i].invert_yaxis()
        axes[i].set(xlabel='Relative Wire', ylabel='Relative Time [$\mu s$]')
        axes[i].xaxis.get_label().set_fontsize(20)
        axes[i].yaxis.get_label().set_fontsize(20)
        axes[i].tick_params(axis='x', labelsize=16)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].xaxis.set_major_formatter(tickformat_wire)
        axes[i].yaxis.set_major_formatter(tickformat_time)
        axes[i].set_xticks(labels_wire)
        axes[i].set_yticks(labels_time)
    for i in range(0, 2):
        axes[i].xaxis.set_major_formatter(tickformat_wire_fine)
        axes[i].yaxis.set_major_formatter(tickformat_time_fine)
        axes[i].set_xticks(labels_wire_fine)
        axes[i].set_yticks(labels_time_fine)
    #axes[0].set_title('Anode Signal\nMeasurement', fontsize=26)
    #axes[1].set_title('Anode Signal\nMeasurement\nUncertainty', fontsize=26)
    #axes[2].set_title('Cathode Signal\nPrediction', fontsize=26)
    #axes[3].set_title('Cathode Signal\nPrediction\nUncertainty', fontsize=26)
    #axes[4].set_title('Cathode Signal\nMeasurement', fontsize=26)
    #axes[5].set_title('Cathode Signal\nMeasurement\nUncertainty', fontsize=26)
    #axes[6].set_title('Local $\chi^{2}$\nContribution', fontsize=26)
    axes[0].set_title('Anode Track\nSignal Hypothesis\n(Fine)', fontsize=26)
    axes[1].set_title('Cathode Track\nSignal Hypothesis\n(Fine)', fontsize=26)
    axes[2].set_title('Anode Track\nSignal Hypothesis\n(Coarse)', fontsize=26)
    axes[3].set_title('Cathode Track\nSignal Hypothesis\n(Coarse)', fontsize=26)
    axes[4].set_title('Anode Signal\nMeasurement', fontsize=26)
    axes[5].set_title('Cathode Signal\nPrediction', fontsize=26)
    axes[6].set_title('Cathode Signal\nMeasurement', fontsize=26)
    f.set_size_inches(28,12)
    f.tight_layout()
    plt.savefig(filename2D)

    f2, axes2 = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1]})
    for h in range(0, 3):
        for k in range(0, 2):
            wire_index = int(((N_wires-1)//2)+(2*k-1)*(h+1))
            axes2[h][k].errorbar(x=range(0, N_ticks), y=anode_hist[:,wire_index]/anode_norm[wire_index], yerr=anode_uncert_hist[:,wire_index]/anode_norm[wire_index], color='black', linewidth=2, label='Anode Measurement')
            axes2[h][k].errorbar(x=range(0, N_ticks), y=pred_hist_shifted[:,wire_index]/pred_norm[wire_index], yerr=pred_uncert_hist_shifted[:,wire_index]/pred_norm[wire_index], color='blue', linewidth=2, label=('Cathode Prediction ($D_{L}$ = %.2f cm$^{2}$/s, $D_{T}$ = %.2f cm$^{2}$/s)' % (DL_hyp, DT_hyp)))
            axes2[h][k].errorbar(x=range(0, N_ticks), y=cathode_hist[:,wire_index]/cathode_norm[wire_index], yerr=cathode_uncert_hist[:,wire_index]/cathode_norm[wire_index], color='red', linewidth=2, label='Cathode Measurement')
            axes2[h][k].set(xlabel='Relative Time [$\mu s$]', ylabel='Arb. Units')
            axes2[h][k].xaxis.get_label().set_fontsize(20)
            axes2[h][k].yaxis.get_label().set_fontsize(20)
            axes2[h][k].tick_params(axis='x', labelsize=16)
            axes2[h][k].tick_params(axis='y', labelsize=16)
            axes2[h][k].xaxis.set_major_formatter(tickformat_time)
            axes2[h][k].set_xticks(labels_time)
            if k == 0:
                axes2[h][k].set_xlim((-45/timeTickSF)+(N_ticks-1.0)/2.0, (10/timeTickSF)+(N_ticks-1.0)/2.0)
            if k == 1:
                axes2[h][k].set_xlim((-10/timeTickSF)+(N_ticks-1.0)/2.0, (45/timeTickSF)+(N_ticks-1.0)/2.0)
    axes2[0][0].set_title('Relative Wire: -1', fontsize=26)
    axes2[0][1].set_title('Relative Wire: +1', fontsize=26)
    axes2[1][0].set_title('Relative Wire: -2', fontsize=26)
    axes2[1][1].set_title('Relative Wire: +2', fontsize=26)
    axes2[2][0].set_title('Relative Wire: -3', fontsize=26)
    axes2[2][1].set_title('Relative Wire: +3', fontsize=26)
    axes2[0][0].legend(fontsize=18)
    f2.set_size_inches(32,16)
    f2.tight_layout()
    plt.savefig(filename1D)

# Make final chi-squared scan plot
def make_chisq_plot(delta_chisq_values, DL_min, DL_max, DL_step, DT_min, DT_max, DT_step, filename):
    point_y, point_x = np.unravel_index(np.argmin(delta_chisq_values), delta_chisq_values.shape)
    
    tickformat_DT = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((DT_step*x/100.0)+DT_min-DT_step/2.0))
    tickformat_DL = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((DL_step*x/100.0)+DL_min-DL_step/2.0))
    labels_DT = []
    for i in np.arange(DT_min, DT_max+0.25, 0.25):
        labels_DT.append((100.0/DT_step)*(i-DT_min+DT_step/2.0))
    labels_DL = []
    for i in np.arange(DL_min, DL_max+0.25, 0.25):
        labels_DL.append((100.0/DL_step)*(i-DL_min+DL_step/2.0))
        
    f, axes = plt.subplots(1, 1, gridspec_kw={'width_ratios': [1]})
    shw = axes.contourf(delta_chisq_values - np.amin(delta_chisq_values))
    shw2 = axes.contour(delta_chisq_values - np.amin(delta_chisq_values), levels = [2.30, 6.17], colors=('red',), linestyles=('dashed', 'solid'), linewidths=(3,))
    shw3 = axes.plot(point_x, point_y, marker='o', markersize=15, color='red')
    shw4 = axes.plot((100.0/DT_step)*(DT_actual-DT_min+DT_step/2.0), (100.0/DL_step)*(DL_actual-DL_min+DL_step/2.0), marker='*', markersize=20, color='fuchsia')
    axes.set_xlabel('$D_{T}$ [cm$^{2}$/s]', fontsize=36, labelpad=15)
    axes.set_ylabel('$D_{L}$ [cm$^{2}$/s]', fontsize=36, labelpad=15)
    axes.tick_params(axis='x', labelsize=28)
    axes.tick_params(axis='y', labelsize=28)
    axes.xaxis.set_major_formatter(tickformat_DT)
    axes.yaxis.set_major_formatter(tickformat_DL)
    axes.set_xticks(labels_DT)
    axes.set_yticks(labels_DL)
    axes.set_title('$\Delta\chi^{2}$ Scan Results', fontsize=48, pad=30)
    cbar = plt.colorbar(shw)
    cbar.set_label('$\Delta\chi^{2}$', fontsize=36, labelpad=15)
    cbar.ax.tick_params(labelsize=28)
    f.set_size_inches(14,10)
    f.tight_layout()
    print('DT Result:  %.2f (%.2f, %.2f)' % ((DT_step*point_x/100.0)+DT_min-DT_step/2.0, (DT_step*min(shw2.collections[0].get_paths()[0].vertices[:,0])/100.0)+DT_min-DT_step/2.0, (DT_step*max(shw2.collections[0].get_paths()[0].vertices[:,0])/100.0)+DT_min-DT_step/2.0))
    print('DL Result:  %.2f (%.2f, %.2f)' % ((DL_step*point_y/100.0)+DL_min-DL_step/2.0, (DL_step*min(shw2.collections[0].get_paths()[0].vertices[:,1])/100.0)+DL_min-DL_step/2.0, (DL_step*max(shw2.collections[0].get_paths()[0].vertices[:,1])/100.0)+DL_min-DL_step/2.0))
    plt.savefig(filename)

##################
# GET INPUT DATA #
##################

#file = uproot.open("results_diffusion/WFresults_Plane2_XX_1DrespMC.root")
#file = uproot.open("WFresults_Plane2_XX_2DrespMC.root")
file = uproot.open("WFresults_Plane2_XX_20kEvts.root")
#file = uproot.open("results_diffusion/WFresults_Plane2_XX_Data.root")
#file = uproot.open("results_diffusion/WFresults_Plane2_EX_Data.root")
#file = uproot.open("results_diffusion/WFresults_Plane2_WX_Data.root")
#file = uproot.open("results_diffusion/WFresults_Plane2_EE_Data.root")
#file = uproot.open("results_diffusion/WFresults_Plane2_EW_Data.root")
#file = uproot.open("results_diffusion/WFresults_Plane2_WE_Data.root")
#file = uproot.open("results_diffusion/WFresults_Plane2_WW_Data.root")

input_sig = np.zeros((len(angle_bins), N_ticks_fine, N_wires_fine))
anode_hist = np.zeros((len(angle_bins), N_ticks, N_wires))
anode_uncert_hist = np.zeros((len(angle_bins), N_ticks, N_wires))
cathode_hist = np.zeros((len(angle_bins), N_ticks, N_wires))
cathode_uncert_hist = np.zeros((len(angle_bins), N_ticks, N_wires))

for k in range(0, len(angle_bins)):
    input_sig[k] = create_signal(np.arange(angle_bins[k][0]+(angle_sim_step/2.0), angle_bins[k][1]-(angle_sim_step/2.0)+0.00001, angle_sim_step))
    anode_hist[k] = np.swapaxes(file['AnodeTrackHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1)
    anode_uncert_hist[k] = np.swapaxes(file['AnodeTrackUncertHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1)
    cathode_hist[k] = np.swapaxes(file['CathodeTrackHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1)
    cathode_uncert_hist[k] = np.swapaxes(file['CathodeTrackUncertHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1)
    #anode_hist[k], anode_uncert_hist[k] = normalize_signal_wires(np.swapaxes(file['AnodeTrackHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1), np.swapaxes(file['AnodeTrackUncertHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1)) # NORMALIZES INDIVIDUAL WIRES BEFORE SMEARING/COMPARING
    #cathode_hist[k], cathode_uncert_hist[k] = normalize_signal_wires(np.swapaxes(file['CathodeTrackHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1), np.swapaxes(file['CathodeTrackUncertHist2D_%dto%d' % (angle_bins[k][0], angle_bins[k][1])].values(),0,1)) # NORMALIZES INDIVIDUAL WIRES BEFORE COMPARING

###############
# DO ANALYSIS #
###############

DL_min = 3.5
DL_max = 5.5
DL_step = 0.25
#DL_min = 3.75
#DL_max = 4.75
#DL_step = 0.1

DT_min = 6.5
DT_max = 9.5
DT_step = 0.25
#DT_min = 7.0
#DT_max = 8.0
#DT_step = 0.1

min_chisq = 99999999.0
min_numvals = 0.0
all_shifts_result = np.zeros((len(angle_bins), N_wires))
all_shifts_actual = np.zeros((len(angle_bins), N_wires))
all_shifts_test = np.zeros((len(angle_bins), N_wires))
chisq_values = np.zeros((int((DL_max-DL_min)/DL_step+1), int((DT_max-DT_min)/DT_step+1)))
num_values = np.zeros((int((DL_max-DL_min)/DL_step+1), int((DT_max-DT_min)/DT_step+1)))
row = 0
for DL in np.arange(DL_min, DL_max+DL_step, DL_step):
    col = 0
    for DT in np.arange(DT_min, DT_max+DT_step, DT_step):
        chisq = 0.0
        numvals = 0.0
        all_shifts = np.zeros((len(angle_bins), N_wires))
        for k in range(0, len(angle_bins)):
            temp_chisq, temp_numvals, shift_vec = calc_chisq(input_sig[k], anode_hist[k], anode_uncert_hist[k], cathode_hist[k], cathode_uncert_hist[k], DL, DT)
            chisq += temp_chisq
            numvals += temp_numvals
            all_shifts[k,:] = shift_vec
            if print_details == True:
                print('    %d %.2f' % (k, temp_chisq))
                with np.printoptions(precision=1, sign=' ', floatmode='fixed', suppress=True):
                    print('   ', k, shift_vec)
        chisq_values[row,col] = chisq
        num_values[row,col] = numvals
        if chisq < min_chisq:
            min_chisq = chisq
            min_numvals = numvals
            all_shifts_result = all_shifts
        if abs(DL-DL_actual) < DL_step and abs(DT-DT_actual) < DT_step:
            all_shifts_actual = all_shifts
        if abs(DL-DL_test) < DL_step and abs(DT-DT_test) < DT_step:
            all_shifts_test = all_shifts
        print('%.2f %.2f %.2f' % (DL, DT, chisq))
        col += 1
    row += 1

delta_chisq_values = ndimage.zoom(chisq_values - np.amin(chisq_values), 100)
point_y, point_x = np.unravel_index(np.argmin(delta_chisq_values), delta_chisq_values.shape)
DL_result = (DL_step*point_y/100.0)+DL_min-DL_step/2.0
DT_result = (DT_step*point_x/100.0)+DT_min-DT_step/2.0

##############
# MAKE PLOTS #
##############

for k in range(0, len(angle_bins)):
    make_signal_plots(input_sig[k], anode_hist[k], anode_uncert_hist[k], cathode_hist[k], cathode_uncert_hist[k], DL_result, DT_result, all_shifts_result[k], ('diffusion_2Dsignal_result_%dto%d.png' % (angle_bins[k][0], angle_bins[k][1])), ('diffusion_1Dsignal_result_%dto%d.png' % (angle_bins[k][0], angle_bins[k][1])))
    make_signal_plots(input_sig[k], anode_hist[k], anode_uncert_hist[k], cathode_hist[k], cathode_uncert_hist[k], DL_actual, DT_actual, all_shifts_actual[k], ('diffusion_2Dsignal_actual_%dto%d.png' % (angle_bins[k][0], angle_bins[k][1])), ('diffusion_1Dsignal_actual_%dto%d.png' % (angle_bins[k][0], angle_bins[k][1])))
    #make_signal_plots(input_sig[k], anode_hist[k], anode_uncert_hist[k], cathode_hist[k], cathode_uncert_hist[k], DL_test, DT_test, all_shifts_test[k], ('diffusion_2Dsignal_test_%dto%d.png' % (angle_bins[k][0], angle_bins[k][1])), ('diffusion_1Dsignal_test_%dto%d.png' % (angle_bins[k][0], angle_bins[k][1])))

make_chisq_plot(delta_chisq_values, DL_min, DL_max, DL_step, DT_min, DT_max, DT_step, 'diffusion_chisq.png')
print('Minimum Chi-Squared:  %.2f' % min_chisq)
print('Minimum Chi-Squared (Reduced):  %.2f' % (min_chisq/(min_numvals-2.0)))
