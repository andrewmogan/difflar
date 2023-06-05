import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .consts import *
from .waveform_functions import smear_signal, convolve, \
     deconvolve, coarsen_signal, fix_baseline, shift_signal_1D

# Create plots illustrating signal distributions in 1D and 2D
def make_signal_plots(input_sig, 
                      anode_hist, anode_uncert_hist, 
                      cathode_hist, cathode_uncert_hist, 
                      DL_hyp, DT_hyp, 
                      shift_vec, 
                      filename2D, filename1D):

    hist_file_name = 'diffusion_hist_data.npz'

    sig_A = smear_signal(input_sig, ticks_drift_A, DL_hyp, DT_hyp)
    sig_C = smear_signal(input_sig, ticks_drift_C, DL_hyp, DT_hyp)
    sig_A_coarse = coarsen_signal(sig_A)
    sig_C_coarse = coarsen_signal(sig_C)

    pred_hist = np.zeros((N_ticks, N_wires))
    pred_uncert_hist = np.zeros((N_ticks, N_wires))
    diffusion_kernel_2D = np.zeros((N_ticks, N_wires))
    for col in range(N_wires):
        sig_A_slice = sig_A_coarse[:, col]
        sig_A_slice = sig_A_slice / sig_A_slice.sum()
        sig_C_slice = sig_C_coarse[:, col]
        sig_C_slice = sig_C_slice / np.real(sig_C_slice).sum()
        diffusion_kernel = deconvolve(sig_C_slice, sig_A_slice)
        diffusion_kernel_2D[:, col] = np.real(diffusion_kernel)
        anode_slice = anode_hist[:, col]
        anode_uncert_slice = anode_uncert_hist[:, col]
        pred_slice = convolve(anode_slice, diffusion_kernel)
        pred_uncert_slice = convolve(anode_uncert_slice, diffusion_kernel)
        pred_hist[:, col] = np.real(pred_slice)
        pred_uncert_hist[:, col] = np.real(pred_uncert_slice)

    pred_hist = fix_baseline(pred_hist, anode_hist)
    pred_uncert_hist = fix_baseline(pred_uncert_hist, anode_uncert_hist)

    pred_hist_shifted = np.zeros((N_ticks, N_wires))
    pred_uncert_hist_shifted = np.zeros((N_ticks, N_wires))
    for i in range(N_wires):
        pred_hist_shifted[:, i] = shift_signal_1D(pred_hist[:, i], shift_vec[i])
        pred_uncert_hist_shifted[:, i] = shift_signal_1D(pred_uncert_hist[:, i], shift_vec[i])
    
    anode_norm = np.zeros((N_wires))
    pred_norm = np.zeros((N_wires))
    cathode_norm = np.zeros((N_wires))
    cathode_max = 0.0
    for row in range(N_ticks_start, N_ticks_end):
        for col in range(N_wires_start, N_wires_end):
            if col == (N_wires-1)//2: continue
            if cathode_hist[row,col] < cathode_max: continue
            cathode_max = cathode_hist[row, col]
                
    for row in range(N_ticks_start, N_ticks_end):
        for col in range(N_wires_start, N_wires_end):
            if col == (N_wires-1)//2: continue
            if cathode_hist[row,col] < threshold_rel*cathode_max: continue
            anode_norm[col] += anode_hist[row, col]
            pred_norm[col] += pred_hist_shifted[row, col]
            cathode_norm[col] += cathode_hist[row, col]

    chisq_hist = np.zeros((N_ticks, N_wires))
    for row in range(N_ticks_start, N_ticks_end):
        for col in range(N_wires_start, N_wires_end):
            if col == (N_wires-1)//2: continue 
            if cathode_hist[row,col] < threshold_rel*cathode_max: continue
            # chi^2 calculation
            expected = pred_hist_shifted[row, col] / pred_norm[col]
            observed = cathode_hist[row, col] / cathode_norm[col]
            sigma_expected = pred_uncert_hist_shifted[row, col] / pred_norm[col]
            sigma_observed = cathode_uncert_hist[row, col] / cathode_norm[col]
            chisq_hist[row, col] = ((expected - observed)**2 / (sigma_expected**2 + sigma_observed**2))

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

    ######################## Relative wire plots ##############################
    # Make relative wire plots in a 3x2 grid with the negative relative wires
    # in the left column and the positive wires in the right column
    f2, axes2 = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1]})
    num_rows = 3
    num_cols = 2
    for h in range(num_rows):
        for k in range(num_cols):
            wire_index = int(((N_wires - 1) // 2) + (2 * k - 1) * (h + 1))
            axes2[h][k].errorbar(x=range(0, N_ticks), 
                                 y=anode_hist[:, wire_index] / anode_norm[wire_index], 
                                 yerr=anode_uncert_hist[:, wire_index] / anode_norm[wire_index], 
                                 color='black', linewidth=2, label='Anode Measurement')
            axes2[h][k].errorbar(x=range(0, N_ticks), 
                                 y=pred_hist_shifted[:, wire_index] / pred_norm[wire_index], 
                                 yerr=pred_uncert_hist_shifted[:, wire_index] / pred_norm[wire_index], 
                                 color='blue', linewidth=2, 
                                 label=('Cathode Prediction ($D_{L}$ = %.2f cm$^{2}$/s, $D_{T}$ = %.2f cm$^{2}$/s)' % (DL_hyp, DT_hyp)))
            axes2[h][k].errorbar(x=range(0, N_ticks), 
                                 y=cathode_hist[:, wire_index] / cathode_norm[wire_index], 
                                 yerr=cathode_uncert_hist[:, wire_index] / cathode_norm[wire_index], 
                                 color='red', linewidth=2, label='Cathode Measurement')
            axes2[h][k].set(xlabel='Relative Time [$\mu s$]', ylabel='Arb. Units')
            axes2[h][k].xaxis.get_label().set_fontsize(20)
            axes2[h][k].yaxis.get_label().set_fontsize(20)
            axes2[h][k].tick_params(axis='x', labelsize=16)
            axes2[h][k].tick_params(axis='y', labelsize=16)
            axes2[h][k].xaxis.set_major_formatter(tickformat_time)
            axes2[h][k].set_xticks(labels_time)
            # Show different tails for negative and positive columns
            if k == 0:
                axes2[h][k].set_xlim((-45 / timeTickSF) + (N_ticks - 1.0) / 2.0, 
                                     ( 10 / timeTickSF) + (N_ticks - 1.0) / 2.0)
            if k == 1:
                axes2[h][k].set_xlim((-10 / timeTickSF) + (N_ticks - 1.0) / 2.0, 
                                     ( 45 / timeTickSF) + (N_ticks - 1.0) / 2.0)

            # Save to file for offline use
            if os.path.exists(hist_file_name): continue
            hists_to_save = {
                #'anode_measurement'  : anode_hist[:, wire_index] / anode_norm[wire_index],
                #'cathode_prediction' : pred_hist_shifted[:,wire_index]/pred_norm[wire_index],
                #'cathode_measurement': cathode_hist[:, wire_index] / cathode_norm[wire_index]
                'anode_measurement'  : anode_hist,
                'cathode_prediction' : pred_hist_shifted,
                'cathode_measurement': cathode_hist 
            }
            np.savez(hist_file_name, **hists_to_save)
            
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

######################### Grid scan plots ###################################
def make_test_statistic_plot(delta_test_statistic_values, 
                             DL_min, DL_max, DL_step, 
                             DT_min, DT_max, DT_step, 
                             test_statistic='chi2',
                             filename='plots/diffusion_test_statistic.png'):

    if test_statistic == "chi2":
        make_chisq_plot(delta_test_statistic_values, 
                        DL_min, DL_max, DL_step, 
                        DT_min, DT_max, DT_step, 
                        filename)
    elif test_statistic == "invariant3":
        #print('Oops not implemented')
        make_chisq_plot(delta_test_statistic_values, 
                        DL_min, DL_max, DL_step, 
                        DT_min, DT_max, DT_step, 
                        filename)
    else:
        raise ValueError('Invalid test_statistic argument provided')

# Make final chi-squared scan plot
def make_chisq_plot(delta_chisq_values, DL_min, DL_max, DL_step, DT_min, DT_max, DT_step, filename):
    point_y, point_x = np.unravel_index(np.argmin(delta_chisq_values), delta_chisq_values.shape)
    
    tickformat_DT = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((DT_step*x/100.0)+DT_min-DT_step/2.0))
    tickformat_DL = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((DL_step*x/100.0)+DL_min-DL_step/2.0))
    labels_DT = []
    for i in np.arange(DT_min, DT_max + 0.25, 0.25):
        labels_DT.append((100.0 / DT_step) * (i - DT_min + DT_step / 2.0))
    labels_DL = []
    for i in np.arange(DL_min, DL_max + 0.25, 0.25):
        labels_DL.append((100.0 / DL_step) * (i - DL_min + DL_step / 2.0))
        
    f, axes = plt.subplots(1, 1, gridspec_kw={'width_ratios': [1]})
    shw = axes.contourf(delta_chisq_values - np.amin(delta_chisq_values))
    shw2 = axes.contour(delta_chisq_values - np.amin(delta_chisq_values), 
                        levels = [2.30, 6.17], 
                        colors=('red',), linestyles=('dashed', 'solid'), linewidths=(3,))
    shw3 = axes.plot(point_x, point_y, marker='o', markersize=15, color='red')
    shw4 = axes.plot((100.0 / DT_step) * (DT_actual - DT_min + DT_step / 2.0), 
                     (100.0 / DL_step) * (DL_actual - DL_min + DL_step / 2.0), 
                     marker='*', markersize=20, color='fuchsia')
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
    print('DT Result:  %.2f (%.2f, %.2f)' % 
         ((DT_step * point_x / 100.0) + DT_min - DT_step / 2.0, 
          (DT_step * min(shw2.collections[0].get_paths()[0].vertices[:, 0]) / 100.0) + DT_min - DT_step / 2.0, 
          (DT_step * max(shw2.collections[0].get_paths()[0].vertices[:, 0]) / 100.0) + DT_min - DT_step / 2.0))
    print('DL Result:  %.2f (%.2f, %.2f)' % 
         ((DL_step * point_y / 100.0) + DL_min - DL_step / 2.0, 
          (DL_step * min(shw2.collections[0].get_paths()[0].vertices[:, 1]) / 100.0) + DL_min - DL_step / 2.0, 
          (DL_step * max(shw2.collections[0].get_paths()[0].vertices[:, 1]) / 100.0) + DL_min - DL_step / 2.0))

    plt.savefig(filename)





