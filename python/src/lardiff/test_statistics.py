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
