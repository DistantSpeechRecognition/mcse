from __future__ import division
import numpy as np
import scipy.io
from scipy.special._ufuncs import expi
from alignment import *
from propagation_vector_free_field import *
from noise_field_coherence_matrix import *
from noise_field_coherence_matrix_limit import *
from noise_field_coherence_matrix_alignment_reestimated import *
from mvdr_beamformer import *
from spectrum_est import *
from stft import *
from ola import *
from scipy.special import *
from sympy import expint

def Pspec_est(p_yy_arr, c_arr, w_arr, limits, fs):
    # Estimates Ps, Ps_zel, Pv_eff and Pout based on Pyy for a single frame.

    ls = 1
    if not isinstance(limits, (int, long)):
        ls = limits.size()
    epsilon=1e-8
    r = p_yy_arr.shape
    N = r[0]
    nfft = r[2]

    #Estimation of the Power Spectrum for
    #the source signal for the specified frame (frame_id).
    p_s_arr = np.zeros((nfft, 1))

    #Estimation of the Power Spectrum for
    #the source signal for the specified frame (frame_id). Only for Zelinski
    #method.
    p_s_zel_arr = np.zeros((nfft, 1))
    p_v_arr = np.zeros((nfft, 1))
    #Estimation of the Power Spectrum for
    #the noise at the output of the MVDR beamformer for the specified frame (frame_id).
    #for different limits.
    p_v_eff_arr = np.zeros((nfft, ls))
    add_term_v=np.zeros((nfft,1))
    add_term_zel=np.zeros((nfft,1))
    #Estimation of the Power Spectrum for the output of the
    #MVDR beamformer. Pout is used only in Mccowan and Zelinski.
    p_out_arr = np.zeros((nfft, 1))
    for n in range(0, N-1):
        for m in range(n+1, N):
            p_yy_arr_nm = p_yy_arr[n, m, :].reshape(nfft, 1, order='F')
            p_yy_arr_nn = p_yy_arr[n, n, :].reshape(nfft, 1, order='F')
            p_yy_arr_mm = p_yy_arr[m, m, :].reshape(nfft, 1, order='F')
            c_arr_nm = c_arr[n, m, :].reshape(nfft, 1, order='F')
            add_term_p_s = (p_yy_arr_nm.copy() - 0.5 * (p_yy_arr_nn.copy() + p_yy_arr_mm.copy()) * c_arr_nm.real.copy()) / (1-c_arr_nm.real.copy())
            add_term_p_s[add_term_p_s < 0] = epsilon
            p_s_arr += add_term_p_s.copy()
	    
	    add_term_zel = p_yy_arr_nm.real.copy()
            add_term_zel[add_term_zel<0] = epsilon
            p_s_zel_arr += add_term_zel

            add_term_v = (0.5*(p_yy_arr_nn + p_yy_arr_mm) - p_yy_arr_nm)/(1-c_arr_nm.real.copy())
            add_term_v[add_term_v < 0] = epsilon
            p_v_arr += add_term_v
                #Must hold the constraint |C(m,n,k)|<= limit for all m~=n
            #where limit < 1.
    p_s_arr = 2 * p_s_arr / (N*(N-1))
    p_s_zel_arr = 2 * p_s_zel_arr / (N * (N-1))
    p_v_arr = 2 * p_v_arr / (N*(N-1))

    f = fs * np.arange(0, nfft, dtype=float).conj().T / nfft - fs/2
    f = np.fft.ifftshift(f)
    for r in range(0, ls):
        if ls>1:
            lim = limits[r]
        else:
            lim = limits
        if lim > fs/2:
            p_v_eff_arr[:, r] = p_v_arr[:, 0].copy()
        else:
            for k in range(0, nfft):
                if abs(f[k]) >= limits[r]:
                    p_v_eff_arr[k, r] = p_v_arr[k,0] * np.dot(w_arr[:, k].conj().T, np.dot( c_arr[:, :, k], w_arr[:, k]))
                else:
                    p_v_eff_arr[k, r] = p_v_arr[k,0]

    for n in range(0, N):
        p_out_arr += p_yy_arr[n, n, :].reshape(nfft, 1).real

    p_out_arr /= N

    return p_s_arr, p_s_zel_arr, p_v_eff_arr, p_out_arr


def G_est(p_s_arr, p_s_zel_arr, p_v_eff_arr, p_out_arr, k_arr):

    #K: MVDR output
    ls = p_v_eff_arr.shape[1]
    n_fft =  p_s_arr.shape[0]

    # In each column of G we store the estimated post-filter for the
    # corresponding method for a single frame.
    # column 1-->Zelinski
    # column 2-->McCowan
    # column 3-->Proposed
    # column 4-->STSA-MVDR
    # column 5-->STLSA-MVDR
    temp=np.zeros((n_fft, 1),np.float64)
    g_arr = np.zeros((n_fft, 2+3*ls),dtype=np.float64)
    kappa=np.zeros((n_fft, 1),dtype=complex)
    nu_mvdr_arr=np.zeros((n_fft,1),dtype=np.float64)
    #Zelinski
    g_arr[:, 0] = p_s_zel_arr[:, 0] / p_out_arr[:,0]
    #McCowan
    g_arr[:, 1] = p_s_arr[:, 0] / p_out_arr[:, 0]
    for i in range(1, ls+1):
        # Proposed
        g_arr[:, 2+3*(i-1)] = p_s_arr[:, 0]/(p_s_arr[:, 0] + p_v_eff_arr[:, i-1])
	nu_mvdr_arr[:,0] = g_arr[:, 2+3*(i-1)] * (k_arr * k_arr.conj()) / p_v_eff_arr[:, i-1]
	
        # Ephraim-Malah STSA estimator [Ephraim 1984].
        g_arr[:, 3+3*(i-1)] = gamma(1.5) * (np.sqrt(nu_mvdr_arr[:,0]) / (k_arr * k_arr.conj())) * p_v_eff_arr[:, i-1] * \
                              np.exp(-nu_mvdr_arr[:,0]/2) * ((1 + nu_mvdr_arr[:,0]) * iv(0, nu_mvdr_arr[:,0]/2) + nu_mvdr_arr[:,0] * iv(1, nu_mvdr_arr[:,0]/2))


        # Check if the post-filter has inf or Nan
        # values. Then for these frequencies we use the Wiener post-filter
        ind = np.where(np.isnan(g_arr[:, 3+3*(i-1)]))
	ind2 = np.where(np.isinf(g_arr[:, 3+3*(i-1)]))
        g_arr[ind, 3+3*(i-1)] = g_arr[ind, 2+3*(i-1)].copy()
	g_arr[ind2, 3+3*(i-1)] = g_arr[ind2, 2+3*(i-1)].copy()
        # Ephraim-Malah STLSA estimator [Ephraim 1985]
        temp[:,0] = np.exp(0.5 * expn(1,nu_mvdr_arr[:,0]))
	g_arr[:, 4+3*(i-1)] = g_arr[:, 2+3*(i-1)].copy() * temp[:,0]

    return g_arr

# Multichannel Speech Enhancement with MVDR beamforming + post-filtering
#
# Inputs:
# ------
# X: 
# sensor_positions: the positions of the sensors. sensor_positions(:, n)
#                   are the coordinates of the n-th sensor.
# source_position: the source position. source_position(:, 1) are the
#                  source coordinates.
# noise_field_type: 'd','diffuse' or 'l','localized'.
# noise_position: only for localized noise fields, the posittion of the
#                 noise source. noise_position(:, 1) are the coordinates of
#                 noise source.
# limit: the maximum value over which the real part of the coherence
#        function is clipped to avoid 1 - real(C) == 0.
# mu_dB: white noise gain constraint in dB (for MVDR), usually in the range
#        [-40,-10]dB.
# a: forgetting factor for recursive Welch periodogram (see spectrum_est).
# c: speed of sound.
# F_s: sampling frequency.
# limits: limits = F_s seems to work best for SSNR improvement, but is not
#         optimal if distortion (e.g. Itakura-Saito distortion measure) is
#         considered.
#
# Outputs:
# -------
# Z: mvdr,zelinski,mccowan, mmse, stsa, log-stsa, 1 per column


def multichannel_speech_enhancement(x_arr, sensor_positions, source_position, window_length, window_overlap,
                                    noise_field_type, noise_position, limit, mu_db, a, c, f_s, limits):

    # Steps 1,2: Find delays and compensate
    n_samples, n_channels = x_arr.shape
    n_fft = 2**(nextpow2(n_samples) + 1)
    d_arr = propagation_vector_free_field(sensor_positions, source_position, c, n_fft, f_s)
    x_arr = alignment(x_arr, d_arr)

    # Step 3: Estimate the noise model.
    n_fft = 2**(nextpow2(window_length) + 1)
    c_arr = noise_field_coherence_matrix(noise_field_type, sensor_positions, noise_position, c, n_fft, f_s)
    d_arr = propagation_vector_free_field(sensor_positions, source_position, c, n_fft, f_s)
    c_arr_alignment_reestimated = noise_field_coherence_matrix_alignment_reestimated(c_arr, d_arr)
    c_arr = noise_field_coherence_matrix_limit(c_arr, limit)

    #Step 4: Find the MVDR weights for the diffuse noise field.
    d_arr = np.ones((n_channels, n_fft))
    w_mvdr_arr = mvdr_beamformer(d_arr, c_arr_alignment_reestimated, mu_db)


    #Step 5:Frame input signals and Apply STFT.
    y_m_arr, n_frames, n_fft = stft(x_arr, window_length, window_overlap)

    #Step 6: Estimate the Transfer functions of all the post-filters.
    ls = 1
    if not isinstance(limits, (int, long)):
        ls = limits.size()

    # Estimation of the post-filters for all the frames.
    g_arr = np.zeros((n_fft, n_frames, 2+3*ls),dtype=np.float64)

    # Output of the MVDR beamformer in the frequency domain.
    k_arr = np.zeros((n_fft, n_frames), dtype=complex)

    p_yy_arr = spectrum_est(y_m_arr, [], window_length)
    for k in range(0, n_fft):
        k_arr[k, 0] = np.dot(w_mvdr_arr[:, k].conj().T,y_m_arr[:, k, 0])

    p_s_arr, p_s_zel_arr, p_v_eff_arr, p_out_arr = Pspec_est(p_yy_arr, c_arr, w_mvdr_arr, limits, f_s)


    g_arr[:, 0, :] = G_est(p_s_arr, p_s_zel_arr, p_v_eff_arr, p_out_arr, k_arr[:, 0])
    for frame_id in range(1, n_frames):
        p_yy_arr = spectrum_est(y_m_arr, [], window_length, frame_id, a, p_yy_arr)
        for k in range(0, n_fft):
            k_arr[k, frame_id] = np.dot(w_mvdr_arr[:, k].conj().T, y_m_arr[:, k, frame_id])

        p_s_arr, p_s_zel_arr, p_v_eff_arr, p_out_arr = Pspec_est(p_yy_arr, c_arr, w_mvdr_arr, limits, f_s)
        g_arr[:, frame_id, :] = G_est(p_s_arr, p_s_zel_arr, p_v_eff_arr, p_out_arr, k_arr[:, frame_id])

    # Step 7: Find the outpus of the System.
    # Outputs of the post-filters.
    g_arr=g_arr.real

    z_arr = np.zeros((n_samples, 3+3*ls),dtype=complex)
    #MVDR output
    z = ola(k_arr, window_length, window_overlap, n_samples)
    z[np.fabs(z.imag) < 10e-10] = z[np.fabs(z.imag) < 10e-10].real
    z_arr[:, 0] = z


    #Post-Filter Outputs
    for i in range(0, 2+3*ls):
        t_arr = g_arr[:, :, i] * k_arr
        z = ola(t_arr, window_length, window_overlap, n_samples)
        z[np.fabs(z.imag) < 10e-8] = z[np.fabs(z.imag) < 10e-8].real
        z_arr[:, i+1] = z

    return z_arr

