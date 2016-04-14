import numpy


def dft_radian_frequencies(N_fft):
    k_upper=numpy.floor((N_fft-1)/2)
    k_lower=1 + k_upper - N_fft
    array_omega = numpy.arange(k_lower , k_upper+1)
    omega = array_omega.T / N_fft * (2*numpy.pi)
    omega = numpy.fft.ifftshift(omega)
    omega=numpy.array(omega)
    return omega
