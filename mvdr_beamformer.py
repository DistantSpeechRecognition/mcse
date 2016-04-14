import numpy


def mvdr_beamformer(D, C, mu_dB):
    W = numpy.zeros(D.shape,dtype=complex)
    num_sensors, N_fft=D.shape
    mu = 10 ** (mu_dB / 10)
    for freqBinIndex in range(N_fft-1,-1,-1):
        D_current = D[:, freqBinIndex].T
        P = C[:, :, freqBinIndex] + mu * numpy.eye(num_sensors)
        nominator = numpy.dot(numpy.linalg.inv(P),D_current)
        denominator = numpy.dot(nominator.conj().T,D_current)
        W[:, freqBinIndex] = nominator / denominator
    return W
