import numpy


def alignment(X,D):
    N_fft=D.shape[1]
    X_dft = numpy.fft.fft(X, N_fft, 0)

    X_aligned_padded = numpy.fft.ifft((1 / D.T) * X_dft,None,0)

    signal_length = X.shape[0]

    if N_fft > signal_length:
        X_aligned = X_aligned_padded[0 : signal_length, :]
    else:
        X_aligned = X_aligned_padded;


    if numpy.all(numpy.isreal(X)):
        X_aligned = numpy.real(X_aligned)
    return X_aligned


