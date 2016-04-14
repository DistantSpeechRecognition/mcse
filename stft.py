import numpy


def stft(Y, winlen, overlap):
    N = Y.shape[1]
    wHamm = numpy.hamming(winlen)
    nfft = 2**(nextpow2(winlen)+1)
    if Y.shape[0] % (winlen-overlap) >= 0:
        nframes = (Y.shape[0]/(winlen-overlap))+1
    else:
        nframes = (Y.shape[0]/(winlen-overlap))
    #print(N,nfft,nframes)
    Ynew = numpy.zeros((N, nfft, nframes))
    yf = numpy.zeros((winlen, nframes))
    yw = numpy.zeros((winlen, nframes))
    fyw = numpy.zeros((N, nfft, nframes), dtype=complex)
    extra_zeros = nframes*winlen - (nframes-1)*overlap - Y.shape[0]-overlap
    first_zeros = numpy.zeros(overlap)
    add_zeros = numpy.zeros(extra_zeros)
    YY = numpy.zeros((nframes*winlen - (nframes-1)*overlap, N))
    #yy=numpy.concatenate(add_zeros,Y)
    #print(y.shape)
    #print(add_zeros)
    #exit(0)
    for i in range(0,N):
        tempo = numpy.concatenate((first_zeros, Y[:, i]), 1)
        #print(add_zeros.shape, Y[:,1].shape)
        YY[:,i] = numpy.concatenate((tempo, add_zeros), 1)
        #print(YY.shape)
        array_ind_down = 0
        array_ind_up = winlen
        for j in range(0, nframes, 1):
            yf[:, j] = YY[array_ind_down:array_ind_up, i]
            yw[:, j] = yf[:, j] * wHamm
            array_ind_down = array_ind_down+(winlen-overlap)
            array_ind_up = array_ind_up+(winlen-overlap)

        #winlen,nframe=yf.shape

        fyw[i, :, :] = numpy.fft.fft(yw, nfft, 0)

    return fyw, nframes, nfft


def nextpow2(i):
    n = 2
    pwr = 1
    while n < i:
        n *= 2
        pwr += 1
    return pwr
