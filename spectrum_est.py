import numpy


def spectrum_est(Ym, Im, winlen, frame_id=0, a=0, Pyy_last=[]):

    epsilon = 1e-8
    N, nfft, nframes=Ym.shape
    wHamm = numpy.hamming(winlen)
    denom = numpy.linalg.norm(wHamm)**2
    Pyy = numpy.zeros((N, N, nfft))


    if Pyy_last==[]:
        for k in range (0, nfft):
            Pyy[:, :, k] = numpy.real(numpy.dot(Ym[:, k, frame_id].conj()[numpy.newaxis].T, Ym[:, k, frame_id][numpy.newaxis])/denom) # attention frame_id or frame_id-1?????
    else:
        for k in range(0,nfft):
            Pyy[:, :, k] = (1-a)*numpy.real(numpy.dot(Ym[:, k, frame_id].conj()[numpy.newaxis].T, Ym[:, k, frame_id][numpy.newaxis])/denom)\
                + a*Pyy_last[:, :, k]

    return Pyy
    #if nargout < 2
    #else
    #Pii=zeros(N,N,nfft);%3D Matrix



    #if (Pii_last==0):
    #	for k in range (0,nfft):
    #		Pii[:,:,k]=numpy.real(dot(Im[:,k,frame_id],Im[:,k,frame_id].conj().T)/denom)
    #else:
    #	for k in range (0,nfft):
    #		Pii[:,:,k]=(1-a)*numpy.real(dot(Im[:,k,frame_id],Im[:,k,frame_id].conj().T)/denom)+a*Pii_last[:,:,k]
