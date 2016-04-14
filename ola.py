import numpy
import scipy.io




def ola(Z,winlen,overlap,slen,opt=[]):
	shift=winlen-overlap
	Z=numpy.fft.ifft(Z,None,0)
	Z=Z[0:winlen,:]
	wHamm=numpy.hamming(winlen)
	Wo=numpy.sum(wHamm)
	nframe=Z.shape[1]
	z=numpy.zeros(winlen+(nframe-1)*shift)
	z=z+0j
	for i in range(1,nframe):
		ind1=(i)*shift+1
		ind2=overlap-(i-1)*shift
		if (ind2>0):
			Z[ind1-1:winlen,0]=Z[ind1-1:winlen,0]+Z[0:ind2,i]
		else:
			break
	z[0:winlen]=Z[0:winlen,0]
	for jj in range(1,nframe-1):
		for i in range(jj+1,nframe):
			n1=overlap+1-(i-jj)*shift
			n2=winlen-(i-jj)*shift
			if (n2>0):
				if (n1<1):
					k=1-n1
					n1=1
					Z[overlap:winlen,jj]=Z[overlap:winlen,jj]+numpy.array([numpy.zeros(k),Z[n1-1:n2,i]])
				else:
					Z[overlap:winlen,jj]=Z[overlap:winlen,jj]+Z[n1-1:n2,i]
			else:
				break
		z[winlen+(jj-1)*shift:winlen+(jj)*shift]=Z[overlap:winlen,jj]

	z[winlen+(nframe-2)*shift:winlen+(nframe-1)*shift]=Z[overlap:winlen,nframe-1]

	if (opt=='nodelay'):
		z=z[0:slen]
		z=z*(shift/Wo)
		
	else:
		zhelp=z[overlap:overlap+slen]
		z=zhelp
		z=z*(shift/Wo)


	return z
	#if nargin < 5
    	#opt=[];
