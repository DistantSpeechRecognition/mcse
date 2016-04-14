import numpy
from time_delay_of_arrival import *
from dft_radian_frequencies import *


def propagation_vector_free_field(sensor_positions, source_position, c, N_fft, F_s):
    D = numpy.zeros((sensor_positions.shape[1],N_fft))
    D = D+0j
    T, a = time_delay_of_arrival(sensor_positions, source_position, c, F_s)
    omega = dft_radian_frequencies(N_fft)

    for sensor_index in range(sensor_positions.shape[1]-1, -1, -1):
        D[sensor_index, :] = a[sensor_index] * numpy.exp(-1j * omega * T[sensor_index])
    return D
