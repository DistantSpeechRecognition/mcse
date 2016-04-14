import numpy
from dft_radian_frequencies import *
from time_delay_of_arrival import *


def noise_field_coherence_matrix(noise_field_type, sensor_positions, noise_position, c, N_fft, F_s):
    noise_field_type = noise_field_type.lower()
    num_sensors = sensor_positions.shape[1]
    C=numpy.zeros((num_sensors, num_sensors, N_fft),dtype=complex)
    omega = dft_radian_frequencies(N_fft)
    ones_vector = numpy.ones((N_fft))
    if noise_field_type=='diffuse' or noise_field_type=='d':
        for sensor_index_1 in range(num_sensors-1, -1, -1):
            C[sensor_index_1, sensor_index_1, :] = ones_vector
            for sensor_index_2 in range(sensor_index_1 - 1, -1, -1):
                sensor_distance_current = numpy.sqrt(numpy.sum((sensor_positions[:, sensor_index_1] - sensor_positions[:, sensor_index_2]) ** 2, 0))
                C_current =numpy.sinc(omega * sensor_distance_current / c * F_s / numpy.pi)
                C[sensor_index_1, sensor_index_2, :] = C_current
                C[sensor_index_2, sensor_index_1, :] = numpy.conj(C_current)
    elif noise_field_type=='localized' or noise_field_type== 'localised' or noise_field_type== 'l':
        T,a = time_delay_of_arrival(sensor_positions, noise_position, c, F_s)
        for sensor_index_1 in range(num_sensors-1, -1 , -1):
            C[sensor_index_1, sensor_index_1, :] = ones_vector
            for sensor_index_2 in range(sensor_index_1 - 1, -1, -1):
                C_current = numpy.exp(-1j * omega * (T[sensor_index_1] - T[sensor_index_2]))
                C[sensor_index_1, sensor_index_2, :] = C_current
                C[sensor_index_2, sensor_index_1, :] = numpy.conj(C_current)

    else:
        print('Noise field type',noise_field_type,' not supported')
    return C
