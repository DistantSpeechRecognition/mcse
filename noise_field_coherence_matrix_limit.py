import numpy


def noise_field_coherence_matrix_limit(C, limit):
	num_sensors = C.shape[0]
	C_limited = C+0j
	for sensor_index_1 in range(num_sensors-1, 0, -1):
                for sensor_index_2 in range(sensor_index_1-1,-1,-1):
        		C_limited_current = C_limited[sensor_index_1, sensor_index_2, :]
        		C_limited_current_real_part = numpy.real(C_limited_current)
        		limit_indices = C_limited_current_real_part > limit
        		C_limited_current_real_part[limit_indices] = limit * (C_limited_current_real_part[limit_indices] / numpy.abs(C_limited_current_real_part[limit_indices]))
        		C_limited_current = C_limited_current_real_part + 1j * numpy.imag(C_limited_current)
        		C_limited[sensor_index_1, sensor_index_2, :] = C_limited_current
        		C_limited[sensor_index_2, sensor_index_1, :] = numpy.conj(C_limited_current)
	return C_limited
