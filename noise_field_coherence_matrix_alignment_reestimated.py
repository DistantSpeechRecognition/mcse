import numpy

def noise_field_coherence_matrix_alignment_reestimated(C,D):
	num_sensors=D.shape[0]
	C_alignment_reestimated=C+0j
	for sensor_index_1 in range(num_sensors-1, 0, -1):
		for sensor_index_2 in range(sensor_index_1-1,-1,-1):
			correction_factor_current = 1/D[sensor_index_1,:] * numpy.conj( 1/D[sensor_index_2, :])
			correction_factor_current = correction_factor_current / numpy.abs(correction_factor_current)
			
			correction_factor_current=correction_factor_current.T
			C_alignment_reestimated_current = correction_factor_current * numpy.squeeze(C[sensor_index_1, sensor_index_2, :]).T
			C_alignment_reestimated[sensor_index_1, sensor_index_2, :] = C_alignment_reestimated_current
			C_alignment_reestimated[sensor_index_2, sensor_index_1, :] = numpy.conj(C_alignment_reestimated_current)
	return C_alignment_reestimated
