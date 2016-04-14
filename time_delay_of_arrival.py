import numpy



def time_delay_of_arrival(sensor_positions, source_position, c, F_s):
	num_sensors = sensor_positions.shape[1]
	r_source = numpy.sqrt(numpy.sum(source_position**2,0))
	r_sensors_source=numpy.zeros((num_sensors,1))
	for sensor_index in range(num_sensors-1,-1,-1):
		r_sensors_source[sensor_index,0]=numpy.sqrt(numpy.sum((sensor_positions[:,sensor_index]-source_position)**2))
	T=(r_sensors_source-r_source) / c * F_s
	a=r_source / r_sensors_source
	return T,a
