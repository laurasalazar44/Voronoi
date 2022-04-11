from gcd_tarea2 import *

coord_x_y = np.loadtxt('airports_CO.dat', usecols=(1, 0))
coord_h = np.loadtxt('airports_CO.dat', usecols=(2))
name_airport = np.loadtxt(
    'airports_CO.dat',
    dtype='str',
    usecols=(5),
    delimiter='"')

airports = FlightOperations(coord_x_y, coord_h, name_airport, 'borders_CO.dat')
vor = airports.Voronoi

# Segundo Punto
radio, centroid = airports.largest_empty_circle()
print('El radio del circulo vacío más grande es: ', radio)
print('Las coordenadas del centro del circulo vacío más grande son:', centroid)

# Tercer Punto
print('Los aeropuertos más cercanos son: ', airports.closest_pair_points())
print('Los aeropuertos más lejanos son: ', airports.fartest_pair_points())

# Cuarto Punto
path = airports.path(
    'General Alfredo Vasquez Cobo', 'Baracoa', threshold=300)
print(path)

# Primer Punto
airports.plot()
