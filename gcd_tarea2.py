from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d
from scipy.spatial import ConvexHull
from scipy.spatial import distance
import matplotlib as mpl
import matplotlib.cm as cm
import random


class FlightOperations:

    def __init__(self, pts, elevation, name, border):
        '''
        Función init donde se define la clase, sus parametros de entrada
        son el conjunto de puntos a trabajar (aeropuertos),
        la elevación de cada punto sobre el nivel del mar, el nombre de cada
        aeropuerto y el dataset con la información del mapa a graficar
        '''
        self.Voronoi = Voronoi(pts, incremental=True)
        self.hull = ConvexHull(pts)
        self.elevation = elevation
        self.name = name
        self.colombia = border

    def calculate_distance_max(self, point, arr):
        '''
        Función encargada de calcular la distancia más larga
        entre un punto y un array
        '''
        dist2 = np.max(
            np.sqrt((point[0] - arr[:, 0])**2 + (point[1] - arr[:, 1])**2))
        dist = np.argmax(
            np.sqrt((point[0] - arr[:, 0])**2 + (point[1] - arr[:, 1])**2))
        points = point, arr[dist]
        return points, dist2, np.int(dist)

    def calculate_distance_min(self, point, arr):
        '''
        Función encargada de calcular la distancia más corta entre
        un punto y un array
        '''
        dist2 = np.min(
            np.sqrt((point[0] - arr[:, 0])**2 + (point[1] - arr[:, 1])**2))
        dist = np.argmin(
            np.sqrt((point[0] - arr[:, 0])**2 + (point[1] - arr[:, 1])**2))
        points = point, arr[dist]
        return points, dist2

    def point_in_hull(self, point, tolerance=0.0001):
        '''
        Función que identifica si el punto dado se encuentra
        dentro del convex hull
        '''
        return all(
            (np.dot(q[:-1], point) + q[-1] <= tolerance)
            for q in self.hull.equations)

    def closest_pair_points(self):
        '''
        Función que retorna los nombres de los aeropuertos más
        cercanos en el diagrama de voronoi
        '''
        vor = self.Voronoi
        dist_min = []
        for arista in vor.ridge_points:
            dist_min.append(distance.euclidean(
                vor.points[arista[0]], vor.points[arista[1]]))
        # minimal = min(dist_min)
        arg_min = np.argmin(dist_min)
        airport1 = self.name[vor.ridge_points[arg_min][0]]
        airport2 = self.name[vor.ridge_points[arg_min][1]]
        colombia = np.loadtxt(self.colombia, unpack=True)
        voronoi_plot_2d(
            vor, show_vertices=False, line_colors='black',
            line_width=1, line_alpha=0.5, point_size=3, s=1)
        plt.plot(*(colombia[1], colombia[0]), linewidth=1.0, color='gray')
        plt.plot([
            vor.points[vor.ridge_points[arg_min][0]][0],
            vor.points[vor.ridge_points[arg_min][1]][0]],
            [
            vor.points[vor.ridge_points[arg_min][0]][1],
            vor.points[vor.ridge_points[arg_min][1]][1]],
            '*r')
        plt.title(f'Los puntos más cercanos son {airport1} y {airport2}')
        plt.show()
        return airport1, airport2

    def fartest_pair_points(self):
        '''
        Función que retorna las coordenadas de los aeropuertos más lejanos
        junto con la distancia entre ambos
        '''
        hull = self.hull
        vor = self.Voronoi
        distance = []
        count1 = 7
        count2 = 1
        while count1 != 1:
            dist = self.calculate_distance_max(
                hull.points[hull.vertices[count1]],
                hull.points[hull.vertices[:-count2]])
            distance.append([dist, count1])
            count1 -= 1
            count2 += 1
        distance = np.array(distance, dtype=object)
        max = np.argmax(distance[:, 1])
        airport1 = self.name[hull.vertices[distance[max][1]]]
        airport2 = self.name[hull.vertices[int(distance[max][0][2])]]
        colombia = np.loadtxt(self.colombia, unpack=True)
        voronoi_plot_2d(
            vor, show_vertices=False, line_colors='black',
            line_width=1, line_alpha=0.5, point_size=3, s=1)
        plt.plot(*(colombia[1], colombia[0]), linewidth=1.0, color='gray')
        plt.plot([
            vor.points[hull.vertices[distance[max][1]]][0],
            vor.points[hull.vertices[int(distance[max][0][2])]][0]],
            [
            vor.points[hull.vertices[distance[max][1]]][1],
            vor.points[hull.vertices[int(distance[max][0][2])]][1]],
            '*r')
        plt.title(f'Los puntos más lejanos son {airport1} y {airport2}')
        plt.show()
        return airport1, airport2

    def plot(self):
        '''
        Función que gráfica el diagrama de Voronoi de los aeropuertos.
        El color de la celda representa la altura sobre el nivel del mar
        de cada aeropuerto, donde el más claro es el más alto y el más
        oscuro el más bajo.
        '''
        vor = self.Voronoi
        vor.add_points([[-500, -500], [-500, 500], [500, -500], [500, 500]])
        vor.close()
        voronoi_plot_2d(
            vor, show_vertices=False, line_colors='black',
            line_width=1, line_alpha=0.5, point_size=3, s=1)
        colombia = np.loadtxt(self.colombia, unpack=True)
        plt.plot(*(colombia[1], colombia[0]), linewidth=1.0, color='gray')
        minima = min(self.elevation)
        maxima = max(self.elevation)
        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if -1 not in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(
                    *zip(*polygon),
                    color=mapper.to_rgba(self.elevation[r])
                    )
        plt.xlim([-83, -65])
        plt.ylim([-5, 15])
        plt.colorbar(mapper)
        plt.title(f'Escala de color dependiendo la altura del aeropuerto')
        plt.show()

    def largest_empty_circle(self):
        '''
        Función encargada de encontrar el largest_empty_circle
        donde se va a ubicar el nuevo aeropuerto
        '''
        vor = self.Voronoi
        hull = self.hull
        new_reg = []
        for r in range(1, len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            for p in region:
                point_is_in_hull = self.point_in_hull(vor.vertices[p])
                if -1 not in region and point_is_in_hull:
                    dist_eu = distance.euclidean(
                        vor.points[r],
                        vor.vertices[p]
                        )
                    new_reg.append([dist_eu, vor.vertices[p]])
        pts_dis = np.array(new_reg, dtype=object)
        ratio = max(pts_dis[:, 0])
        argmax = np.argmax(pts_dis[:, 0])
        centroid = pts_dis[argmax][1]
        fig = voronoi_plot_2d(
            vor, show_vertices=False,
            line_colors='black',
            line_width=1,
            line_alpha=0.5,
            point_size=3,
            s=1
            )
        colombia = np.loadtxt(self.colombia, unpack=True)
        plt.plot(*(colombia[1], colombia[0]), linewidth=1.0, color='gray')
        plt.plot(centroid[0], centroid[1], '*g')
        plt.title(f'El circulo vacío más grande')
        cr = plt.Circle((centroid[0], centroid[1]), radius=ratio, fill=False)
        plt.gca().add_artist(cr)
        plt.show()
        return ratio, centroid

    def recursion(self, origin, path, airports_thr):
        '''
        Función en la cual se hace recursión sobre los vecinos del punto origin
        '''
        vor = self.Voronoi
        pt = origin
        ini_ridges = np.where(vor.ridge_points[:, 0] == pt)[1]
        final_ridges = np.where(vor.ridge_points[:, 1] == pt)[1]
        if not (ini_ridges.size > 0 and final_ridges.size > 0):
            return path
        else:
            if ini_ridges.size > 0:
                for ini in ini_ridges:
                    if ini in airports_thr and ini not in path:
                        path.append(ini)
                        self.recursion((np.array([ini]),), path, airports_thr)
                    else:
                        a = random.choice(ini_ridges)
                        if final_ridges.size > 0:
                            b = random.choice(final_ridges)
                            a = random.choice([a, b])
                        self.recursion((np.array([a]),), path, airports_thr)
            elif final_ridges.size > 0:
                for fin in final_ridges:
                    if fin in airports_thr:
                        path.append(final_ridges[fin])
                        self.recursion((np.array([fin]),), path, airports_thr)
                    else:
                        b = random.choice(final_ridges)
                        self.recursion((np.array([b]),), path, airports_thr)

    def path(self, name_origin, name_destination, threshold=300):
        '''
        Función que crea un camino entre dos aeropuertos dependiendo
        un umbral.
        '''
        vor = self.Voronoi
        if threshold < 0:
            return 'El umbral debe ser mayor que 0'
        else:
            idx_origin = np.where(self.name == name_origin)
            idx_destination = np.where(self.name == name_destination)

            airports_thr = []
            for i in range(len(vor.point_region)):
                if self.elevation[i] >= threshold:
                    airports_thr.append(i)
            if idx_origin not in airports_thr:
                airports_thr.append(idx_origin)
            if idx_destination not in airports_thr:
                airports_thr.append(idx_destination)

            path = [idx_origin[0][0]]

            self.recursion(idx_origin, path, airports_thr)
            path.append(idx_destination[0][0])
            colombia = np.loadtxt(self.colombia, unpack=True)
            voronoi_plot_2d(vor, show_vertices=False, line_colors='black',
                            line_width=1, line_alpha=0.5, point_size=3, s=1)
            plt.plot(*(colombia[1], colombia[0]), linewidth=1.0, color='gray')
            plt.plot(
                [vor.points[i][0] for i in path],
                [vor.points[i][1] for i in path], color='magenta', marker='o')
            plt.title(f'El camino entre:  {name_origin} y {name_destination}')
            plt.show()
            return [self.name[i] for i in path]
