import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import pandas as pd
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import Point, Polygon

proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

def geodesic_point_buffer(lat, lon, dist):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(dist, cap_style=3)  # distance in metres
    return Polygon(transform(project, buf).exterior.coords[:])

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    else: pass

# Configure CRHD format (street widths and colors)
street_types = ['service', 'residential', 'tertiary_link', 'tertiary', 'secondary_link', 'primary_link',
                 'motorway_link', 'secondary', 'trunk', 'primary', 'motorway']
#major_roads = ['secondary', 'trunk', 'primary', 'motorway']
#minor_roads = ['secondary', 'service', 'residential', 'tertiary_link', 'tertiary', 'secondary_link', 'primary_link', 'motorway_link']

street_widths = {'service' : 1,
                 'residential' : 1,
                 'tertiary_link':1,
                 'tertiary': 2,
                 'secondary_link': 2,
                 'primary_link':2,
                 'motorway_link':2,
                 'secondary': 3,
                 'trunk': 4,
                 'primary' : 4,
                 'motorway' : 2.5}

# Colors for CRHDs with black background
street_colors_b = {'service' : 'blue',
                 'residential' : 'blue',
                 'tertiary_link': 'blue',
                 'tertiary':'cornflowerblue',
                 'secondary_link': 'cornflowerblue',
                 'primary_link':'cornflowerblue',
                 'motorway_link':'cornflowerblue',
                 'trunk_link':'cornflowerblue',
                 'secondary': 'lightblue',
                 'trunk': 'white',
                 'primary' : 'white',
                 'motorway' : 'lightgrey'}

# Colors for CRHDs with white background
street_colors_w = {'service' : 'skyblue',
                 'residential' : 'skyblue',
                 'tertiary_link': 'skyblue',
                 'tertiary':'cornflowerblue',
                 'secondary_link': 'cornflowerblue',
                 'primary_link':'cornflowerblue',
                 'trunk_link':'cornflowerblue',
                 'motorway_link':'darkred',
                 'secondary': 'darkblue',
                 'trunk': 'black',
                 'primary' : 'black',
                 'motorway' : 'darkred'}


def PlotCRHD(center_point, dist, name=None, save_path=None, dpi=100, format='png'):
    '''
    Plot the CRHD for a given central coordinate and radius.

    :param center_point: Tuple of central coordinates -> (lng, lat).
    :param dist: Radius in meter -> int.
    :param name: Name of the plotted urban zone -> str.
    :param save_path: The save path of the CRHD image. If None, the CRHD would be shown inline.
    :param dpi: Image dpi.
    :param format: Image format.
    :return: If save_path is None, return the CRHD image. If save_path is not None, return None.
    '''
    try:
        G = ox.graph_from_point(center_point=center_point, network_type='all', dist=dist)
        gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
        for stype in street_types:
            if (gdf.highway==stype).any():
                gdf[gdf.highway==stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_w[stype])
    except:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.axis('off')
    if save_path:
        filename = os.path.join(save_path, f'{str(dist)}_{name}.{format}')
        plt.savefig(filename, dpi=dpi, bbox_inches='tight',pad_inches=0, format=format)
        plt.close()
    else:
        #plt.show()
        return plt.gcf()


def PlotCRHD_grid(idx, center_point, dist, save_path, dpi=100, format='png'):

    filename = os.path.join(save_path, f'{idx}.{format}')
    G = ox.graph_from_point(center_point=center_point, network_type='all', dist=dist)
    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
    for stype in street_types:
        if (gdf.highway==stype).any():
            gdf[gdf.highway==stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_w[stype])

    plt.axis('off')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight',pad_inches=0, format=format)
    plt.close()

def PlotCRHD_grid_local(idx, center_point, dist, save_path, gdf, dpi=100, format='png'):

    filename = os.path.join(save_path, f'{idx}.{format}')
    #G = ox.graph_from_point(center_point=center_point, network_type='all', dist=dist)
    #gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf.fclass = gdf.fclass.map(lambda x: x[0] if isinstance(x, list) else x)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
    for stype in street_types:
        if (gdf.fclass==stype).any():
            gdf[gdf.fclass==stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_w[stype])

    plt.axis('off')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight',pad_inches=0, format=format)
    plt.close()

def PlotList(grids, dist, save_path, if_local, dpi=100, format='png'):
    failures = list(grids.index)
    iter = 1
    while failures and iter<=3:
        f = []
        for _id in failures:
            try:
                print(f'Round{str(iter)}: drawing {_id}')
                if if_local:
                    road_in_grid = gpd.clip(self.local_links, grids.loc[_id])
                    PlotCRHD_grid_local(_id, grids.loc[_id].coord, dist, save_path, road_in_grid, dpi=100, format='png')
                else:
                    PlotCRHD_grid(_id, grids.loc[_id].coord, dist, save_path, dpi=dpi, format=format)
            except:
                print('-'*10, f'Round{str(iter)}: {_id} image generation failed', '-'*10)
                f.append(_id)
        failures = f
        iter += 1

def PlotCity(grid_path, save_path, city_name, dist, if_local=False, start_idx=0, dpi=100, format='png'):
    '''
    # Plot CRHD images for a gridded city.

    :param dist: radius in meter -> int.
    :param grid_path: filepath of the grids Shapefile.
    :param save_path: filepath to save the CRHD images.
    :param dpi: image dpi.
    :param format: image format
    :return: None
    '''
    mkdir(save_path)
    grids = gpd.read_file(grid_path).to_crs(epsg=4326).iloc[start_idx:,:]
    grids.index = grids.index.map(lambda x: city_name+'_'+str(x))
    grids['coord'] = [(centroid.y, centroid.x) for centroid in grids.geometry.centroid]
    PlotList(grids, dist, save_path, if_local, dpi, format)
    print('complete!')

def PlotCityList(list_path, data_dic, dist, dpi=100, format='png'):
    citylist = pd.read_csv(list_path, encoding='utf-8', header=0)
    record_path = os.path.join(data_dic, 'completed_cites.txt')
    if os.path.exists(record_path):
        with open(record_path, 'r') as f:
            completed_num = len(f.read().split('\n'))-1
        citylist = citylist.iloc[completed_num:, :]

    for i, row in citylist.iterrows():
        print(f'Currently working on {row.city}')

        grid_path = os.path.join(data_dic, row.city, 'grids', 'grids.shp')
        save_path = os.path.join(data_dic, row.city, 'images')
        mkdir(save_path)

        start_idx = len(os.listdir(save_path))
        if row.country == 'China':
            PlotChineseCity(grid_path, save_path, row.city, dist, start_idx, dpi=dpi, format=format)
        else:
            PlotCity(grid_path, save_path, row.city, dist, start_idx, dpi=dpi, format=format)

        with open(record_path, 'a+') as f:
            f.write(row.city+"\n")
        print(f'{i}th city ({row.city}) completed!')
        print('='*100)

class CityListPloter():
    def __init__(self, dpi = 100, img_format = 'png'):
        self.local_links = None
        self.dpi = dpi
        self.img_format =  img_format

    def load_local_links(self, local_links_path):
        self.local_links = gpd.read_file(local_links_path)
        print('local links loaded!')

    def PlotCRHD_grid(self, idx, center_point, dist, save_path):

        filename = os.path.join(save_path, f'{idx}.{self.img_format}')
        G = ox.graph_from_point(center_point=center_point, network_type='all', dist=dist)
        gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
        for stype in street_types:
            if (gdf.highway == stype).any():
                gdf[gdf.highway == stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_w[stype])

        plt.axis('off')
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight', pad_inches=0, format=self.img_format)
        plt.close()

    def PlotCRHD_grid_local(self, idx,  save_path, gdf):

        filename = os.path.join(save_path, f'{idx}.{self.img_format}')
        # G = ox.graph_from_point(center_point=center_point, network_type='all', dist=dist)
        # gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf.fclass = gdf.fclass.map(lambda x: x[0] if isinstance(x, list) else x)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
        for stype in street_types:
            if (gdf.fclass == stype).any():
                gdf[gdf.fclass == stype].plot(ax=ax, linewidth=street_widths[stype], edgecolor=street_colors_w[stype])

        plt.axis('off')
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight', pad_inches=0, format=self.img_format)
        plt.close()

    def PlotList(self, grids, dist, save_path, if_local):
        if if_local:
            city_boundary = grids.dissolve().buffer(0.2)
            #city_boundary = gpd.GeoDataFrame([{'geometry': city_boundary}], crs='EPSG:4326')
            city_roads = gpd.clip(self.local_links, city_boundary)
        failures = list(grids.index)
        iter = 1
        while failures and iter <= 3:
            f = []
            for _id in failures:
                try:
                    print(f'Round{str(iter)}: drawing {_id}')
                    if if_local:
                        boundary = geodesic_point_buffer(*grids.loc[_id].coord, dist)
                        boundary = gpd.GeoDataFrame([{'geometry': boundary}], crs='EPSG:4326')
                        road_in_grid = gpd.clip(city_roads, boundary)
                        self.PlotCRHD_grid_local(_id, save_path, road_in_grid)
                    else:
                        self.PlotCRHD_grid(_id, grids.loc[_id].coord, dist, save_path)
                except:
                    print('-' * 10, f'Round{str(iter)}: {_id} image generation failed', '-' * 10)
                    f.append(_id)
            failures = f
            iter += 1

    def PlotCity(self, grid_path, save_path, city_name, dist, if_local=False, start_idx=0):
        '''
        # Plot CRHD images for a gridded city.

        :param dist: radius in meter -> int.
        :param grid_path: filepath of the grids Shapefile.
        :param save_path: filepath to save the CRHD images.
        :param dpi: image dpi.
        :param format: image format
        :return: None
        '''
        mkdir(save_path)
        grids = gpd.read_file(grid_path).to_crs(epsg=4326).iloc[start_idx:, :]
        grids.index = grids.index.map(lambda x: city_name + '_' + str(x))
        grids['coord'] = [(centroid.y, centroid.x) for centroid in grids.geometry.centroid]
        self.PlotList(grids, dist, save_path, if_local)
        print('complete!')

    def PlotCityList(self, list_path, data_dic, dist, local_links_path=None):
        citylist = pd.read_csv(list_path, encoding='utf-8', header=0)
        record_path = os.path.join(data_dic, 'completed_cites_china.txt')
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                completed_num = len(f.read().split('\n')) - 1
            citylist = citylist.iloc[completed_num:, :]

        for i, row in citylist.iterrows():
            print(f'Currently working on {row.city_ascii}')

            grid_path = os.path.join(data_dic, row.city_ascii, 'grids', 'grids.shp')
            save_path = os.path.join(data_dic, row.city_ascii, 'images')
            mkdir(save_path)

            start_idx = len(os.listdir(save_path))
            if row.country == 'China' and self.local_links is None:
                self.load_local_links(local_links_path)

            #if row.country == 'China':
            self.PlotCity(grid_path, save_path, row.city, dist, row.country=='China', start_idx)

            with open(record_path, 'a+') as f:
                f.write(row.city_ascii + "\n")
            print(f'{i}th city ({row.city_ascii}) completed!')
            print('=' * 100)

if __name__ == '__main__':
    pass