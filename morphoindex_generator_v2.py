import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from crhd_generator_v2 import PlotCRHD_grid
from config import config

import tensorflow as tf
config1 = tf.compat.v1.ConfigProto()
config1.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config1)

from Build_model import Build_model



def get_index(grid_path, road_path, building_path, landuse_path):
    
    # load grids
    grids = gpd.read_file(grid_path)
    grids.set_index('id',drop=False,inplace=True)
    grids_for_match = grids[['id', 'geometry']]
    print('Grids loaded!')

    # Road Density & Intersection
    roads = gpd.read_file(road_path)
    roads = roads[roads.geometry.notnull()]
    intersection = gpd.sjoin(grids_for_match, roads, how="right")
    for _id in grids_for_match.index:
        intersection[intersection.id == _id] = gpd.clip(intersection[intersection.id == _id],
                                                        grids_for_match.loc[[_id]])
    intersection = intersection[intersection.geometry.notnull()]
    intersection['length'] = intersection.to_crs(epsg=3857).geometry.length
    grids['RD'] = intersection.groupby('id')['length'].sum()

    # for i, row in grids.iterrows():
    #     try:
    #         centroid = row.geometry.centroid
    #         G = ox.graph_from_point(center_point=(centroid.y, centroid.x), network_type='all', dist=500)
    #         intersections = ox.graph_to_gdfs(G, nodes=True, edges=True)[0]
    #         grids.loc[i,'ID'] = intersections.shape[0]
    #     except:
    #         grids.loc[i,'ID'] = 0
    print('Road completed!')

    # building density & average building area
    buildings = gpd.read_file(building_path)
    buildings = buildings[buildings.geometry.notnull()]
    buildings = buildings.to_crs(epsg=3857)
    buildings['area'] = buildings.geometry.area
    buildings = buildings.to_crs(epsg=4326)
    intersection = gpd.sjoin(grids_for_match, buildings, how="right")
    grids['BuD'] = intersection.groupby('id')['area'].sum()
    grids['ABFA'] = intersection.groupby('id')['area'].mean()
    print('Building completed!')

    # block density & average block area & entropy

    def cross_entropy(row):
        p = np.array(row)
        log_p = np.log(p)
        log_p[log_p==-inf] = 0
        return -np.sum(p*log_p)

    landuse = gpd.read_file(landuse_path)
    landuse = landuse[landuse.geometry.notnull()]

    intersection = gpd.sjoin(grids_for_match, landuse, how="right")
    intersection['geometry'] = intersection.buffer(0)
    for _id in grids_for_match.index:
        intersection[intersection.id == _id] = gpd.clip(intersection[intersection.id == _id],
                                                        grids_for_match.loc[[_id]])
    intersection = intersection[intersection.geometry.notnull()]
    intersection['area'] = intersection.to_crs(epsg=3857).geometry.area
    grids['BID'] = intersection.groupby('id')['area'].count()
    grids['ABA'] = intersection.groupby('id')['area'].mean()
    land_use_areas = intersection.groupby(['id', 'fclass'])['area'].sum().unstack(1).fillna(0)
    land_use_areas['_sum'] = land_use_areas.sum(axis=1)
    land_use_areas = land_use_areas.apply(lambda x: x / x._sum, axis=1)
    grids['LUM'] = land_use_areas.apply(cross_entropy, axis=1)
    print('Block completed!')

    # Save
    #grids.drop(columns=['id'], inplace=True)
    #grids.to_file(save_path if save_path else grid_path,
    #              driver='ESRI Shapefile',
    #              encoding='utf-8')

    return grids

def getIntersection(grids):
    grids.set_index('id', inplace=True)
    print('Grids loaded!')

    i_cnt=[]
    cnt=0
    for i, row in grids.iterrows():
        try:
            centroid = row.geometry.centroid
            G = ox.graph_from_point(center_point=(centroid.y, centroid.x), network_type='all', dist=500)
            intersections = ox.graph_to_gdfs(G, nodes=True, edges=True)[0]
            i_cnt.append(intersections.shape[0])
        except:
            i_cnt.append(0)
        cnt += 1
        if cnt%50==0:
            print(f'current id: {i}')

    grids['ID'] = i_cnt
    #grids.to_file(save_path if save_path else grid_path,
    #              driver = 'ESRI Shapefile',
    #              encoding='utf-8')
    print('Completed!')
    return grids

def dropNonbuiltGrid(grids):
    grids.dropna(subset=['BuD','BID'], how='any', inplace=True)
    #grids.reset_index(drop=True, inplace=True)
    #grids.id = grids.index.map(lambda x: f'{cityName}_{str(x)}')
    #grids.set_index('id', inplace=True)
    #grids.to_file(save_path if save_path else grid_path,
    #              driver = 'ESRI Shapefile',
    #              encoding='utf-8')
    return grids

def get_MorphoIndex(grid_path, road_path, building_path, landuse_path, save_path=None, get_intersection=False, drop_nonbuilt=True):
    '''
    :param grid_path: filepath of grid Shapefile.
    :param road_path: filepath of road Shapefile.
    :param building_path: filepath of building Shapefile.
    :param landuse_path: filepath of land use Shapefile (should include a column named 'fclass' denoting the land use type).
    :param get_intersection: Boolean, whether to calculate number of road intersections.
        *Notice: This calculation needs internet and is very time consuming.
    :return: grids GeoDataFrame
    '''
    print('Start getting mophoindex')
    print('-' * 50)
    grids = get_index(grid_path, road_path, building_path, landuse_path)
    print('-'*50)
    if drop_nonbuilt:
        print('Dropping non-built grids')
        grids = dropNonbuiltGrid(grids)
        print('-'*50)
    if get_intersection:
        print('Getting intersections')
        grids = getIntersection(grids)

    # Save
    grids.drop(columns=['id'], inplace=True)
    grids.to_file(save_path if save_path else grid_path,
                  driver='ESRI Shapefile',
                  encoding='utf-8')
    return grids

def deal_image(img_path):
    image = cv2.imread(img_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thre_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_TRUNC)[1]
    mask = np.where(thre_img > 190)
    image[mask] = 255
    return image

def deal_divide_image(img_path):
    image1 = cv2.imread(img_path)
    image2 = image1.copy()
    gray_img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    thre_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_TRUNC)[1]
    mask1 = np.where(np.logical_or(thre_img < 130, thre_img > 190))
    mask2 = np.where(thre_img > 180)
    image1[mask1], image2[mask2] = 255, 255
    return image1, image2

def xy2vertex(x, y, w, h):
    tlx, tly = x - w // 2, y - h // 2
    brx, bry = x + w // 2, y + h // 2
    return [tlx, tly, brx, bry]

def split_img(img):
    h, w, c = img.shape
    mx, my = w // 2, h // 2
    sub_xys = [
        (mx - w // 8, my - h // 8),
        (mx + w // 8, my - h // 8),
        (mx - w // 8, my + h // 8),
        (mx + w // 8, my + h // 8)]
    subs = []
    for x, y in sub_xys:
        vertex = xy2vertex(x, y, w // 2, h // 2)
        subs.append(img[vertex[1]:vertex[3], vertex[0]:vertex[2]])
    # top-left, top-right, bottom-left, bottom_right
    return subs

def find_neighbor(idx, direction, grids):
    if idx is None:
        return
    row = grids.loc[idx]
    if direction == 'east':
        skewed = Point(row.centroid.x + 0.00833333, row.centroid.y)
    elif direction == 'west':
        skewed = Point(row.centroid.x - 0.00833333, row.centroid.y)
    elif direction == 'south':
        skewed = Point(row.centroid.x, row.centroid.y - 0.00833333)
    elif direction == 'north':
        skewed = Point(row.centroid.x, row.centroid.y + 0.00833333)
    for i, c in enumerate(grids.centroid):
        if skewed.distance(c) <= 10e-6:
            return i
    return

def merge_grids(grids, drop=True, save_path=None):
    if isinstance(grids, str):
        grids = gpd.read_file(grids)
    grids['centroid'] = grids.centroid
    grids['assigned'] = -1
    grids['position'] = None
    group_id = 0
    pass_line = False
    flag = True
    for i in grids.index:
        if not find_neighbor(i, 'east', grids) and i < len(grids.index) - 1 and (grids.loc[i, 'centroid'].y-grids.loc[i+1, 'centroid'].y>0.000001):
            pass_line = not pass_line
            flag = False
        if pass_line or grids.loc[i, 'assigned'] >= 0:
            continue
        east_n, north_n = find_neighbor(i, 'east', grids), find_neighbor(i, 'north', grids)
        northeast_n = find_neighbor(north_n, 'east', grids)
        north_nx, northeast_nx = north_n, northeast_n
        step = 1
        if not flag:
            while (north_nx is None) and (northeast_nx is None):
                north_nx = find_neighbor(i + step, 'north', grids)
                northeast_nx = find_neighbor(north_nx, 'east', grids)
                step += 1
            if step % 2 == 1 and (((north_nx is None) ^ (northeast_nx is None)) or (
                    grids.loc[north_nx, 'assigned'] != grids.loc[northeast_nx, 'assigned'])):
                continue
        south_n = find_neighbor(i, 'south', grids)
        southeast_n = find_neighbor(south_n, 'east', grids)
        if east_n and south_n and southeast_n and grids.loc[east_n, 'assigned'] < 0 and grids.loc[
            south_n, 'assigned'] < 0 and grids.loc[southeast_n, 'assigned'] < 0:
            grids.loc[[i, east_n, south_n, southeast_n], 'assigned'] = group_id
            grids.loc[[i, east_n, south_n, southeast_n], 'position'] = ['tl', 'tr', 'bl', 'br']
            group_id += 1
            flag = True
        else:
            flag = False
    if drop:
        grids = grids[grids['assigned'] != -1].drop(columns='centroid')
    else:
        grids = grids.drop(columns='centroid')
    if save_path:
        grids.to_file(save_path, driver='ESRI Shapefile')
    return grids

def corp_margin(img):
    img2=img.sum(axis=2)
    row, col =img2.shape
    row_top=0
    raw_down=0
    col_top=0
    col_down=0
    for r in range(0,row):
        if img2.sum(axis=1)[r]<765*col/255:
            row_top=r
            break
    for r in range(row-1,0,-1):
        if img2.sum(axis=1)[r]<765*col/255:
            raw_down=r
            break
    for c in range(0,col):
        if img2.sum(axis=0)[c]<765*row/255:
            col_top=c
            break
    for c in range(col-1,0,-1):
        if img2.sum(axis=0)[c]<765*row/255:
            col_down=c
            break
    new_img=img[row_top:raw_down+1,col_top:col_down+1,:]
    return new_img

def img_stitch(img1, img2, axis):
    min_up, min_down, min_diff = 0, 0, np.inf
    if axis == 0:
        img2 = cv2.resize(img2, (img1.shape[1], img2.shape[0]*img1.shape[1]//img2.shape[1]))
        _img1, _img2 = img1.sum(axis=2), img2.sum(axis=2)
        up, down = _img1.shape[0]-1, 0
        while up >= _img1.shape[0]*7/8:
            down = 0
            while down <= _img2.shape[0]*1/8:
                diff = np.square(_img1[up-2:up, :]-_img2[down:down+2,:]).sum()
                if diff < min_diff:
                    min_up, min_down, min_diff = up, down, diff
                down += 1
            up -= 1
        return np.concatenate((img1[:min_up,:,:], img2[min_down:,:,:]))
    elif axis == 1:
        img2 = cv2.resize(img2, (img2.shape[1]*img1.shape[0]//img2.shape[0], img1.shape[0]))
        _img1, _img2 = img1.sum(axis=2), img2.sum(axis=2)
        up, down = _img1.shape[1]-1, 0
        while up >= _img1.shape[1]*7/8:
            down = 0
            while down <= _img2.shape[1]*1/8:
                diff = np.square(_img1[:,up-2:up]-_img2[:, down:down+2]).sum()
                if diff < min_diff:
                    min_up, min_down, min_diff = up, down, diff
                down += 1
            up -= 1
        return np.concatenate((img1[:,:min_up,:], img2[:,min_down:,:]), axis=1)


class prob_calculator():
    def __init__(self, channeles=3):
        self.model = None
        self.channeles = channeles
        self.class_idx = {0: 'Gridiron', 1: 'Linear', 2: 'Nopattern', 3: 'Organic', 4: 'Radial', 5: 'Tributary'}

    def load_model(self, model_path):
        #modelPath = model_path
        if self.channeles == 3:
            self.model = Build_model(config).build_model()
        elif self.channeles == 6:
            self.model = Build_model(config).build_mymodel()
        self.model.load_weights(model_path)

    def get_prob(self, img):
        if isinstance(img, str):
            img = deal_image(img)
        img = cv2.resize(img, (config.normal_size, config.normal_size))
        img = img_to_array(img) / 255
        img = img[np.newaxis, :]
        pred = self.model.predict(img)[0]
        return pred

    def get_grids_prob(self, grids, image_path, save_path=None):
        if not self.model:
            raise Exception('No model is loaded.')
        if isinstance(grids, str):
            grids = gpd.read_file(grids)
        if grids.index.name != 'id':
            if 'id' not in grids.columns:
                grids['id'] = grids['city_ascii'] + '_' + grids.index.map(str)
            grids.set_index('id', inplace=True)
        print('=' * 50, '\n', 'Start predicting road network pattern of grids...')
        grids[['Prob_G', 'Prob_L', 'Prob_N', 'Prob_O', 'Prob_R', 'Prob_T']] = None

        imgNameList = grids.index.to_list()
        # load images

        for i, imgName in enumerate(imgNameList):
            imgPath = imgName + '.png'
            file = os.path.join(image_path, imgPath)

            if not os.path.exists(file):  # if image not generated yet, plot it now
                print(f'trying to plot grid {imgName}')
                centroid = grids.loc[imgName].geometry.centroid
                point = (centroid.y, centroid.x)
                try:
                    PlotCRHD_grid(imgName, point, 1000, image_path)
                except:
                    #grids.loc[imgName, ['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']] = None
                    grids.loc[imgName, ['Prob_G', 'Prob_L', 'Prob_N', 'Prob_O', 'Prob_R', 'Prob_T']] = None
                    grids['cls'] = 'No pattern'
                    continue
            if self.channeles==3:
                try:
                    img = deal_image(file)
                except:
                    #grids.loc[imgName, ['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']] = None
                    grids.loc[imgName, ['Prob_G', 'Prob_L', 'Prob_N', 'Prob_O', 'Prob_R', 'Prob_T']] = None
                    grids['cls'] = 'No pattern'
                    continue
                img = cv2.resize(img, (config.normal_size, config.normal_size))
                img = img_to_array(img) / 255
                img = img[np.newaxis, :]
                pred = self.model.predict(img)[0]
            elif self.channeles==6:
                img1, img2 = deal_divide_image(file)
                img1 = cv2.resize(img1, (config.normal_size, config.normal_size))
                img1 = img_to_array(img1) / 255
                img1 = img1[np.newaxis, :]
                img2 = cv2.resize(img2, (config.normal_size, config.normal_size))
                img2 = img_to_array(img2) / 255
                img2 = img2[np.newaxis, :]
                pred = self.model.predict([img2, img1])[0]

            grids.loc[imgName, ['Prob_G', 'Prob_L', 'Prob_N', 'Prob_O', 'Prob_R', 'Prob_T']] = pred
            if i % 100 == 0:
                print(f'Current grid: {imgName}')

        grids[['Prob_G', 'Prob_L', 'Prob_N', 'Prob_O', 'Prob_R', 'Prob_T']] = grids[['Prob_G', 'Prob_L', 'Prob_N', 'Prob_O', 'Prob_R', 'Prob_T']].astype(float)
        cls = grids[grids.Prob_O.notna()].apply(lambda row: np.argmax([row.Prob_G, row.Prob_L, row.Prob_N, row.Prob_O, row.Prob_R, row.Prob_T]), axis=1)
        grids.loc[grids.Prob_O.notna(), 'cls'] = cls.map(self.class_idx)
        if save_path:
            grids.to_file(save_path,
                          driver='ESRI Shapefile',
                          encoding='utf-8')
        print('Completed!')
        return grids

    def get_larger_scale_prob(self, grids, image_path, save_path=None):
        if not self.model:
            raise Exception('No model is loaded.')
        if isinstance(grids, str):
            grids = gpd.read_file(grids)
        if 'assigned' not in grids.columns:
            print('Starting grouping...')
            grids = merge_grids(grids)
        if grids.index.name != 'id':
            if 'id' not in grids.columns:
                grids['id'] = grids['city_ascii'] + '_' + grids.index.map(str)
            grids.set_index('id', inplace=True)
        print('=' * 50, '\n', 'Start predicting road network pattern in larger scale...')
        grids['larger_scale_cls'] = None

        for group_id in grids.assigned.unique():
            group = grids[grids['assigned'] == group_id]
            img_path = os.path.join(image_path, group[group['position'] == 'tl'].index.values[0] + '.png')
            try:
                tl = deal_image(img_path)
            except:
                tl = np.ones([224,224,3])*255
            tl = tl[:tl.shape[0] * 3 // 4, :tl.shape[1] * 3 // 4, :]
            img_path = os.path.join(image_path, group[group['position'] == 'tr'].index.values[0] + '.png')
            try:
                tr = deal_image(img_path)
            except:
                tr = np.ones([224, 224, 3]) * 255
            tr = tr[:tr.shape[0] * 3 // 4, tr.shape[1] * 1 // 4:, :]
            img_path = os.path.join(image_path, group[group['position'] == 'bl'].index.values[0] + '.png')
            try:
                bl = deal_image(img_path)
            except:
                bl = np.ones([224, 224, 3]) * 255
            bl = bl[bl.shape[0] * 1 // 4:, :bl.shape[1] * 3 // 4:, :]
            img_path = os.path.join(image_path, group[group['position'] == 'br'].index.values[0] + '.png')
            try:
                br = deal_image(img_path)
            except:
                br = np.ones([224, 224, 3]) * 255
            br = br[br.shape[0] * 1 // 4:, br.shape[1] * 1 // 4:, :]
            left = img_stitch(tl, bl, axis=0)
            right = img_stitch(tr, br, axis=0)
            w, h = (left.shape[1] + right.shape[1]) // 2, (left.shape[0] + right.shape[0]) // 2
            left = cv2.resize(left, (w, h))
            right = cv2.resize(right, (w, h))
            concated = img_stitch(left, right, axis=1)
            pred = self.get_prob(concated)
            upper_cls = self.class_idx[np.argmax(pred)]
            grids.loc[grids.assigned == group_id, 'larger_scale_cls'] = upper_cls
        if save_path:
            grids.to_file(save_path,
                          driver='ESRI Shapefile',
                          encoding='utf-8')
        print('Larger scale classification completed!')
        return grids

    def get_smaller_scale_prob(self, grids, image_path, save_path=None):
        if not self.model:
            raise Exception('No model is loaded.')
        if isinstance(grids, str):
            grids = gpd.read_file(grids)
        if grids.index.name != 'id':
            if 'id' not in grids.columns:
                grids['id'] = grids['city_ascii'] + '_' + grids.index.map(str)
            grids.set_index('id', inplace=True)
        print('=' * 50, '\n', 'Start predicting road network pattern in smaller scale...')
        grids['smaller_scale_cls'] = None

        imgNameList = grids.index.to_list()
        for i, idx in enumerate(imgNameList):
            smaller_scale_cls = ''
            try:
                image = deal_image(os.path.join(image_path, idx + '.png'))
            except:
                continue
            sub_imgs = split_img(image)
            for sub_img in sub_imgs:
                try:
                    pred = self.get_prob(sub_img)
                    sub_img_cls = self.class_idx[np.argmax(pred)]
                    smaller_scale_cls += sub_img_cls + ','
                except:
                    smaller_scale_cls += 'Nopattern' + ','
            grids.loc[idx, 'smaller_scale_cls'] = smaller_scale_cls[:-1]

            if i % 100 == 0:
                print(f'Current grid: {idx}')

        if save_path:
            grids.to_file(save_path,
                          driver='ESRI Shapefile',
                          encoding='utf-8')
        print('Smaller scale classification completed!')
        return grids

    def get_multiscale_prob(self, grids, image_path, save_path=None, drop=True):
        grids = merge_grids(grids, drop=drop)
        grids = self.get_grids_prob(grids, image_path, save_path)
        grids = self.get_larger_scale_prob(grids, image_path, save_path)
        grids = self.get_smaller_scale_prob(grids, image_path, save_path)
        return grids


