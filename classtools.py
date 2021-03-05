
#本文件包含封装起来的常用于遥感图像地物分类的的一些处理函数

import math
from osgeo import gdal, ogr, osr
import numpy as np
import pandas as pd

# 点的包容性公式，光栅投影算法
def point_in_poly(x0, y0, poly):

    # 判断点是否是顶点
    if (x0, y0) in poly:
        return True

    # 判断点是否在边框上(多边形为一个环，首尾点相同)
    for i in range(len(poly)):
        # p1, p2分别为直线的两个端点
        p1 = None
        p2 = None

        if i == 0:
            p1 = poly[0]
            p2 = poly[1]
        else:
            p1 = poly[i-1]
            p2 = poly[i]
        # 点在直线上，且不是直线的两个点
        # 垂直于x轴的情况
        if x0 == p1[0] and x0 == p2[0]:
            return True
        # 点在直线上
        elif math.isclose((x0-p2[0])*(y0-p1[1]),(x0 - p1[0])*(y0 - p2[1]), rel_tol=1e-5) and y0 > min(p1[1],
            p2[1]) and y0 < max(p1[1], p2[1]):
            return True

    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y0 > min(p1y, p2y):
            if y0 <= max(p1y, p2y):
                if x0 <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y0 - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x0 <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    if inside:
        return True
    else:
        return False


def shp_file_to_csv(shp_ds: ogr.DataSource, raster_ds: gdal.Dataset) -> list:
    """
    return a list contain the tuple (x, y, class) of the point contain of polygon shp_ds.
    (x, y) is the coords of the shp_ds's spatial reference
    (the same as the raster_ds's spatial reference),
    the first element represent the x value,
    the second element represent the y value,
    the third element represent the class value.
    The class value is the code of one land cover.

    Parameters:
    --------------
    shp_ds      - shape file(GeometryType is Polygon) Data source
    raster_ds   - raster file Dataset

    Returns:
    -------------
    train_data_coords       - The train data's list contains the  item (x,y, class)

    """

    # 创建一个shp
    train_data_coords = []

    # 获取首个图层
    poly_lyr: ogr.Layer = shp_ds.GetLayer(0)
    # 获取shp文件的坐标系
    shp_osr: osr.SpatialReference = poly_lyr.GetSpatialRef()
    shp_osr.GetAttrValue('AUTHORITY', 1)                # ??? 获取对象属性值

    # 获取Gtiff文件坐标系
    raster_osr = osr.SpatialReference()
    raster_osr.ImportFromWkt(raster_ds.GetProjection())

    # 获取Gtiff文件地理变换
    gtiff_geotrans = raster_ds.GetGeoTransform()

    if raster_osr.GetAttrValue('AUTHORITY', 1) != shp_osr.GetAttrValue('AUTHORITY', 1):
        print('Error: The shape file and the raster file have the differnet spatial refer')
        return train_data_coords

    inv_gt = gdal.InvGeoTransform(gtiff_geotrans)

    # 获取要素的数量
    feat_count = poly_lyr.GetFeatureCount()
    # 保存训练集在栅格上的坐标（X,Y）以及地理编码值（class），需要三列

    # 遍历图层获取每一要素
    for feat_i in range(feat_count):
        # 获取要素
        poly_feat: ogr.Feature = poly_lyr.GetFeature(feat_i)

        # 提取类别编码值
        if 'class' not in poly_feat.keys():
            print("Error: The shape file don't have the 'class' Field")
            break
        name = poly_feat.GetField('class')
        # 从要素中获取多边形几何（是一个环）
        poly_geom: ogr.Geometry = poly_feat.geometry()
        if poly_geom.GetGeometryName() != 'POLYGON':
            print("Error: The geometry type of shape file isn't the polygon.")
            break
        for ring_i in range(poly_geom.GetGeometryCount()):
            # 获取多边形几何的第i个环
            ring: ogr.Geometry = poly_geom.GetGeometryRef(ring_i)

            # 获取几何多边形的边界(西东南北)
            left, right, lower, upper = ring.GetEnvelope()
            points = ring.GetPoints()
            # 判断点在多边形上
            # int OGRPolygon::PointOnSurface(OGRPoint * poPoint) const [virtual]
            for px in np.arange(left, right, gtiff_geotrans[1]):
                for py in np.arange(upper, lower, gtiff_geotrans[5]):
                    # 创建一个点
                    if point_in_poly(px, py, points):
                        offsets = gdal.ApplyGeoTransform(inv_gt, px, py)
                        # 转换为像素坐标（整数值）
                        xoff, yoff = map(int, offsets)
                        train_data_coords.append((xoff, yoff, px, py, name))
                        train_data_coords.append((px, py, name))
    return train_data_coords



def shp_files_to_csv(shp_files: list, raster_ds: gdal.Dataset, out_csv_file) -> list:
    """
    return a list contain the tuple (x, y, class) of the point contain of polygon shp_ds.
    (x, y) is the coords of the shp_ds's spatial reference
    (the same as the raster_ds's spatial reference),
    the first element represent the x value,
    the second element represent the y value,
    the third element represent the class value.
    The class value is the code of one land cover.

    Parameters:
    --------------
    shp_files   - list of shape file(GeometryType is Polygon) Data source
    raster_ds   - raster file Dataset

    Returns:
    -------------
    train_data_coords       - The train data's list contains the  item (x,y, class)

    """

    # 创建一个shp
    train_data_coords = []

    for i in range(len(shp_files)):
        code = i+1
        shp_ds = ogr.Open(shp_files[i])
        print(shp_files[i])
        # 获取首个图层
        poly_lyr: ogr.Layer = shp_ds.GetLayer(0)
        # 获取shp文件的坐标系
        shp_osr: osr.SpatialReference = poly_lyr.GetSpatialRef()
        shp_osr.GetAttrValue('AUTHORITY', 1)

        # 获取Gtiff文件坐标系
        raster_osr = osr.SpatialReference()         # 获取地理参考系统
        raster_osr.ImportFromWkt(raster_ds.GetProjection())     # 从一个WKT定义的坐标系统来构造一个SpatialReference类对象

        # 获取Gtiff文件地理变换
        gtiff_geotrans = raster_ds.GetGeoTransform()

        if raster_osr.GetAttrValue('AUTHORITY', 1) != shp_osr.GetAttrValue('AUTHORITY', 1):
            print('Error: The shape file and the raster file have the differnet spatial refer')
            return train_data_coords

        inv_gt = gdal.InvGeoTransform(gtiff_geotrans)

        # 获取要素的数量
        feat_count = poly_lyr.GetFeatureCount()
        # 保存训练集在栅格上的坐标（X,Y）以及地理编码值（class），需要三列

        # 遍历图层获取每一要素
        for feat_i in range(feat_count):
            # 获取要素
            poly_feat: ogr.Feature = poly_lyr.GetFeature(feat_i)

            # 没有编码值
            # 从要素中获取多边形几何（是一个环）
            poly_geom: ogr.Geometry = poly_feat.geometry()
            if poly_geom.GetGeometryName() != 'POLYGON':
                print("Error: The geometry type of shape file isn't the polygon.")
                break
            for ring_i in range(poly_geom.GetGeometryCount()):
                # 获取多边形几何的第i个环
                ring: ogr.Geometry = poly_geom.GetGeometryRef(ring_i)

                # 获取几何多边形的边界(西东南北)
                left, right, lower, upper = ring.GetEnvelope()
                points = ring.GetPoints()
                # 判断点在多边形上
                # int OGRPolygon::PointOnSurface(OGRPoint * poPoint) const [virtual]
                for px in np.arange(left, right, gtiff_geotrans[1]):
                    for py in np.arange(upper, lower, gtiff_geotrans[5]):
                        # 创建一个点
                        if point_in_poly(px, py, points):
                            offsets = gdal.ApplyGeoTransform(inv_gt, px, py)
                            # 转换为像素坐标（整数值）
                            xoff, yoff = map(int, offsets)
                            train_data_coords.append((xoff, yoff, px, py, code))
                            #train_data_coords.append((px, py, code))
        df = pd.DataFrame(train_data_coords,
                                       columns=['xoff', 'yoff', 'px', 'py', 'class'])
        df.to_csv(out_csv_file)
        del shp_ds
    print('train.csv have successfully write in.')
    return train_data_coords




# 利用gdal库创建一个单波段GeoTiff格式的栅格文件
# 需要一个一个数据源ds,保存的文件名路径fn，numpy二维数组
# 以及保存的数据类型data_type
def make_raster(in_ds: gdal.Dataset, fn: str, data: np.ndarray, data_type: object, Nodata=None) -> gdal.Dataset:
    """Create a one-band GeoTIFF.

    Parameters:
    ------------
    in_ds       - datasource to copy projection and geotransfrom from
    fn          - path to the file to create
    data        - NUmpy array containing data to write
    data_type   - output data type
    nodata      - optional NoData value

    Returns:
    ------------
    out_ds      - datasource to output
    """
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        fn, in_ds.RasterXSize, in_ds.RasterYSize, 1, data_type)
    out_ds: gdal.Dataset
    # 从输入数据源中复制投影（坐标系信息）
    out_ds.SetProjection(in_ds.GetProjection())
    # 从输入数据源中复制地理变换
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    out_band: gdal.Band
    if Nodata is not None:
        out_band.SetNoDataValue(Nodata)
    out_band.WriteArray(data)
    out_band.FlushCache()
    out_band.ComputeStatistics(False)
    return out_ds



# 将多个栅格文件(通常是.tif文件)打开并叠加起来，返回一个三维数组（ndarray）
def stack_bands(filenames: list) -> np.ndarray:
    """
    Returns a 3D array containing all band data from all files.

    Parameters:
    -------------
    filenames  - file_name list(str list)

    Returns:
    ------------
      - numpy.ndarray of all the bands of the input files
    """
    bands = []
    for fn in filenames:
        try:
            ds: gdal.Dataset = gdal.Open(fn)
            for i in range(1, ds.RasterCount + 1):
                bands.append(ds.GetRasterBand(i).ReadAsArray())
        except:
            print('Could not Open the file for ' + fn)
            print(gdal.GetLastErrorMsg())
    return np.dstack(bands)


def compute_overview_levels(band: gdal.Band) -> list:
    """Return an appropriate list of overview levels.

    Parameters:
    --------------
    band  - a raster file band

    Returns:
    -------------
    overviews   - an appropriate list of overview levels
    """
    max_dim = max(band.XSize, band.YSize)
    overviews = []
    level = 1
    while max_dim > 256:
        level *= 2
        overviews.append(level)
        max_dim /= 2
    return overviews


