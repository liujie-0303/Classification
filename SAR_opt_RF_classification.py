# 1. 基本库
import os
import glob
import numpy as np
from osgeo import gdal, ogr, osr
import pandas as pd
import matplotlib.pyplot as plt
from classtools import *
import cv2
# 2. 机器学习相关库
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error
import mglearn
import skll

gdal.UseExceptions()    # 绑定python与gdal库的错误报告连接

# 1.需要准备好数据集，影像集，矢量数据（.shp）
# 2.需要利用矢量数据提取栅格影像的训练样本区栅格点（训练集），转换为csv文件
# 3.需要一个展示分类后图像的函数

# 利用和栅格影像范围对应从.shp文件提取出特征转换为.csv文件保存下来
# 有两种shp文件，一种是单一的多边形数据集，每类为一个shp文件，以文件名区分类别
# 另一种是多个shp文件合并为一个shp文件，属性表的字段属性值区分
# shp文件属性表需要具有编码值（整数）
# 需要有一个检查shp和栅格影像投影的参数
# 似乎只需要栅格影像的地理变换


def read_img(file_name):
    dataset = gdal.Open(file_name)                          # 打开文件
    im_width = dataset.RasterXSize                          # 栅格矩阵的列数
    im_height = dataset.RasterYSize                         # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()                 # 仿射矩阵
    im_proj = dataset.GetProjection()                       # 地图投影信息
    band = dataset.GetRasterBand(1)
    im_array = band.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    return dataset

def read_csv(csv_file,shp_files):
    land_cover_names = []
    for shp_file in shp_files:
        basename = os.path.basename(shp_file)                   # 返回文件名

        # split（'_'）函数将带有 '_' 的字符串分割成列表数据；然后 '_'.join（）函数将列表中的元素用'_'连接起来;
        land_cover_name = '_'.join(basename.split('_')[:-1])    # 选取列表中除去最后一个元素的全部元素；
        land_cover_names.append(land_cover_name)                # 生成一个分类名列表

    # 读取成dataFrame格式（将其看成series 组成的字典）；DataFrame的行索引是index，列索引是columns
    train_data = pd.read_csv(csv_file)
    xys = np.array(train_data[['xoff', 'yoff']])                # 获取图像坐标信息（xoff, yoff）
    classes = np.array(train_data['class'], dtype=int)          # 提取类别标签数据
    del train_data                                              # 关闭csv文件
    cols, rows = zip(*xys)                                      # zip()函数： 将列表中对应的元素打包成一个个元组
    del xys
    return land_cover_names,classes, rows, cols

def extract_sample_DN(opt_files, sar_files, rows, cols):
    opt_samples = []
    for opt_file in opt_files:
        opt_ds = gdal.Open(opt_file)
        bands = opt_ds.RasterCount
        opt_sample = []
        for i in range(1, bands+1, 1):
            band = opt_ds.GetRasterBand(i)
            band_min, band_max = band.ComputeRasterMinMax()
            band_array = band.ReadAsArray()  # ReadAsArray(始列， 始行，列间隔，行间隔)
            data = band_array[rows, cols].flatten()
            data = (data - band_min) * 255 / (band_max - band_min)
            opt_sample.append(data)
        del opt_ds
        opt_list = np.vstack(opt_sample)
        opt_samples.append(opt_list)

    sar_samples = []
    for sar_file in sar_files:
        sar_ds = gdal.Open(sar_file)
        # 维度为：波段数, 行数, 列数
        sar_float32 = sar_ds.ReadAsArray()
        sar_array = (sar_float32 * 255).astype(np.uint8)
        sar_bandcount = sar_ds.RasterCount
        # 将三维数据转化为二维数据并输出
        sar_sample = sar_array[:, rows, cols].reshape(sar_bandcount, -1)  # ( (rows * cols), 3)
        sar_samples.append(sar_sample)
        del sar_ds
    samples = opt_samples + sar_samples
    samples = np.vstack(tuple(samples)).T  # 水平叠加延展 （cols* rows, 3*file_n）
    samples = samples.astype(np.uint8)

    print('smaples data is :', samples)
    print('samples shape is:', samples.shape)
    test_nan = np.sum(np.isnan(samples), axis=0)
    print('test_nan_sum: ', test_nan.sum())
    return samples

def data_generator(opt_files, sar_files, batchs):
    opt_ds = gdal.Open(opt_files[0])
    rows = opt_ds.RasterYSize
    cols = opt_ds.RasterXSize
    opt_bands = opt_ds.RasterCount
    sar_ds = gdal.Open(sar_files[0])
    sar_bands = sar_ds.RasterCount

    y_step = rows // batchs          # “ // ” 表示整数除法，返回不大于结果的一个最大整数
    y_residual = rows - (y_step * batchs)
    del opt_ds, sar_ds

    for i in range(0, batchs):
        if i != batchs-1:
            datas_opt = []
            for opt_file in opt_files:
                ds = gdal.Open(opt_file)
                bands = ds.RasterCount
                data_opt = []
                for j in range(1, bands+1, 1):
                    band = ds.GetRasterBand(j)
                    band_min, band_max= band.ComputeRasterMinMax()
                    opt_array = band.ReadAsArray(0, i * y_step, cols, y_step)  # ReadAsArray(始列， 始行，列间隔，行间隔)
                    data = (opt_array - band_min) * 255 / (band_max - band_min)
                    data = data.flatten()
                    data_opt.append(data)
                del ds
                opt_list = np.vstack(data_opt)
                datas_opt.append(opt_list)

            data_sar = []
            for sar_file in sar_files:
                ds = gdal.Open(sar_file)
                data_float32 = ds.ReadAsArray(0, i * y_step, cols, y_step)
                data = (data_float32 * 255).astype(np.uint8)
                del ds
                data = data.reshape(sar_bands, -1)
                data_sar.append(data)
            datas = datas_opt + data_sar
            datas = np.vstack(datas)
        else:
            datas_opt = []
            for opt_file in opt_files:
                ds = gdal.Open(opt_file)
                bands = ds.RasterCount
                data_opt = []
                for j in range(1, bands+1, 1):
                    band = ds.GetRasterBand(j)
                    band_min, band_max = band.ComputeRasterMinMax()
                    data = band.ReadAsArray(0, i * y_step, cols, y_step + y_residual)  # ReadAsArray(始列， 始行，列间隔，行间隔)
                    data = (data - band_min) * 255 / (band_max - band_min)
                    data = data.flatten()
                    data_opt.append(data)
                del ds
                opt_list = np.vstack(data_opt)
                datas_opt.append(opt_list)
            data_sar = []
            for sar_file in sar_files:
                ds = gdal.Open(sar_file)
                data_float32 = ds.ReadAsArray(0, i * y_step, cols, y_step + y_residual)
                data = (data_float32 * 255).astype(np.uint8)
                del ds
                data = data.reshape(sar_bands, -1)
                data_sar.append(data)
            datas = datas_opt + data_sar
            datas = np.vstack(datas)
        yield datas.T.astype(np.uint8)

def RF_model(X, y):

    RF_clf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=100,
                                           min_samples_leaf=100, n_jobs=-1)
    # 再度分割数据集, 取1/3数据用作训练集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.66, random_state=42)
    # 拟合模型
    RF_clf.fit(X_train, y_train)

# 做预测
    y_pred = RF_clf.predict(X_test)
    print("Accuracy: {0:.3f}\n".format(accuracy_score(y_test, y_pred)))
    print('Kappa:', skll.kappa(y_test, y_pred))
    print("\nConfusion matrix:\n{}\n".format(confusion_matrix(y_test, y_pred)))

    labels = np.unique(np.concatenate((y_test, y_pred)))
    scores_image = mglearn.tools.heatmap(
        confusion_matrix(y_test, y_pred), xlabel='Predicted label',
        ylabel='True label', xticklabels=labels, yticklabels=labels,
        cmap='viridis', fmt='%d')
    plt.title("Confusion matrix")
    plt.gca().invert_yaxis()

    print(classification_report(y_test, y_pred))

    print("The training of the model ends!")
    return  RF_clf

# 利用生成器处理大影像数据集, 按行分割数据块进行预测避免内存不足

def RF_predict(RF_model,batchs):
    y_preds = []
    for data in data_generator(opt_files, sar_files, batchs):
        X = data
        y_pred = RF_model.predict(X)    # 预测结果 y_pred 输出为为行向量
        y_pred[np.sum(X, 1) == 0] = 0    # 按列相加 -> 各波段都没有数据的像素设置为0
        y_preds.append(y_pred)
    X = np.array(X)
    print('X is :{}\nX_shape is :{}'.format(X, X.shape))
    pred = np.hstack(tuple(y_preds))
    print('y = :{}\ny_shape = :{}'.format(pred, pred.shape))
    return pred

# 创建栅格文件，并建立金字塔
def write_img(y_pred, out_fn, in_ds):
    x_size = in_ds.RasterXSize                          
    y_size = in_ds.RasterYSize
    y_array = y_pred.reshape((y_size, x_size))
    predict_ds = make_raster(in_ds, out_fn, y_array, gdal.GDT_Byte, 0)
    predict_ds.FlushCache()
    levels = compute_overview_levels(predict_ds.GetRasterBand(1))
    predict_ds.BuildOverviews('NEAREST', levels)
    del in_ds, predict_ds


###################################################
# 数据目录
sar_input_path = r'D:\image_process\S2_CCD_RF_Classification\S2_Data\data_test\ccd'
opt_input_path = r'D:\image_process\S2_CCD_RF_Classification\S2_Data\data_test\opt'
train_shps_path = r'D:\image_process\S2_CCD_RF_Classification\S2_Data\train_samples_WGS_84_UTM50N'
train_csv_path = r'D:\image_process\S2_CCD_RF_Classification\S2_Data'

# 获取取shp, tiff文件
opt_files = glob.glob(os.path.join(opt_input_path, '*.dat'))
sar_files = glob.glob(os.path.join(sar_input_path, '*.dat'))
shp_files = glob.glob(os.path.join(train_shps_path, '*.shp'))
csv_file = os.path.join(train_csv_path, 'train.csv')

# 输出文件名
out_fn = r'D:\image_process\S2_CCD_RF_Classification\map_process\Class_result\test_picture\RF_opt_SAR_new.tif'
out_model = r'D:\image_process\S2_CCD_RF_Classification\map_process\Class_result\train_model\RF_model.m'
# 读取影像文件，获取基本信息
in_img = opt_files[0]
in_ds= read_img(in_img)

# 1. 获取样本坐标、分类标签值
land_cover_names,classes, rows, cols = read_csv(csv_file,shp_files)

# 2. 提取训练样本数据
X = extract_sample_DN(opt_files, sar_files, rows, cols)
y = classes

# 3. 训练 RF 分类器
print("Start to train the model!!!")
RF_clf = RF_model(X, y)

# svm_clf = joblib.load(out_model)      #从本地加载训练好的模型
#joblib.dump(svm_clf, out_model)         # 保存模型到本地
# 4. 利用分类模型进行图像分类
print("Start to Predict!!! ")
y_pred = RF_predict(RF_clf, batchs=10)


# 5. 将分类结果写入图像
write_img(y_pred, out_fn, in_ds)
print("All jobs end!")



