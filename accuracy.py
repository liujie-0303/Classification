# -*- coding: utf-8 -*-
# Author: 超级禾欠水

# 导入相关库
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import skll
from classtools import *
import mglearn

#%%

# 数据目录
pic_folder = r'D:\image_process\S2_CCD_RF_Classification\map_process\Class_result\test_picture'
shp_folder = r'D:\image_process\S2_CCD_RF_Classification\S2_Data\valid_samples_WGS_84_UTM50N'
acc_folder = r'D:\image_process\S2_CCD_RF_Classification\map_process\Class_result\accuracy'
shp_files = glob.glob(os.path.join(shp_folder, '*.shp'))

# 分类结果文件
prediction_fn = os.path.join(pic_folder, 'RF_opt_new.tif')
out_matrix_fn = os.path.join(acc_folder, 'RF_opt_confusion_matrix_new.csv')
valid_fn = os.path.join(acc_folder, 'valid.csv')
out_report_csv = os.path.join(acc_folder, 'RF_opt_report_new.csv')

tif_ds = gdal.Open(prediction_fn)
valid_df = pd.read_csv(valid_fn)
class_type = ['裸地', '建筑', '农田', '林地', '工业用地', '潮滩', '稻田', '芦苇草地', '盐田', '水域']

# 获取图像坐标信息（xoff, yoff）
xys = np.array(valid_df[['xoff', 'yoff']])
# 提取类别标签数据
y = np.array(valid_df['class'], dtype=int)
del valid_df

cols, rows = zip(*xys)
del xys

data = tif_ds.GetRasterBand(1).ReadAsArray()
y_hat = data[rows, cols]

print('Kappa:', skll.kappa(y, y_hat))

labels = np.unique(np.concatenate((y, y_hat)))
matrix = confusion_matrix(y, y_hat, labels)
print("\nConfusion matrix:\n{}\n".format(matrix))
scores_image = mglearn.tools.heatmap(
    matrix, xlabel='Predicted label',
    ylabel='True label', xticklabels=labels, yticklabels=labels,
    cmap='viridis', fmt='%d')
plt.title("RF_opt_Confusion matrix")
plt.gca().invert_yaxis()

report = classification_report(y, y_hat, target_names=class_type, digits=2, output_dict=True)
print(report)

df = pd.DataFrame(report).transpose()
df.to_csv(out_report_csv, encoding='utf_8_sig')


matrix = np.insert(matrix, 0, labels, 0)
matrix = np.insert(matrix, 0, np.insert(labels, 0, 0), 1)
np.savetxt(out_matrix_fn, matrix, fmt='%1.0f', delimiter=',')

del tif_ds
print('finish!!!!!')
#plt.show()




