import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):
	# 载入自制数据集的第一步是得到所有的图片路径，并以列表形式返回
	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	train_list = image_list_lowlight
	# 随机打乱列表顺序
	random.shuffle(train_list)
	# 返回乱序的列表，使用时可以通过索引来取值
	return train_list

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):
		# 载入数据集的第一步，是通过数据集路径，得到数据集的乱序列表
		self.train_list = populate_train_list(lowlight_images_path) 
		self.size = 256
		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))

	# 带有__ __的函数，是python中的魔法函数，此处比较特殊，getitem是train_loader被调用时同步被调用的
	def __getitem__(self, index):
		# 这个索引是train_loader传递过来的
		data_lowlight_path = self.data_list[index]
		# 读取了图片
		data_lowlight = Image.open(data_lowlight_path)
		# 将图像都缩放到256*256并且会抗锯齿
		# 是否是因为这里的缩放，导致了测试集图像的模糊？
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		# 将图片转换成numpy数组，并且将其归一化
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		# 将numpy数组转换成tensor
		data_lowlight = torch.from_numpy(data_lowlight).float()
		# 将tensor的维度转换成[3,256,256]
		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

