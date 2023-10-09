import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 
def lowlight(image_path):
	# 设置GPU0作为当前使用的GPU设备
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	
	# 读取图片
	data_lowlight = Image.open(image_path)
	# 归一化图片:480x640x3
	data_lowlight = (np.asarray(data_lowlight)/255.0)
	# 转换成张量:480x640x3
	data_lowlight = torch.from_numpy(data_lowlight).float()
	# 从HWD转换成CHW:3x480x640
	data_lowlight = data_lowlight.permute(2,0,1)
	# 转移到GPU上，并由CHW转换成BCHW:1x3x480x640
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	# 这里涉及了一个python的基本点
	# 将其他东西封装成类，使用库和类的方法调用
	DCE_net = model.enhance_net_nopool().cuda()
	# 这一步用到了pytorch载入权重文件安的两个函数
	# torch.load是将文件载入成变量
	# model.load_state_dict是将数据字典变量载入到模型中
	DCE_net.load_state_dict(torch.load('snapshots/Epoch195.pth'))

	# 这里简单的统计了一下时间
	start = time.time()
	# 这里是前向传播推理的全过程，得到了增强后的图像，注意是逐像素进行增强的。
	_,enhanced_image,_ = DCE_net(data_lowlight)
	end_time = (time.time() - start)
	print(end_time)

	# 这里是替换路径，是一个实用好用的函数呀。
	#######################################################
	# image_path = image_path.replace('test_data','result')
	#######################################################
	image_path = image_path.replace('wxldata','result')
	result_path = image_path
	# 这里是确保父文件夹必然存在
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# 用的是torchision中的图像保存函数。
	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
	#推理阶段，所以要先关闭梯度计算
	with torch.no_grad():
		# 得到测试集图片的所有图片路径
		##############################
		# filePath = 'data/test_data/'
		##############################
		filePath = 'data/wxldata/'
		file_list = os.listdir(filePath)
		# 读取
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				# image = image
				print(image)
				lowlight(image)

		

