import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		number_f = 32
		# 卷积核最最基本的了，似乎这一次记住了，输入通道数，输出通道数，卷积核大小，步长，拓延的宽度，是否使用偏置
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 
		# 最大池化层是保留池化窗口的最大值作为特征图的值，且因步长等于池化窗口的大小，而不会重叠
		# 最大池化层的作用是使特征图的尺度减小，且保留小可能重要的特征
		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		# 通过上采样层，再将池化后的图像恢复到原本的尺寸。
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
		
	def forward(self, x):
		# 经过简单的卷积和激活函数
		# x = 8x3x480x640
		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))
		# 这里就是残差块的思想了，对应的输入输出的尺寸变化也能理解
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		# 输出的部分用的是tanh激活函数
		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		# 这里有dim = 1 ，说明就是从纬度1开始分割的，纬度1指的是特征图的深度
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
		# ri = 1x3x480x640,对应着和原图片相同大小的张量

		# 论文中提到的二次函数，用于将图片进行LE映射，这一步是逐像素的。
		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		# 在第四步LE映射的时候，取得了一个中间值
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		# 在第八步LE映射的时候，取得了一个最终值
		enhance_image = x + r8*(torch.pow(x,2)-x)
		# 这里的1不是一个实体张量，只是说明前面list中的张量需要按照纬度1进行合并，本质上又是得到了x_r。
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		return enhance_image_1,enhance_image,r