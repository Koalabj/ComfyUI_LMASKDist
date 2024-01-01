import os
import json
import torch
import requests
import numpy as np
from PIL import Image,ImageDraw
from PIL.PngImagePlugin import PngInfo
from base64 import b64encode
from io import BytesIO
import cv2
import random
import math

def pil_to_tensor_grayscale(pil_image):
    # 将PIL图像转换为NumPy数组
    numpy_image = np.array(pil_image)

    # 归一化像素值
    numpy_image = numpy_image.astype(np.float32) / 255.0

    # 添加一个通道维度 [H, W] -> [1, H, W]
    numpy_image = np.expand_dims(numpy_image, axis=0)

    # 将NumPy数组转换为PyTorch张量
    tensor_image = torch.from_numpy(numpy_image)

    return tensor_image
#处理为灰度图
def im_read(face_mask):
	numpy_image=face_mask.cpu().numpy()  
	face_mask_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
	if face_mask_image.shape[0] == 3:
		face_mask_image = face_mask_image.transpose(1, 2, 0)
        # 转换为灰度图像
		face_mask_image = cv2.cvtColor(face_mask_image, cv2.COLOR_RGB2GRAY)
	if face_mask_image.shape[0] == 1:
		face_mask_image = face_mask_image.squeeze(0)
	_, threshold_image = cv2.threshold(face_mask_image, 128, 255, cv2.THRESH_BINARY)
	if len(threshold_image.shape) != 2:
		threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2GRAY)
	threshold_image = np.uint8(threshold_image)
	return threshold_image
    
def Direction_face_ZuoBiao(threshold):
    # 轮廓检测
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取每个轮廓的点坐标
    max_contour = max(contours, key=cv2.contourArea)
    contours_squeeze = max_contour.squeeze()

    for contour in contours:
        # 找到轮廓的上、左、右边界 (index) (641,305)
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0]) # (Y, X)
        buttonmost = tuple(contour[contour[:, :, 1].argmax()][0])

        # 计算蒙版图的中心点
        center_x = int((leftmost[0] + rightmost[0]) / 2)
        center_y = int((topmost[1] + buttonmost[1]) / 2)
        size=center_y-int(topmost[1])

    return contours_squeeze, center_x, center_y,size
def find_center_and_max_radius(mask):
    """
    从黑白蒙版图中找到白色区域的中心点和最大半径。

    :param mask: 黑白蒙版图，白色区域为感兴趣的区域
    :return: 中心点坐标 (x, y) 和最大半径
    """
    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最大半径和中心点
    max_radius = 0
    center_x, center_y = 0, 0

    # 遍历所有轮廓，找到最大半径的圆
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > max_radius:
            max_radius = radius
            center_x, center_y = x, y

    return int(center_x), int(center_y), int(max_radius)

def draw_irregular_shape_with_cv2(center_x, center_y, min_radius, max_radius, image_size):
    """
    使用OpenCV绘制不规则图形。

    :param center_x: 中心点的 X 坐标
    :param center_y: 中心点的 Y 坐标
    :param min_radius: 最小半径
    :param max_radius: 最大半径
    :param image_size: 图像尺寸 (宽度, 高度)
    :return: OpenCV 图像
    """
    # 创建一个黑色背景图像
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # 生成随机多边形的顶点
    num_points = random.randint(5, 10)  # 随机选择 5 到 10 个顶点
    points = []
    for _ in range(num_points):
        angle = random.uniform(0, 2 * math.pi)  # 随机角度
        r = random.uniform(min_radius, max_radius)  # 随机半径
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
        points.append([x, y])

    # 将点转换为适合OpenCV的格式
    pts = np.array([points], np.int32)
    pts = pts.reshape((-1, 1, 2))

    # 绘制多边形
    cv2.fillPoly(img, [pts], (255, 255, 255))

    return img

def Borad_draw(threshold, qualified_Zuobiao, horizon_num, vertical_num, center_x, center_y):
    # 如何区别上左右轮廓的点呢？ 计算到中心点坐标的xy坐标值      contours[y, x]
    for i in range(len(qualified_Zuobiao)):
        # 随机数生成
        # top_button_num = random.randint(1, vertical_num)
        # left_right_rand_num = random.randint(vertical_num, horizon_num)
        # 区分左右脸
        if qualified_Zuobiao[i][0] <= center_x and qualified_Zuobiao[i][1] <= center_y: # 左上脸
            # 更改坐标点值
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] - random.randint(150,300) # x
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] - random.randint(1,50) # y
        if qualified_Zuobiao[i][0] >= center_x and qualified_Zuobiao[i][1] <= center_y: # 右上脸
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] + random.randint(150,300)
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] - random.randint(1,50)
        if qualified_Zuobiao[i][0] <= center_x and qualified_Zuobiao[i][1] >= center_y:  # 左下脸
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] - random.randint(150,300) # x
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] + random.randint(1,50) # y
        if qualified_Zuobiao[i][0] >= center_x and qualified_Zuobiao[i][1] >= center_y:  # 右下脸
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] + random.randint(150,300)
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] + random.randint(1,50)

    points = np.array(qualified_Zuobiao)
    contours_a = np.array([points])
    # 根据坐标点画轮廓
    cv2.drawContours(threshold, contours_a, -1, (255, 255, 255), 2)

    # 轮廓检测
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取每个轮廓的点坐标
    max_contour = max(contours, key=cv2.contourArea)
    contours_squeeze = max_contour.squeeze()

    return threshold, contours_squeeze


def borad_pz(expanded_mask, contour):
    cv2.drawContours(expanded_mask, [contour], -1, (255), thickness=cv2.FILLED)
    return expanded_mask

def pil_to_tensor(source_image):
    image = Image.fromarray(np.clip(255. * source_image.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
    return image
class LoadImageUrl:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"face_mask": ("IMAGE",),
				"body_mask": ("IMAGE",)
			},
            "optional": {
                "image": ("IMAGE",)
            }
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "load_image_url"
	CATEGORY = "remote"

	def load_image_url(self, face_mask,body_mask):
		face_image=im_read(face_mask)
		center_x, center_y,max_size = find_center_and_max_radius(face_image)
		print(f'最小半径{max_size}')
		image_size = (face_image.shape[0], face_image.shape[1])  # 图像尺寸
		expanded_mask_copy = draw_irregular_shape_with_cv2(center_x, center_y, max_size+50, max_size+100, image_size)
		# 参数定义
		Horizon_num = 300 # 坐标点扩散距离
		Vertical_num = 400
		# expanded_mask, contours = Borad_draw(threshold_image, qualified_Zuobiao, Horizon_num, Vertical_num, center_x, center_y)
		# expanded_mask_copy = borad_pz(expanded_mask, contours)
		body=im_read(body_mask)
		width = body.shape[0]; height = body.shape[1]
		im1_copy = cv2.resize(expanded_mask_copy, (height, width))
		img_face_expect_body = cv2.multiply(im1_copy, body)
		result = cv2.cvtColor(img_face_expect_body, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(result)
		pil_image.save("/root/autodl-tmp/ComfyUI/output/test.png")
		torch_img=pil_to_tensor_grayscale(pil_image)
        # 转换为PyTorch张量
		return (torch_img,)






