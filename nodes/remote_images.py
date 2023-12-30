import os
import json
import torch
import requests
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from base64 import b64encode
from io import BytesIO
import cv2
import random

# def Direction_face_ZuoBiao(threshold, dis):
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

    return contours_squeeze, center_x, center_y
def Borad_draw(threshold, qualified_Zuobiao, horizon_num, vertical_num, center_x, center_y):
    # 如何区别上左右轮廓的点呢？ 计算到中心点坐标的xy坐标值      contours[y, x]
    for i in range(len(qualified_Zuobiao)):
        # 随机数生成
        top_button_num = random.randint(1, vertical_num)
        left_right_rand_num = random.randint(vertical_num, horizon_num)
        # 区分左右脸
        if qualified_Zuobiao[i][0] <= center_x and qualified_Zuobiao[i][1] <= center_y: # 左上脸
            # 更改坐标点值
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] - left_right_rand_num # x
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] - top_button_num # y
        if qualified_Zuobiao[i][0] >= center_x and qualified_Zuobiao[i][1] <= center_y: # 右上脸
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] + left_right_rand_num
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] - top_button_num
        if qualified_Zuobiao[i][0] <= center_x and qualified_Zuobiao[i][1] >= center_y:  # 左下脸
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] - left_right_rand_num # x
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] + top_button_num # y
        if qualified_Zuobiao[i][0] >= center_x and qualified_Zuobiao[i][1] >= center_y:  # 右下脸
            qualified_Zuobiao[i][0] = qualified_Zuobiao[i][0] + left_right_rand_num
            qualified_Zuobiao[i][1] = qualified_Zuobiao[i][1] + top_button_num

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

def Borad_PengZhang(expanded_mask, contour):
    # 遍历图像的每个像素
    for y in range(expanded_mask.shape[0]):
        for x in range(expanded_mask.shape[1]):
            point = (x, y)
            is_inside = cv2.pointPolygonTest(contour, point, measureDist=False)
            # 在轮廓内部
            if is_inside == 1:
                expanded_mask[y, x] = 255
                print('(' + str(x) + ',' + str(y) + '),是轮廓中的点')
    print('填充完毕')

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
		image = np.asarray(face_mask)
		img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		expanded_mask = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)
		# expanded_mask = cv2.imdecode(face_mask,cv2.IMREAD_GRAYSCALE)
		# _, threshold = cv2.threshold(expanded_mask, 128, 255, cv2.THRESH_BINARY)
		expanded_mask = cv2.convertScaleAbs(expanded_mask)
		_, threshold = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		qualified_Zuobiao, center_x, center_y = Direction_face_ZuoBiao(threshold)
		# 参数定义
		Horizon_num = 300 # 坐标点扩散距离
		Vertical_num = 200
		expanded_mask, contours = Borad_draw(threshold, qualified_Zuobiao, Horizon_num, Vertical_num, center_x, center_y)
		expanded_mask_copy = Borad_PengZhang(expanded_mask, contours)
		bodymask=cv2.imdecode(body_mask,cv2.IMREAD_GRAYSCALE)
		body=cv2.threshold(bodymask, 128, 255, cv2.THRESH_BINARY)
		width = body.shape[0]; height = body.shape[1]
		im1_copy = cv2.resize(expanded_mask_copy, (height, width))
		img_face_expect_body = cv2.multiply(im1_copy, body)
		return (img_face_expect_body,)






