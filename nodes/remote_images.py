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
		threshold_image=im_read(face_mask)
		qualified_Zuobiao, center_x, center_y = Direction_face_ZuoBiao(threshold_image)
		# 参数定义
		Horizon_num = 300 # 坐标点扩散距离
		Vertical_num = 200
		expanded_mask, contours = Borad_draw(threshold_image, qualified_Zuobiao, Horizon_num, Vertical_num, center_x, center_y)
		expanded_mask_copy = borad_pz(expanded_mask, contours)
		body=im_read(body_mask)
		width = body.shape[0]; height = body.shape[1]
		im1_copy = cv2.resize(expanded_mask_copy, (height, width))
		img_face_expect_body = cv2.multiply(im1_copy, body)
		img_face_expect_body = cv2.cvtColor(img_face_expect_body, cv2.COLOR_BGR2RGB)
        
		return (torch.cat(img_face_expect_body),)






