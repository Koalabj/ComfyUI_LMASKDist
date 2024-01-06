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
from ultralytics import YOLO
from torchvision.transforms import ToPILImage
from torchvision import transforms

def create_smooth_bezier_polygon(mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_mask = np.zeros_like(mask)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        original_radius = max(w, h) / 2

        # 计算新的中心点
        center_x = x + w // 2
        vertical_shift = int(original_radius * random.uniform(0.5, 1))
        center_y = y + h // 2 + vertical_shift

        # 设置圆的半径
        radius = int(original_radius * random.uniform(1.5,2))

        # 调整圆的位置和大小，确保它在图像范围内
        center_x = max(radius, min(mask.shape[1] - radius, center_x))
        center_y = max(radius, min(mask.shape[0] - radius, center_y))
        radius = min(radius, center_x, mask.shape[1] - center_x, center_y, mask.shape[0] - center_y)

        # 绘制圆形
        cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)

    return circle_mask

def create_mask_from_contours(mask1, mask2):
    """
    创建一个新的蒙版，保留mask1中不在mask2的白色轮廓范围内的部分。

    :param mask1: 第一个黑白蒙版图
    :param mask2: 第二个黑白蒙版图
    :return: 新的蒙版图
    """
    # 在mask2中找到白色轮廓
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与mask1大小相同的全黑图像
    new_mask = np.zeros_like(mask1)

    # 将mask2的轮廓填充为白色
    cv2.drawContours(new_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # 保留mask1中不在mask2轮廓范围内的部分
    result_mask = cv2.bitwise_and(mask1, cv2.bitwise_not(new_mask))

    return result_mask
def blacken_below_y(image, y_coord):
    """
    将图像中y坐标以下的所有非黑色部分变为黑色。

    :param image: 输入的黑白蒙版图，应为灰度图像
    :param y_coord: 给定的y坐标
    :return: 修改后的图像
    """
    # 确保y坐标在图像高度范围内
    y_coord = max(0, min(y_coord, image.shape[0] - 1))

    # 将y坐标以下的区域变为黑色
    image[y_coord:, :] = 0

    return image

def blacken_above_y(image, y_coord):
    # 确保y坐标在图像高度范围内
		y_coord = max(0, min(y_coord, image.shape[0]))

    # 将y坐标以上的区域变为黑色
		image[:y_coord, :] = 0

		return image
def tensor_to_pil(img_tensor, batch_index=0):
    # Convert tensor of shape [batch_size, channels, height, width] at the batch_index to PIL Image
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img
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
def getMaskBootm(threshold):
	contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		buttonmost = tuple(contour[contour[:, :, 1].argmax()][0])
	return  buttonmost[1]

def getMaskTop(threshold):
    # 轮廓检测
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取每个轮廓的点坐标
    # max_contour = max(contours, key=cv2.contourArea)
    # contours_squeeze = max_contour.squeeze()

    for contour in contours:
        # 找到轮廓的上、左、右边界 (index) (641,305)
        # leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        # rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0]) # (Y, X)
        # buttonmost = tuple(contour[contour[:, :, 1].argmax()][0])

        # 计算蒙版图的中心点
        # center_x = int((leftmost[0] + rightmost[0]) / 2)
        # center_y = int((topmost[1] + buttonmost[1]) / 2)
        # size=center_y-int(topmost[1])

    return  topmost[1]
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
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        radius = random.uniform(min_radius, max_radius)
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
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
		print("开始加载")
		face_image=im_read(face_mask)
		print("脸部加载")
		dilated_mask = create_smooth_bezier_polygon(face_image)

		print("随机完成")
		body=im_read(body_mask)
		# 反色
		body = cv2.bitwise_not(body)

		print("身体加载完成")          
		if dilated_mask.shape[:2] != body.shape[:2]:
			body=cv2.resize(body, (dilated_mask.shape[1], dilated_mask.shape[0]))
		if len(dilated_mask.shape) == 2:
			dilated_mask = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR)
		if len(body.shape) == 2:
			body = cv2.cvtColor(body, cv2.COLOR_GRAY2BGR)

		img_face_expect_body = cv2.multiply(dilated_mask, body)
		print("替换完成")
		result = cv2.cvtColor(img_face_expect_body, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(result)
		torch_img=pil_to_tensor_grayscale(pil_image)
        # 转换为PyTorch张量
		return (torch_img,)
class BodyMask:
	def __init__(self):
		pass
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"image": ("IMAGE",),
				"body_mask": ("IMAGE",),
				"person_mask":("IMAGE",)
			},
            "optional": {
                "image": ("IMAGE",)
            }
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "BodyMaskMake"
	CATEGORY = "remote"
	def BodyMaskMake(self,image,body_mask,person_mask):
		body=im_read(body_mask)
		person_img_mask=im_read(person_mask)
		top=getMaskTop(body)
		print(f"顶部坐标{top}")

		hight=body.shape[0]
		print(f"图片高度{hight}")
		bootm=int((hight-top)*2/10+top)
		print(f"底部坐标{bootm}")
		body20=blacken_below_y(body,bootm)

        # 保存图片
		pic=tensor_to_pil(image)
		path="/root/autodl-tmp/ComfyUI/input/yt.png"
		pic.save(path)

		original_img = cv2.imread(path)
		gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
		_, img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

		# model = YOLO(task='detect',model='/root/autodl-tmp/ComfyUI/models/ultralytics/segm/person_yolov8n-seg.pt')
		# # model.eval()
		# results = model(source=path, mode='val')
		# # # 测试 将结果解析为pil图片
		# #  # View results
		# for r in results:
		# 	person_zuobiao_masks = r.masks
		# person_img_mask = np.ones_like(img)
		# if person_zuobiao_masks != None:
		# 	person_zuobiao = person_zuobiao_masks.xy
		# 	for item in person_zuobiao:
		# 		points = np.array(item, dtype=np.int32)
        #    		# 将轮廓列表转换为多维数组格式
		# 		contours_a = np.array([points])
		# 	cv2.fillPoly(person_img_mask, contours_a, (255, 255, 255))
		# 	print('人体填充完毕')
		# else:
		# 	print('未检测到物体，固未填充')
		
		model = YOLO(task='segment',model='/root/autodl-tmp/ComfyUI/models/ultralytics/segm/face_yolov8m-seg_60.pt')
    	# Run inference on an image
		results = model(source=path, mode='val')  # results list
    	# View results
		for r in results:
			face_zuobiao_masks = r.masks

		face_img_mask = np.ones_like(img)
		if face_zuobiao_masks != None:
        # 找出每个检测框
			face_zuobiao = face_zuobiao_masks.xy
			for item in face_zuobiao:
				points = np.array(item, dtype=np.int32)
				# 将轮廓列表转换为多维数组格式
				contours_a = np.array([points])
			cv2.fillPoly(face_img_mask, contours_a, (255, 255, 255))
            # 找到脸部的最低端坐标(找到 y 轴坐标最大的坐标点)
			max_y_coordinate_face = item[np.argmax(item[:, 1])]
			print('人脸填充完毕')	
		else:
			print('未检测到物体，固未填充')
		
		hair_img_mask = None
		model = YOLO(task='segment',model='/root/autodl-tmp/ComfyUI/models/ultralytics/segm/best_hair_117_epoch_v4.pt')
		results = model(source=path, mode='val')  # results list
		for r in results:
			hair_zuobiao_masks = r.masks
        	# 把检测到的所有边界框的四个坐标拿出来
			bbox_zuobiao = r.boxes.xyxy.cpu().numpy()
		left_top_zuobiao = bbox_zuobiao[:, 1]
		# 增加校验
		if left_top_zuobiao is None or len(left_top_zuobiao) == 0:
			print('未检测到物体，固未填充')
		else:
			max_index = np.argmin(left_top_zuobiao)
			max_value = left_top_zuobiao[max_index]
			hair_img_mask = np.zeros_like(img)
			if hair_zuobiao_masks != None:
				hair_zuobiao = hair_zuobiao_masks.xy
				item = hair_zuobiao[max_index]
				points = np.array(item, dtype=np.int32)
        		# 将轮廓列表转换为多维数组格式
				contours_a = np.array([points])
				cv2.fillPoly(hair_img_mask, contours_a, (255, 255, 255))
				# 找到头发的最低端坐标
				max_y_coordinate_hair = item[np.argmax(item[:, 1])]
				print('头发填充完毕')
			else:
				print('未检测到物体，固未填充')

		# 头发+脸部的蒙版
		if hair_img_mask is not None:
			hair_face_img = cv2.add(hair_img_mask, face_img_mask)
		else:
			hair_face_img=face_img_mask

		# if max_y_coordinate_face[1] > max_y_coordinate_hair[1]:
		# 	button_zuobiao = max_y_coordinate_face
		# else:
		# 	buttton_zuobiao = max_y_coordinate_hair
		
		# hair_face_img[:int((button_zuobiao[1] * 5) / 6), :] = 255
		final_img = cv2.subtract(person_img_mask, hair_face_img)
		# final_img=blacken_above_y(final_img,top)
		final_img1=np.copy(final_img)
		final_img20=blacken_below_y(final_img1,bootm)
		rs=create_mask_from_contours(final_img20,body20)
		# rs=blacken_below_y(rs,top)

		final=cv2.subtract(final_img,rs)
		# 反色处理
		inverted_mask = cv2.bitwise_not(final)

		result = cv2.cvtColor(inverted_mask, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(result)
		torch_img=pil_to_tensor_grayscale(pil_image)

		return (torch_img,)
	
	
		
		





