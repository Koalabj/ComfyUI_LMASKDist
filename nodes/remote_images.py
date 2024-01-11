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

def make_non_black_white(tensor):
    """
    Convert all non-black pixels of an image tensor to white and return the modified image tensor.

    Parameters:
    tensor (torch.Tensor): An image tensor with shape (C, H, W) and pixel values in [0, 1].

    Returns:
    torch.Tensor: Modified image tensor where all non-black pixels have been changed to white.
    """
    tensor = tensor.to(torch.float32)
    # 检查张量的维度
    if tensor.ndim == 2:  # 灰度图（无通道维度）
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:  # 单通道图
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] != 3:
            raise ValueError("Input tensor must have 1 or 3 channels")
    elif tensor.ndim == 4 and tensor.shape[1] == 3:  # 批处理张量
        # 这里可以根据需要处理批处理张量
        tensor = tensor[0]
        # raise ValueError("Batch tensors are not supported. Please input a single image tensor.")
    else:
        raise ValueError("Unsupported tensor format")
	# 将张量转换为 0 到 1 范围
    tensor = torch.clamp(tensor, min=0.0, max=1.0)

    # 创建一个与输入张量形状相同的白色张量
    white_tensor = torch.ones_like(tensor)

    # 寻找所有非黑色像素（即任何通道大于 0 的像素）
    non_black_mask = tensor.max(dim=0, keepdim=True).values > 0

    # 将非黑色像素替换为白色
    tensor.masked_scatter_(non_black_mask, white_tensor.masked_select(non_black_mask))

    # # Ensure tensor is on CPU and convert to numpy array
    # numpy_image = tensor.cpu().numpy()

    # # Check if the tensor is in the expected shape (C, H, W)
    # if numpy_image.shape[0] != 3:
    #     raise ValueError("Input tensor must have 3 channels")

    # # Convert to 8-bit format and transpose to HWC for processing
    # numpy_image = np.transpose(numpy_image, (1, 2, 0))
    # numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)

    # # Find all non-black pixels (any pixel where all RGB values are not 0) and set them to white
    # mask = np.any(numpy_image != 0, axis=-1)
    # numpy_image[mask] = [255, 255, 255]

    # # Convert back to CHW format and normalize to [0, 1]
    # numpy_image = np.transpose(numpy_image, (2, 0, 1)).astype(np.float32) / 255.0

    # Convert back to PyTorch tensor
    return tensor

#平滑处理二值图
def tensor_to_image(tensor: torch.Tensor) -> np.array:
    """Converts a PyTorch tensor image to a numpy image.

    In case the tensor is in the GPU, it will be copied back to CPU.

    Args:
        tensor (torch.Tensor): image of the form :math:`(H, W)`, :math:`(C, H, W)` or
            :math:`(B, C, H, W)`.

    Returns:
        numpy.ndarray: image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: np.array = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        image = image
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(
            "Cannot process tensor with shape {}".format(input_shape))

    return image

def blacken_above_y(mask, y_coord):
    if y_coord < 0 or y_coord >= mask.shape[0]:
        raise ValueError("y_coord is out of the image bounds.")

    # 将纵坐标以上的部分设置为黑色
    mask[:y_coord, :] = 0
    return mask

def create_smooth_bezier_polygon(mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_mask = np.zeros_like(mask)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        original_radius = max(w, h) / 2

        # 计算新的中心点
        center_x = x + w // 2
        # vertical_shift = int(original_radius * random.uniform(0.5, 1))
        # center_y = y + h // 2 + vertical_shift
        center_y = y + h // 2

        # 设置圆的半径
        radius = int(original_radius * random.uniform(1.5,1.8))

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
		# body = cv2.bitwise_not(body)

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
		# 获取衣服的蒙版图
		body=im_read(body_mask)
		#获取整个人的蒙版图
		# person_img_mask=tensor_to_image(person_mask)
		person=tensor_to_pil(person_mask)
		path="/root/autodl-tmp/ComfyUI/input/yt1.png"
		person.save(path)
		#获取衣服蒙版的定坐标
		top=getMaskTop(body)
		print(f"顶部坐标{top}")
		person=cv2.imread(path)
		person_img = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
		# _, numpy_image = cv2.threshold(person_img, 127, 255, cv2.THRESH_BINARY)
		# kernel = np.ones((3,3),np.uint8) 
		# numpy_image = cv2.erode(numpy_image, kernel, iterations=1)
		# person_img_mask = cv2.dilate(numpy_image, kernel, iterations=1)
		# path="/root/autodl-tmp/ComfyUI/input/yt2.png"
		# cv2.imwrite(path,person_img_mask)
		# person_img_mask=person
		# 获取输入的图片
		pic=tensor_to_pil(image)
		path="/root/autodl-tmp/ComfyUI/input/yt.png"
		pic.save(path)

		original_img = cv2.imread(path)
		gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
		_, img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

		
		
		model = YOLO(task='segment',model='/root/autodl-tmp/ComfyUI/models/ultralytics/segm/face_yolov8m-seg_60.pt')
    	# Run inference on an image
		results = model(source=path, mode='val')  # results list
    	# View results
		for r in results:
			face_zuobiao_masks = r.masks

		face_img_mask = np.ones_like(img)
		max_y_coordinate_face = -np.inf 
		if face_zuobiao_masks != None:
        # 找出每个检测框
			face_zuobiao = face_zuobiao_masks.xy
			for item in face_zuobiao:
				points = np.array(item, dtype=np.int32)
				# 将轮廓列表转换为多维数组格式
				contours_a = np.array([points])
			cv2.fillPoly(face_img_mask, contours_a, (255, 255, 255))
            # 找到脸部的最低端坐标(找到 y 轴坐标最大的坐标点)
			y_coordinate = max(item[:, 1])
			if y_coordinate > max_y_coordinate_face:
				max_y_coordinate_face = y_coordinate
			print(f"脸部y最低坐标：{max_y_coordinate_face}")
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
        
		path="/root/autodl-tmp/ComfyUI/input/yt3.png"
		cv2.imwrite(path,hair_face_img)
		hair_face_img1=cv2.imread(path)
		hair_face_img1 = cv2.cvtColor(hair_face_img1, cv2.COLOR_BGR2GRAY)

		# 使用全身照减去头部
		final_img = cv2.subtract(person_img, hair_face_img1)
		# 计算最低点坐标
		# 临时测试
		bootm=0
		if(top>max_y_coordinate_face):
			bootm=max_y_coordinate_face
		else:
			bootm=top
		
		final_img1=np.copy(final_img)
		print(f"计算的最低点坐标为{bootm}")
		final=blacken_above_y(final_img1,int(bootm))
		# 反色处理
		inverted_mask = cv2.bitwise_not(final)
		if len(inverted_mask.shape) == 2 or inverted_mask.shape[2] == 1:
			image = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)
		result_image = inverted_mask.copy()

		# 创建一个黑色掩膜，标记出所有非黑色的像素
		black_mask = np.all(inverted_mask == [0, 0, 0], axis=-1)

		# 创建一个白色掩膜，标记出所有非白色的像素
		white_mask = np.all(inverted_mask == [255, 255, 255], axis=-1)

		# 标记出所有既非黑色也非白色的像素
		non_black_white_mask = ~(black_mask | white_mask)

		# 将非黑非白的像素替换为白色
		result_image[non_black_white_mask] = [255, 255, 255]




		result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(result)
		torch_img=pil_to_tensor_grayscale(pil_image)
        # # 反色处理
		# s = 1.0 - torch_img
        
		# s1=make_non_black_white(s)
		# s_copy = s.clone()
		# numpy_image = s_copy.cpu().numpy()
		# if numpy_image.ndim == 4:
        # # 假设是批量处理的图像，取第一个图像
		# 	numpy_image = numpy_image[0]
		# if numpy_image.ndim == 3 and numpy_image.shape[0] == 3:
        # # 正确的 CHW 维度，进行转置
		# 	numpy_image = np.transpose(numpy_image, (1, 2, 0))
		# # numpy_image = np.transpose(numpy_image, (1, 2, 0))  # 转换 CHW 到 HWC
		# numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
        # # 处理图像
		# height, width, _ = numpy_image.shape
		# threshold=220
		# for y in range(height):
		# 	for x in range(width):
        #     # 检查是否接近白色
		# 		if all(numpy_image[y, x] > threshold):
		# 				numpy_image[y, x] = [255, 255, 255]
		# if len(numpy_image.shape) == 2:
		# 	numpy_image = numpy_image[:, :, np.newaxis]

		# # 转换数组形状为 (C, H, W)
		# numpy_image = np.transpose(numpy_image, (2, 0, 1))

		# # 将数组类型转换为浮点数，并标准化到 [0.0, 1.0]
		# numpy_image = numpy_image.astype(np.float32) / 255.0

		# # 将 NumPy 数组转换为 PyTorch 张量
		# tensor_image = torch.from_numpy(numpy_image)
		


		return (torch_img,)
	
	
		
		





