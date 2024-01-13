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
def optimize_jagged_edges(image, blur_kernel=(5, 5), morph_kernel=(3, 3)):
    # 高斯模糊以平滑图像
    blurred_image = cv2.GaussianBlur(image, blur_kernel, 0)
    
    # 二值化处理
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    
    # 形态学操作的内核
    kernel = np.ones(morph_kernel, np.uint8)
    
    # 先腐蚀再膨胀，称为开运算，可以去除小的噪点
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    # 再次膨胀后腐蚀，称为闭运算，可以填补小的闭合区域
    result_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel)
    
    return result_image


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

# def blacken_above_y(mask, y_coord):
#     if y_coord < 0 or y_coord >= mask.shape[0]:
#         raise ValueError("y_coord is out of the image bounds.")

#     # 将纵坐标以上的部分设置为黑色
#     mask[:y_coord, :] = 0
#     return mask

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

def cv2_image_to_tensor(image):
    # 确保图像是三维的
    if len(image.shape) == 2:
        # 如果是单通道图像，添加一个新的维度
        image = image[:, :, np.newaxis]
    
    # 转换BGR图像为RGB格式
    if image.shape[2] == 3:  # 如果有3个通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换图像为CHW格式
    image = image.transpose((2, 0, 1))
    
    # 将图像数据类型转换为float，并缩放到[0, 1]范围
    image = image.astype(np.float32) / 255.0
    
    # 将numpy数组转换为torch张量
    tensor = torch.from_numpy(image)
    
    # 添加批次维度
    tensor = tensor.unsqueeze(0)
    
    return tensor
# 膨胀腐蚀
def process_image_to_tensor(image_path):
    """
    This function processes an image: it converts an image to a grayscale mask,
    performs erosion followed by dilation, and returns the result as a tensor.

    :param image_path: Path to the input image.
    :return: Processed image as a tensor.
    """

    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to create a binary mask
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Define the kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    # Erode and then dilate the mask
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1)

    # Convert the processed image to a tensor
    tensor_output = np.array(mask_dilated)

    return tensor_output

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
def overlay_images(background_img, overlay_img, position=(0, 0)):
    """
    Overlays an image with transparency over another image at a specified position.

    Parameters:
    background_img (numpy.ndarray): The background image.
    overlay_img (numpy.ndarray): The overlay image (must have an alpha channel).
    position (tuple): A tuple (x, y) specifying where the overlay image should be placed on the background.

    Returns:
    numpy.ndarray: The resulting image after overlay.
    """

    # Split the overlay image into its RGB and Alpha components
    if overlay_img.shape[2] == 4:
        overlay_rgb = overlay_img[..., :3]
        overlay_alpha = overlay_img[..., 3] / 255.0
    else:
        # 如果没有alpha通道，创建一个完全不透明的alpha通道
        overlay_rgb = overlay_img
        overlay_alpha = np.ones((overlay_img.shape[0], overlay_img.shape[1]))


    # Extract the region of interest (ROI) from the background image where the overlay will be placed
    x, y = position
    h, w = overlay_img.shape[:2]
    roi = background_img[y:y+h, x:x+w]

    # Use the alpha channel as a mask for blending
    img1_bg = cv2.multiply(1.0 - overlay_alpha[..., np.newaxis], roi)
    img2_fg = cv2.multiply(overlay_alpha[..., np.newaxis], overlay_rgb)

    # Combine the background and overlay images
    combined = cv2.add(img1_bg, img2_fg)

    # Place the combined image back into the original image
    background_img[y:y+h, x:x+w] = combined

    return background_img
def tensor_to_cv2_image(tensor):
    # 假设 tensor 是一个 numpy 数组

    # 如果张量的数据类型是浮点数，将其转换为 [0, 255] 范围内的整数
    if tensor.dtype == np.float32 or tensor.dtype == np.float64:
        tensor = (tensor * 255).astype(np.uint8)

    # 如果张量的形状是 [通道数, 高度, 宽度]，转换为 [高度, 宽度, 通道数]
    if len(tensor.shape) == 3 and tensor.shape[0] < tensor.shape[1] and tensor.shape[0] < tensor.shape[2]:
        tensor = np.transpose(tensor, (1, 2, 0))

    # 如果张量是 RGB 格式，转换为 BGR 格式
    if tensor.shape[2] == 3:
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)

    return tensor
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
class addImage:
    def __init__(self):
      pass
    @classmethod
    def INPUT_TYPES(s):
     return {
			"required": {
				"faceImage": ("IMAGE",),
			},
			"optional": {
				"image": ("IMAGE",)
			}
		}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "AddImage"
    CATEGORY = "remote"
    def AddImage(self,faceImage):
        person=tensor_to_pil(faceImage)
        path="/root/autodl-tmp/ComfyUI/input/yt1.png"
        person.save(path)
        face=process_image_to_tensor(path)
        return (face,)
        
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
		head=im_read(image)
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
		person_img = optimize_jagged_edges(person_img)



		# 获取输入的图片
		# pic=tensor_to_pil(image)
		# path="/root/autodl-tmp/ComfyUI/input/yt.png"
		# pic.save(path)

		# original_img = cv2.imread(path)
		# gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
		# _, img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

		
		
		# model = YOLO(task='segment',model='/root/autodl-tmp/ComfyUI/models/ultralytics/segm/face_yolov8m-seg_60.pt')
    	# # Run inference on an image
		# results = model(source=path, mode='val')  # results list
    	# # View results
		# for r in results:
		# 	face_zuobiao_masks = r.masks

		# face_img_mask = np.ones_like(img)
		# max_y_coordinate_face = -np.inf 
		# if face_zuobiao_masks != None:
        # # 找出每个检测框
		# 	face_zuobiao = face_zuobiao_masks.xy
		# 	for item in face_zuobiao:
		# 		points = np.array(item, dtype=np.int32)
		# 		# 将轮廓列表转换为多维数组格式
		# 		contours_a = np.array([points])
		# 	cv2.fillPoly(face_img_mask, contours_a, (255, 255, 255))
        #     # 找到脸部的最低端坐标(找到 y 轴坐标最大的坐标点)
		# 	y_coordinate = max(item[:, 1])
		# 	if y_coordinate > max_y_coordinate_face:
		# 		max_y_coordinate_face = y_coordinate
		# 	print(f"脸部y最低坐标：{max_y_coordinate_face}")
		# 	print('人脸填充完毕')	
		# else:
		# 	print('未检测到物体，固未填充')
		
		# hair_img_mask = None
		# model = YOLO(task='segment',model='/root/autodl-tmp/ComfyUI/models/ultralytics/segm/best_hair_117_epoch_v4.pt')
		# results = model(source=path, mode='val')  # results list
		# for r in results:
		# 	hair_zuobiao_masks = r.masks
        # 	# 把检测到的所有边界框的四个坐标拿出来
		# 	bbox_zuobiao = r.boxes.xyxy.cpu().numpy()
		# left_top_zuobiao = bbox_zuobiao[:, 1]
		# # 增加校验
		# if left_top_zuobiao is None or len(left_top_zuobiao) == 0:
		# 	print('未检测到物体，固未填充')
		# else:
		# 	max_index = np.argmin(left_top_zuobiao)
		# 	max_value = left_top_zuobiao[max_index]
		# 	hair_img_mask = np.zeros_like(img)
		# 	if hair_zuobiao_masks != None:
			# 	hair_zuobiao = hair_zuobiao_masks.xy
			# 	item = hair_zuobiao[max_index]
			# 	points = np.array(item, dtype=np.int32)
        	# 	# 将轮廓列表转换为多维数组格式
			# 	contours_a = np.array([points])
			# 	cv2.fillPoly(hair_img_mask, contours_a, (255, 255, 255))
			# 	# 找到头发的最低端坐标
			# 	max_y_coordinate_hair = item[np.argmax(item[:, 1])]
			# 	print('头发填充完毕')
			# else:
			# 	print('未检测到物体，固未填充')
		
		# 头发+脸部的蒙版
		# if hair_img_mask is not None:
		# 	hair_face_img = cv2.add(hair_img_mask, face_img_mask)
		# else:
		# 	hair_face_img=face_img_mask
        
		# path="/root/autodl-tmp/ComfyUI/input/yt3.png"
		# cv2.imwrite(path,hair_face_img)
		# hair_face_img1=cv2.imread(path)
		# hair_face_img1 = cv2.cvtColor(hair_face_img1, cv2.COLOR_BGR2GRAY)

		# 使用全身照减去头部
		final_img = cv2.subtract(person_img, head)
		# 计算最低点坐标
		# 临时测试
		bootm=0
		# if(top>max_y_coordinate_face):
		# 	bootm=max_y_coordinate_face
		# else:
		bootm=top
		
		final_img1=np.copy(final_img)
		print(f"计算的最低点坐标为{bootm}")
		final=blacken_above_y(final_img1,int(bootm))
		# 反色处理
		inverted_mask = cv2.bitwise_not(final)
            
		




		result = cv2.cvtColor(inverted_mask, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(result)
		torch_img=pil_to_tensor_grayscale(pil_image)
        
		


		return (torch_img,)
	
	
		
		





