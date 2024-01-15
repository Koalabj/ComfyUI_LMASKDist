NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}



def remote_images():
	global NODE_CLASS_MAPPINGS
	global NODE_DISPLAY_NAME_MAPPINGS
	from .nodes.remote_images import LoadImageUrl,BodyMask,addImage,exImage
	NODE_CLASS_MAPPINGS.update({
		"LMASKImage": LoadImageUrl,
		"BodyMask":BodyMask,
		"addImage":addImage,
		"exImage":exImage,
	})
	NODE_DISPLAY_NAME_MAPPINGS.update({
		"LMASKImage": "LMASKImage",
		"BodyMask":"BodyMask",
		"addImage":"addImage",
		"exImage":"exImage"
	})



print("Loading network distribution node pack")
remote_images()
