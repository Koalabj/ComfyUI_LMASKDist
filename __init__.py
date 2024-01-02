NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}



def remote_images():
	global NODE_CLASS_MAPPINGS
	global NODE_DISPLAY_NAME_MAPPINGS
	from .nodes.remote_images import LoadImageUrl,BodyMask
	NODE_CLASS_MAPPINGS.update({
		"LMASKImage": LoadImageUrl,
		"BodyMask":BodyMask
	})
	NODE_DISPLAY_NAME_MAPPINGS.update({
		"LMASKImage": "LMASKImage",
		"BodyMask":BodyMask
	})



print("Loading network distribution node pack")
remote_images()
