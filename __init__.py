NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}



def remote_images():
	global NODE_CLASS_MAPPINGS
	global NODE_DISPLAY_NAME_MAPPINGS
	from .nodes.remote_images import LoadImageUrl
	NODE_CLASS_MAPPINGS.update({
		"LMASKImage": LoadImageUrl,
	})
	NODE_DISPLAY_NAME_MAPPINGS.update({
		"LMASKImage": "LMASKImage",
	})



print("Loading network distribution node pack")
remote_images()
