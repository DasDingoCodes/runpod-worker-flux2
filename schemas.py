INPUT_SCHEMA = {
    "check_cuda": {
        "type": bool,
        "required": False,
        "default": False,
    },
    "prompt": {
        "type": str,
        "required": False,
    },
    "image_inputs": {
        "type": list,
        "required": True,
    },
    "height": {
        "type": int,
        "required": False,
        "default": 1080,
    },
    "width": {
        "type": int,
        "required": False,
        "default": 1920,
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 28,
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 2.5,
    },
    "num_images": {
        "type": int,
        "required": False,
        "default": 1,
    },
    "seed": {
        "type": list,
        "required": False,
        "default": None,
    },
}
