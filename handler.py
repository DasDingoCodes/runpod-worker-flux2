import os
import io
import json
import base64
import pprint
import torch
import runpod

from PIL import Image
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from transformers import Mistral3ForConditionalGeneration
from diffusers.utils import load_image

from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

REPO_ID = "diffusers/FLUX.2-dev-bnb-4bit"
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"

def find_model_path(model_name) -> str:
    """
    Find the path to a cached model.
    
    Args:
        model_name: The model name from Hugging Face
        (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')
    
    Returns:
        The full path to the cached model, or None if not found
    """
    # Convert model name format: "Org/Model" -> "models--Org--Model"
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(CACHE_DIR, f"models--{cache_name}", "snapshots")
    
    # Check if the model exists in cache
    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            # Return the path to the first (usually only) snapshot
            return os.path.join(snapshots_dir, snapshots[0])
    
    raise Exception("Model Path not found")

# -----------------------------------------------------------------------------
# Model Loader
# -----------------------------------------------------------------------------
class ModelHandler:
    def __init__(self):
        self.pipe = None
        self.load_models()

    def load_models(self):
        model_path = find_model_path(REPO_ID)
        torch_dtype = torch.bfloat16

        transformer = Flux2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            device_map="cpu",
            local_files_only=True,
        )

        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=torch_dtype,
            device_map="cpu",
            load_in_4bit=True,
            local_files_only=True,
        )

        self.pipe = Flux2Pipeline.from_pretrained(
            model_path,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

        self.pipe.enable_model_cpu_offload()


MODELS = ModelHandler()

# -----------------------------------------------------------------------------
# Utility: Load image from URL or base64
# -----------------------------------------------------------------------------
def load_input_image(image_input):
    """
    Supports:
    - URL (http / https)
    - Base64 string (with or without data:image/... prefix)
    """
    if isinstance(image_input, str) and image_input.startswith("http"):
        return load_image(image_input).convert("RGB")

    if isinstance(image_input, str):
        if image_input.startswith("data:image"):
            image_input = image_input.split(",", 1)[1]

        image_bytes = base64.b64decode(image_input)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    raise ValueError("Unsupported image input format")

# -----------------------------------------------------------------------------
# Save + Upload
# -----------------------------------------------------------------------------
def save_and_upload(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    urls = []

    for idx, image in enumerate(images):
        path = f"/{job_id}/{idx}.png"
        image.save(path)

        if os.environ.get("BUCKET_ENDPOINT_URL"):
            urls.append(rp_upload.upload_image(job_id, path))
        else:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                urls.append(f"data:image/png;base64,{encoded}")

    rp_cleanup.clean([f"/{job_id}"])
    return urls

# -----------------------------------------------------------------------------
# RunPod Handler
# -----------------------------------------------------------------------------
@torch.inference_mode()
def generate_image(job):
    """
    Generate an image using FLUX.2
    """
    # -------------------------------------------------------------------------
    # 🐞 DEBUG LOGGING
    # -------------------------------------------------------------------------
    print("[generate_image] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    job_input = job["input"]

    print("[generate_image] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    try:
        validated = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        print("[generate_image] Validation exception:", err, flush=True)
        raise

    if "errors" in validated:
        return {"error": validated["errors"]}

    job_input = validated["validated_input"]

    print("[generate_image] validated input:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # -------------------------------------------------------------------------
    # Seed handling
    # -------------------------------------------------------------------------
    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator(device="cuda").manual_seed(job_input["seed"])

    # -------------------------------------------------------------------------
    # Load input images
    # -------------------------------------------------------------------------
    images = []
    for idx, img_input in enumerate(job_input["image_inputs"]):
        try:
            images.append(load_input_image(img_input))
        except Exception as err:
            return {"error": f"Failed to load image {idx}: {err}"}

    # -------------------------------------------------------------------------
    # Run FLUX
    # -------------------------------------------------------------------------
    result = MODELS.pipe(
        prompt=job_input.get("prompt"),
        image=images,
        generator=generator,
        num_inference_steps=job_input["num_inference_steps"],
        guidance_scale=job_input["guidance_scale"],
        height=job_input["height"],
        width=job_input["width"],
        num_images_per_prompt=job_input["num_images"],
    )

    image_urls = save_and_upload(result.images, job["id"])

    return {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }


runpod.serverless.start({"handler": generate_image})
