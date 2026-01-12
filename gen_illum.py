import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image
from PIL import Image

model_id = "VisualCloze/VisualClozePipeline-512"
pipe = VisualClozePipeline.from_pretrained(
    model_id,
    resolution=512,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

example_input = load_image("images/Place34_1.jpg")
example_output = load_image("images/Place34_12.jpg")

query_input = load_image("images/Place33_1.jpg")

image_paths = [
    [example_input, example_output],
    [query_input, None]
]

task_prompt = "Change the global illumination of the image to intense red lighting while preserving the object's structure and details."
content_prompt = "A high-quality photo of the object under strong red ambient light, realistic shadows, cinematic atmosphere."

result = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    guidance_scale=7.5,
    num_inference_steps=30,
    upsampling_width=1024,
    upsampling_height=1024,
    upsampling_strength=0.0,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0][0]

result.save("augmented_illumination_red.jpg")