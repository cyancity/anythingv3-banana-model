from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO


model_id = "Linaqruf/anything-v3.0"


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    # device = 0 if torch.cuda.is_available() else -1
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = pipe.to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    result = model(prompt)

    output_buffer = BytesIO()
    result.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)

    # Return the results as a dictionary
    return base64_str
