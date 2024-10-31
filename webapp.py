from pet_seg_core.model import UNet
from pet_seg_core.config import PetSegWebappConfig
from pet_seg_core.gdrive_utils import GDriveUtils

from torchvision import transforms as T
import torch
import gradio as gr
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()

device = torch.device("cpu")

if PetSegWebappConfig.DOWNLOAD_MODEL_WEIGTHS_FROM_GDRIVE:
    GDriveUtils.download_file_from_gdrive(
        PetSegWebappConfig.MODEL_WEIGHTS_GDRIVE_FILE_ID, PetSegWebappConfig.MODEL_WEIGHTS_LOCAL_PATH
    )

model = UNet.load_from_checkpoint(PetSegWebappConfig.MODEL_WEIGHTS_LOCAL_PATH).to(device)
model.eval()

def segment_image(img):
    img = T.ToTensor()(img).unsqueeze(0).to(device)
    mask = model(img)
    mask = torch.argmax(mask, dim = 1).squeeze().detach().cpu().numpy()
    return mask

def overlay_mask(img, mask, alpha=0.5):
    # Define color mapping
    colors = {
        0: [255, 0, 0],   # Class 0 - Red
        1: [0, 255, 0],   # Class 1 - Green
        2: [0, 0, 255]    # Class 2 - Blue
        # Add more colors for additional classes if needed
    }

    # Create a blank colored overlay image
    overlay = np.zeros_like(img)

    # Map each mask value to the corresponding color
    for class_id, color in colors.items():
        overlay[mask == class_id] = color

    # Blend the overlay with the original image
    output = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    return output

def transform(img):
    mask=segment_image(img)
    blended_img = overlay_mask(img, mask)
    return blended_img

app = gr.Interface(
    fn=transform, 
    inputs=gr.Image(label="Input Image"), 
    outputs=gr.Image(label="Image with Segmentation Overlay"), 
    title="Image Segmentation on Pet Images",
    description="Segment image of a pet animal into three classes: background, pet, and boundary.",
    examples=[
        "example_images/img1.jpg",
        "example_images/img2.jpg",
        "example_images/img3.jpg"
    ]
)