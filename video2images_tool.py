import cv2
from einops import rearrange
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from video_loader import VideoLoader


class Video2ImageDSConvertor:
    @staticmethod
    def convert(video_path, output_path):
        video_loader = VideoLoader(video_path, transforms=Compose([Resize((224, 224)),
                                                                   CenterCrop((224, 224)),
                                                                   ToTensor()]))
        for idx, frame in enumerate(video_loader):
            frame = rearrange(frame, "c h w -> h w c")
            frame = (frame * 255).numpy().astype("uint8")
            cv2.imwrite(f"{output_path}/{idx}.jpg", frame)
