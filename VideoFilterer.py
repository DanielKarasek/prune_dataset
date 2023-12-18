import time

import torch
from torch.utils import data
from torchmetrics.functional import pairwise_euclidean_distance
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from resnet18 import Resnet18EmbeddingModel
from video_loader import VideoLoader


def filter_embeddings(embeddings: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    keep_indices = torch.full((embeddings.shape[0],), True, dtype=torch.bool).cuda()
    similarity_mat = 1.0 - pairwise_euclidean_distance(embeddings, zero_diagonal=False)
    similarity_mat[range(len(similarity_mat)), range(len(similarity_mat))] = 0.0
    for i in range(embeddings.shape[0]):
        if not keep_indices[i]:
            continue
        mask = torch.where(similarity_mat[i, :] < threshold, True, False).cuda()
        keep_indices = keep_indices & mask
    return keep_indices


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / torch.norm(embeddings, dim=1, keepdim=True)


class VideoFilterer:
    def __init__(self, video_path: str):
        video_loader = VideoLoader(video_path, transforms=Compose([Resize((224, 224)),
                                                                   CenterCrop((224, 224)),
                                                                   ToTensor(),
                                                                   Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])], ))
        # video_loader.play_video()
        model = Resnet18EmbeddingModel().cuda()

        dl = data.DataLoader(video_loader, batch_size=512, shuffle=False, num_workers=5)
        keep_indices_all = torch.empty(0, dtype=torch.int64).cuda()
        time_start = time.time()
        for batch_idx, img_batch in enumerate(dl):
            img_batch = img_batch.cuda()
            embeddings = model(img_batch)
            embeddings = normalize_embeddings(embeddings)
            keep_indices = filter_embeddings(embeddings, threshold=0.78)
            keep_indices = torch.where(keep_indices)[0] + batch_idx * 512
            keep_indices_all = torch.cat((keep_indices_all, keep_indices))
        print(f"Time taken: {time.time() - time_start}")
        keep_indices_all = keep_indices_all
        print(keep_indices_all)
        video_loader.keep_images(keep_indices_all)
        video_loader.play_video()
        print(len(video_loader))
