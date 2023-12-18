import cv2
import PIL.Image as Image

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset


class VideoLoader(TorchDataset):
    def __init__(self, video_path: str, transforms):
        super(VideoLoader, self).__init__()
        self._video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self._frame_count = 10_000
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (self._frame_width, self._frame_height)

        self._all_frames = [cap.read()[1] for _ in range(self._frame_count)]
        self._transformation = transforms
        self._iter_idx = 0

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def __len__(self):
        return self._frame_count

    def __getitem__(self, idx: int) -> Tensor:
        if idx >= self._frame_count:
            raise Exception("Frame index out of bounds")

        frame = Image.fromarray(self._all_frames[idx])
        frame = self._transformation(frame)
        return frame

    def play_video(self):
        for frame in self._all_frames:
            cv2.imshow("video", frame)
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def show_frame(self, idx: int):
        if idx >= self._frame_count:
            raise Exception("Idx out of bounds")
        cv2.imshow("video", self._all_frames[idx])
        cv2.waitKey(0)

    def keep_images(self, keep_indices: Tensor):
        self._all_frames = [self._all_frames[idx] for idx in keep_indices]
        self._frame_count = len(self._all_frames)

    def __iter__(self):
        return self

    def __next__(self,):
        if self._iter_idx >= self._frame_count:
            raise StopIteration
        self._iter_idx += 1
        return self[self._iter_idx-1]
