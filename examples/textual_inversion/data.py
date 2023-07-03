from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import random
from datasets import load_dataset


PIL_INTERPOLATION = {
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        learnable_property: str="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.ds = load_dataset(dataset_name, split="train")

        self.num_images = len(self.ds)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = PIL_INTERPOLATION[interpolation]

        from templates import IMAGENET_STYLE_TEMPLATES_SMALL, IMAGENET_TEMPLATES_SMALL
        self.templates = IMAGENET_STYLE_TEMPLATES_SMALL if learnable_property == "style" else IMAGENET_TEMPLATES_SMALL
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = self.ds[i % self.num_images]['image']

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            h, w = img.shape[0], img.shape[1]
            crop = min(h, w)
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

