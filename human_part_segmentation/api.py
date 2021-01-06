import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils import model_zoo
from torchvision import transforms

from . import networks
from .utils.transforms import get_affine_transform, transform_logits


DATASET_SETTINGS = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'labels': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'],
        'ckpt': 'https://github.com/mayankgrwl97/human-part-segmentation/releases/download/0.0.1/exp-schp-201908261155-lip.pth'
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'labels': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'],
        'ckpt': 'https://github.com/mayankgrwl97/human-part-segmentation/releases/download/0.0.1/exp-schp-201908301523-atr.pth'
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'labels': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
        'ckpt': 'https://github.com/mayankgrwl97/human-part-segmentation/releases/download/0.0.1/exp-schp-201908270938-pascal-person-part.pth'
    }
}


class ImagePreprocessor:
    def __init__(self, input_size, transform):
        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def process(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }
        return input, meta


class HumanPartSegmentation:
    def __init__(self, dataset='lip', gpu='0'):
        # Input arguments
        self.dataset = dataset
        self.gpu = gpu
        self.device = torch.device('cuda:{}'.format(gpu))

        # Dataset Setting
        self.num_classes = DATASET_SETTINGS[self.dataset]['num_classes']
        self.input_size = DATASET_SETTINGS[self.dataset]['input_size']
        self.ckpt_path = DATASET_SETTINGS[self.dataset]['ckpt']
        self.labels = DATASET_SETTINGS[self.dataset]['labels']
        self.label_mapping = dict([(label, label_index) \
            for label_index, label in enumerate(self.labels)])

        # Load Model
        self.model = networks.init_model('resnet101',
            num_classes=self.num_classes, pretrained=None)
        state_dict = model_zoo.load_url(self.ckpt_path)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Image Preprocessor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                 std=[0.225, 0.224, 0.229])])
        self.image_preprocessor = ImagePreprocessor(
            input_size=self.input_size, transform=self.transform)

    def get_pixel_labels(self, img_path):
        """
        Get pixel-wise label predictions.

        Args:
            img_path (str): Path to Image
        """

        image, meta = self.image_preprocessor.process(img_path)

        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']

        image = image.unsqueeze(0)
        output = self.model(image.to(self.device))
        upsample = torch.nn.Upsample(
            size=self.input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(
            upsample_output.data.cpu().numpy(), c, s, w, h,
            input_size=self.input_size)

        pixel_labels = np.argmax(logits_result, axis=2)

        return pixel_labels

    def get_part_mask(self, img_path, part_labels):
        """
        Get segmentation mask for pixels corresponding to desired part labels

        Args:
            img_path (str): Path to Image
            part_labels (list): List of part labels
        """

        # predicted label indices for each pixel
        pixel_labels = self.get_pixel_labels(img_path)

        # label_indices[label] --> index of particular label
        part_label_indices = \
            [self.label_mapping[part_label] for part_label in part_labels]

        # compare predicted labels with desired labels
        pixel_labels_arr = np.expand_dims(pixel_labels, -1) # H x W x 1
        part_label_arr = np.unique(part_label_indices).reshape(1, 1, -1) # 1 x 1 x NUM_LABELS
        matching_pixels = np.any(pixel_labels_arr == part_label_arr, axis=-1) # H X W

        part_segmentation_mask = matching_pixels.astype(np.uint8) * 255

        return part_segmentation_mask
