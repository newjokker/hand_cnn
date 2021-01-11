import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def train_transforms():
    return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(p=0.9),
                A.RandomBrightnessContrast(p=0.9),
                A.RandomGamma(p=0.9),
            ], p=0.9),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            # A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def valid_transforms():
    return A.Compose([
            ToTensorV2(p=1.),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def test_transform():
    return A.Compose(
        [
            ToTensorV2(p=1.),
        ],
        p=1.0
    )
