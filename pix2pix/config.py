import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dir = 'data/maps/maps/train'
val_dir = 'data/maps/maps/val'
lr = 2e-4
batch_size = 32
num_workers = 2
image_size = 256
channels_img = 3
L1_Lambda = 100
Lambda_GP = 10
num_epochs = 5
load_model = True
save_model = False
checkpoint_disc = 'Pix2Pix_Satellite_to_Map/disc.pth.tar'
checkpoint_gen = 'Pix2Pix_Satellite_to_Map/gen.pth.tar'

both_transform = A.Compose(
            [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
            )

transform_only_input = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.2),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                ToTensorV2(),
            ]
            )

transform_only_mask = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                ToTensorV2(),
            ]
            )

