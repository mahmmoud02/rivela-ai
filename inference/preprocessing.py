from PIL import Image
from torchvision import transforms

val_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(path_or_file):
    image = path_or_file if isinstance(path_or_file, Image.Image) else Image.open(path_or_file)
    image = image.convert("RGB")
    tensor = val_transforms(image)
    return tensor.unsqueeze(0)
