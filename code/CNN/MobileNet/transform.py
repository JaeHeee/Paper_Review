from torchvision import transforms


def transform():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])

    return transform
