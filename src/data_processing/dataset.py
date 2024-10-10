from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),  
    transforms.ToTensor()          
])

def SpectrogramDataset(root, transform=transform):    

    return datasets.ImageFolder(root=root,
                                transform=transform)
