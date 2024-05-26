from torch import device, cuda, load, no_grad, sigmoid, Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from data import EvaluationDataset
from model import UNet
from utils import save_results

if __name__ == '__main__':
    device = device("cuda" if cuda.is_available() else "cpu")
    model = UNet(3, 1).to(device)
    model.load_state_dict(load("unet.pth"))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset_internal = EvaluationDataset("data/imagesTs-Internal", transform=transform)
    loader_internal = DataLoader(dataset_internal, batch_size=4, shuffle=False)
    dataset_external = EvaluationDataset("data/imagesTs-External", transform=transform)
    loader_external = DataLoader(dataset_external, batch_size=4, shuffle=False)
    for batch_idx, targets in enumerate(loader_internal):
        assert isinstance(targets, Tensor)
        targets = targets.to(device)
        with no_grad():
            masks = sigmoid(model(targets))
            masks = masks.float()
        save_results(masks, "results/internal", dataset_internal, batch_idx)
    for batch_idx, targets in enumerate(loader_external):
        assert isinstance(targets, Tensor)
        targets = targets.to(device)
        with no_grad():
            masks = sigmoid(model(targets))
            masks = masks.float()
        save_results(masks, "results/external", dataset_external, batch_idx)
