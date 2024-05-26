from model import UNet
from data import TrainingDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import device, cuda, nn, optim, save


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    loader = DataLoader(TrainingDataset("data/imagesTr", "data/labelsTr", transform), batch_size=8, shuffle=True)
    device = device("cuda" if cuda.is_available() else "cpu")
    model = UNet(3, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 128
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(loader)}")
    save(model.state_dict(), "unet.pth")
