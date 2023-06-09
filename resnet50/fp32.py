import torch
import torchvision
from time import time

model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()

#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
######################################################

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


test_dataset = torchvision.datasets.ImageNet(root='/home/krishna-intel/imagenet/data', split='val', transform=transform)
val_dataset_subset = torch.utils.data.Subset(test_dataset, range(1000))


test_loader = torch.utils.data.DataLoader(val_dataset_subset, batch_size=128, shuffle=False)

def inferModel(model, test_loader):
    correct = 0
    total = 0
    infer_time = 0

    with torch.no_grad():
        num_batches = len(test_loader)
        batches=0

        for i, data in enumerate(test_loader):

            # Record time for Inference
            start_time = time()
            images, labels = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Record time after finishing batch inference
            end_time = time()

            infer_time += (end_time-start_time)
            batches += 1


    accuracy = 100 * correct / total
    print('Accuracy = ', accuracy)
    print('Time Taken', infer_time*1000/(batches*128))


with torch.no_grad():
  inferModel(model, test_loader)