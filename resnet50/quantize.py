import torch
#################### code changes ####################
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
######################################################

##### Example Model #####
import torchvision.models as models
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
data = torch.rand(1, 3, 224, 224)
#########################

qconfig = ipex.quantization.default_static_qconfig

prepared_model = prepare(model, qconfig, example_inputs=data, inplace=False)

##### Example Dataloader #####
import torchvision
DOWNLOAD = True
DATA = '/home/krishna-intel/imagenet/data'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.ImageNet(root='/home/krishna-intel/imagenet/data', split='val', transform=transform)
val_dataset_subset = torch.utils.data.Subset(test_dataset, range(1000))
calibration_data_loader = torch.utils.data.DataLoader(val_dataset_subset, batch_size=128, shuffle=False)


for batch_idx, (d, target) in enumerate(calibration_data_loader):
  print(f'calibrated on batch {batch_idx} out of {len(calibration_data_loader)}')
  prepared_model(d)
##############################

converted_model = convert(prepared_model)
with torch.no_grad():
  traced_model = torch.jit.trace(converted_model, data)
  traced_model = torch.jit.freeze(traced_model)

traced_model.save("quantized_model.pt")