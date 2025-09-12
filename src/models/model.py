import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import copy
import time
import os

cudnn.benchmark = True


class ImageClassifierModel:
    def __init__(self, data_dir, num_classes=2, batch_size=4, num_workers=4, num_epochs=25):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs

        # Device and dataloaders from shared utils
        from src.utils.data_utils import get_device, create_dataloaders
        self.device = get_device()
        self.dataloaders, self.dataset_sizes, self.class_names = create_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=224,
            device=self.device,
        )

        # Model setup (transfer learning with ResNet18) using new weights API
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train_model(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model
