from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import fargv
from clamm_dataset import ClammDataset


p = {
    "device": "cpu",
    "train_ds_imgroot": "./data/ICDAR2017_CLaMM_task1_task3/",
    "train_ds_gt_fname": "data/ICDAR2017_CLaMM_task1_task3/@ICDAR2017_CLaMM_task1_task3.csv",
    "val_ds_imgroot": "./data/ICDAR2017_CLaMM_task1_task3/",
    "val_ds_gt_fname": "data/ICDAR2017_CLaMM_task1_task3/@ICDAR2017_CLaMM_task1_task3.csv",
    "model_name": ("squeezenet", "resnet", "alexnet", "vgg", "densenet", "inception"),
    "num_workers": 4,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": 15,
    "feature_extracting": 1,
    "use_pretrained": 1,
}

args, _ = fargv.fargv(p)


def train_model(args, model, dataloaders, criterion, optimizer, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                print("Labels:",np.unique(labels.detach().cpu().numpy()))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history




def initialize_model(args):
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if args.model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model_ft, args.feature_extracting)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model_ft, args.feature_extracting)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model_ft, args.feature_extracting)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model_ft, args.feature_extracting)
        model_ft.classifier[1] = nn.Conv2d(512, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = args.num_classes
        input_size = 224

    elif args.model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model_ft, args.feature_extracting)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model_ft, args)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


train_ds = ClammDataset(img_root="./data/ICDAR2017_CLaMM_task1_task3/", script_not_date = True, gt_fname="data/ICDAR2017_CLaMM_task1_task3/@ICDAR2017_CLaMM_task1_task3.csv")
val_ds = ClammDataset(img_root="./data/ICDAR2017_CLaMM_task1_task3/", script_not_date = True, gt_fname="data/ICDAR2017_CLaMM_task1_task3/@ICDAR2017_CLaMM_task1_task3.csv")
dataloaders_dict = {x: torch.utils.data.DataLoader(eval(f"{x}_ds"), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) for x in ['train', 'val']}


# Initialize the model for this run
model_ft, input_size = initialize_model(args)

# Print the model we just instantiated
print(model_ft)




# Send the model to GPU
model_ft = model_ft.to(args.device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if args.feature_extracting:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
#model_ft, hist = train_model(args, model_ft, dataloaders_dict, criterion, optimizer_ft, is_inception=(args.model_name=="inception"))



# Initialize the non-pretrained version of the model used for this run
scratch_model,_ = initialize_model(args)
scratch_model = scratch_model.to(args.device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
#_,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=args.num_epochs, is_inception=(args.model_name=="inception"))
_, scratch_hist = train_model(args, scratch_model, dataloaders_dict, criterion, optimizer_ft, is_inception=(args.model_name=="inception"))
# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,args.num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,args.num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, args.num_epochs+1, 1.0))
plt.legend()
plt.show()