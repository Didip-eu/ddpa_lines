from contextlib import ExitStack
import torch
import tqdm
import numpy as np
from torchvision import models
from torch import nn


from typing import Tuple, Union, List

last = lambda x: x[sorted(x.keys())[-1]]


def initialize_model(args):
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0
    if args.model == "resnet":
        model = models.resnet18(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model, args.feature_extracting)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224
    elif args.model == "alexnet":
        model = models.alexnet(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model, args.feature_extracting)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224
    elif args.model == "vgg":
        model = models.vgg11_bn(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model, args.feature_extracting)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224
    elif args.model == "squeezenet":
        model = models.squeezenet1_0(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model, args.feature_extracting)
        model.classifier[1] = nn.Conv2d(512, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = args.num_classes
        input_size = 224
    elif args.model == "densenet":
        model = models.densenet121(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model, args.feature_extracting)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224
    elif args.model == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=args.use_pretrained)
        set_parameter_requires_grad(model, args)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
        input_size = 299
    else:
        raise ValueError("Invalid model name")
    model.last_epoch = 0
    model.train_history = {}
    model.val_history = {}
    model.args_history = {0: args}
    model = model.to(args.device)
    return model, input_size


def save(model, fname=""):
    if fname == "":
        fname = last(model.args_history).resume_fname
    metadata = {"last_epoch": model.last_epoch,
                "train_history": model.train_history,
                "val_history": model.val_history,
                "args_history": model.args_history}
    state = model.state_dict()
    torch.save({"state": state, "metadata": metadata}, fname)


def resume(args, allow_filenotfound=True):
    model, sz = initialize_model(args)
    try:
        data = torch.load(args.resume_fname)
        metadata, state = data["metadata"], data["state"]
        model.load_state_dict(state)
        model.__dict__.update(metadata)
        resumed = True
    except FileNotFoundError:
        if not allow_filenotfound:  # conditionally raising todo(anguelos) clean this up
            raise FileNotFoundError
        resumed = False
    if resumed:
        loaded_args = last(metadata["val_history"])
        assert loaded_args.model == args.model and loaded_args.num_classes == args.num_classes
    model = model.to(args.device)
    return model


def iterate_epoch(model, dataloader, criterion, optimizer=None, desc=""):
        device = next(model.parameters()).device
        is_training = optimizer is not None
        desc += f"Training on {device}" if is_training else f"Validating on {device}"
        with ExitStack() as stack:
            if not is_training:
                model.eval()
                stack.enter_context(torch.no_grad())
            else:
                model.train()
            targets, predictions, losses = [], [], []
            for inputs, target in tqdm.tqdm(dataloader, desc=desc):
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs)
                batch_loss = criterion(output, target)
                if is_training:
                    batch_loss.mean().backward()
                    optimizer.step()
                    optimizer.zero_grad()
                targets.append(target.detach())
                predictions.append(output.detach())
                losses.append(batch_loss.detach())
            if is_training:
                model.last_epoch += 1
        targets = torch.cat(targets)
        predictions = torch.cat(predictions)
        losses = torch.cat(losses)
        return {"targets": targets, "predictions": predictions, "losses": losses, "contexts": None}


def evaluate_classifier_epoch(targets: torch.Tensor, predictions: torch.Tensor, contexts: Union[torch.Tensor, None], **kwargs):
    _, prediction_idx = predictions.max(dim=1)
    accuracy = (prediction_idx == targets).float().mean()
    confusion = torch.zeros(targets.max() + 1, targets.max() + 1)
    confusion[prediction_idx, targets] += torch.ones(prediction_idx.size(0))
    return {'Accuracy': accuracy, 'Confusion': confusion}
