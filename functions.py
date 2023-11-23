import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import numpy
import random
from pprint import pprint
from collections import defaultdict


def modify_data(original_dataset):
    return ModifiedDataset(original_dataset, mel_spectro, fixed_sample_rate, num_samples, labels)


def train_epoch(model, trainloader, optim, criterion, device, epoch, train_losses, train_accuracies):

    train_loss = 0
    correct = 0

    model.train()

    for i, (wf, sr, labels, transform) in enumerate(trainloader):
        transform, labels = transform.to(device), labels.to(device)

        optim.zero_grad()
        preds = model(transform)
        loss = criterion(preds, (labels))
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        
        _, predicted = preds.max(1)
        correct += predicted.eq(labels).sum().item()
        
        if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(transform), len(trainloader.dataset),
                    100. * i / len(trainloader), loss.item()))


    average_loss = train_loss / len(trainloader)
    accuracy = correct / len(trainloader.dataset) * 100.0
    
    train_losses.append(average_loss)
    train_accuracies.append(accuracy)

    print(f"epoch n.{epoch} : Average Loss = {average_loss}, Accuracy = {accuracy:.2f}%")
    
    return average_loss, accuracy
 

def train_epochs(model, trainloader, valloader, optim, train_criterion, val_criterion, device, n_epochs, saving_path, fold, patience=5, min_improvement=0.001):
    
    best_val_loss_overall = float('inf')
#     best_model_state = None
    counter = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    
    for epoch in range(n_epochs):
        
        print(f'Starting epoch {epoch}')

        train_loss, train_accuracy = train_epoch(model, trainloader, optim, train_criterion, device, epoch, train_losses, train_accuracies)
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_correct = 0
            val_total = 0

        
            for i, (wf, sr, labels, transform) in enumerate(valloader):
                transform, labels = transform.to(device), labels.to(device)

                preds = model(transform)
                val_loss += val_criterion(preds, labels).item()

                _, predicted = preds.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

            average_val_loss = val_loss / len(valloader)
            val_accuracy = val_correct / len(valloader.dataset) * 100.0

            val_losses.append(average_val_loss)
            val_accuracies.append(val_accuracy)
            
            model.train()
            
            # Check if validation loss has improved
            if best_val_loss_overall - val_loss > min_improvement:
                best_val_loss_overall = val_loss
                counter = 0
                best_model_state_overall = model.state_dict()

            else:
                counter += 1

            if counter >= patience:
                print("Early stopping! No improvement for", patience, "epochs.")
                break
                 
#         torch.save(best_model_state, saving_path)

        print(f"epoch n.{epoch} : Val Average Loss = {average_val_loss}, Val Accuracy = {val_accuracy:.2f}%")
    
    
    # Accuracy over the validation split (with the final model after all epochs)
    correct, total = 0, 0
    with torch.no_grad():
        for i, (wf, sr, labels, transform) in enumerate(valloader): 
            transform, labels = transform.to(device), labels.to(device)
            preds = model(transform)
            _, predicted = preds.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
                
        # Print accuracy over all fold
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)
         

    print("Training finished.")
    # Return the learning curves
    return train_losses, train_accuracies, val_losses, val_accuracies

    

def test_model(model, testloader, criterion, device):
    test_loss = 0
    correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for wvf, sr, target, transform in tqdm(testloader):
            transform, target = transform.to(device), target.to(device)
            output = model(transform)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Compute accuracies by class
            for label, prediction in zip(target.numpy(), pred.numpy().flatten()):
                class_correct[label] += int(label == prediction)
                class_total[label] += 1

    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
    test_loss, correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))
    
    print("Accuracy by class:")
    for i in range(len(class_correct)):
        accuracy = 100. * class_correct[i] / class_total[i]
        print(f"Class {i}: {accuracy:.2f}%")

    return test_loss, accuracy


def predict(model, input_data, target):
    model.eval()
    with torch.no_grad():
        output = model(input_data.unsqueeze(dim=0))
        pred = output.argmax(dim=1, keepdim=True)
        predicted_label = pred.item()
        expected_label = target
    return predicted_label, expected_label


### --------------------------------------------------------------------------------------------------------------- ###

# functions to plot spectrograms

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()
    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    
### --------------------------------------------------------------------------------------------------------------- ###

# Visualize class distributions

def class_distrib(dataset, dataset_name):
    class_distribution = {}
    total_samples = len(dataset)

    for i, (_, _, label, _) in tqdm(enumerate(dataset)):
#         label = label.item()
        if label in class_distribution:
            class_distribution[label] += 1
        else:
            class_distribution[label] = 1

    class_distribution_percent = {label: count / total_samples * 100 for label, count in class_distribution.items()}
    # class_distribution = dict(sorted(class_distribution.items()))
    class_weights = {label: total_samples / (len(class_distribution) * count) for label, count in class_distribution.items()}
    class_weights = dict(sorted(class_weights.items()))

    plt.figure(figsize=(10, 6))
    plt.bar(class_distribution_percent.keys(), class_distribution_percent.values())
    plt.xlabel("Class Label")
    plt.xticks(range(4))
    plt.ylabel("Percentage of Samples")
    plt.title(f"Class Distribution in {dataset_name} (in %)")
    plt.show()

    for key in class_distribution_percent:
        class_distribution_percent[key] = round(class_distribution_percent[key])

    pprint(class_distribution_percent)
    return class_distribution_percent, class_weights


# for less computations (approximation of the weights)
def class_distrib_approx(dataset, num_samples, dataset_name):
    class_distribution = {}

    # Sample a subset of data
    sample_indices = random.sample(range(len(dataset)), num_samples)
    for i in tqdm(sample_indices):
        _, _, label, _ = dataset[i]
        label = label.item()
        if label in class_distribution:
            class_distribution[label] += 1
        else:
            class_distribution[label] = 1

    total_samples = num_samples
    class_weights = {label: total_samples / (len(class_distribution)*class_count) for label, class_count in class_distribution.items()}
    class_weights = dict(sorted(class_weights.items()))

    class_distribution_percent = {label: count / total_samples * 100 for label, count in class_distribution.items()}
 
    plt.figure(figsize=(10, 6))
    plt.bar(class_distribution_percent.keys(), class_distribution_percent.values())
    plt.xlabel("Class Label")
    plt.ylabel("Percentage of Samples")
    plt.title(f"Class Distribution in {dataset_name} (in %)")
    plt.show()

    return class_distribution_percent, class_weights


# To convert the class weights to tensors
def class_weights_tensor(class_weights_dict, device) :
    weights_list = [class_weights_dict[i] for i in range(len(class_weights_dict))]
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)
    
    return class_weights_tensor




def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
#             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
