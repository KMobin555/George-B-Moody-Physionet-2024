#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim


from PIL import Image
from collections import OrderedDict
from data_loader import get_training_and_validation_loaders
from functools import partial
from helper_code import *
from matplotlib import pyplot as plt
from classification_model_file import CLASSIFICATION_MODEL
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Callable, Optional


# testing digitization part 
# from data_loader_digit import get_training_and_validation_loaders_digit
from digitization_model_file import DIGITIZATION_MODEL
#  import DIGITIZATION_MODEL

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Classification Model Training parameters

EPOCHS = 55
CLASSIFICATION_THRESHOLD=0.45
CLASSIFICATION_DISTANCE_TO_MAX_THRESHOLD=0.1
LIST_OF_ALL_LABELS=['NORM', 'Acute MI', 'Old MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB/AFL', 'TACHY', 'BRADY'] 
RESIZE_TEST_IMAGES=(425, 650)
OPTIM_LR=1e-3
OPTIM_WEIGHT_DECAY=1e-4
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1
EPOCH_LIMIT_CLASS = 9

# Digitization Model Training parameters

DIGIT_EPOCH = 20
DIGIT_LR = 1e-3
DIGIT_WEIGHT_DECAY = 1e-4
SCHEDULER_GAMMA_DIG = 4
SCHEDULER_STEP_SIZE_DIG = 0.1
EPOCH_LIMIT_DIG = 7

PRINT_AFTER_ITR = 50

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()
    
    def forward(self, outputs, targets):

        # Flatten targets and outputs tensors
        targets_flat = targets.view(-1)  # Flatten to 1D
        outputs_flat = outputs.view(-1)  # Flatten to 1D
        
        # Move tensors from CUDA to CPU and convert to numpy arrays
        targets_np = targets_flat.cpu().detach().numpy()
        outputs_np = outputs_flat.cpu().detach().numpy()

        # print("targets model signal shape in snr after ",targets_np.shape)
        # print("outputs model signal shape  in snr after",outputs_np.shape)

        snr = compute_snr(targets_np,outputs_np)

        # Convert SNR numpy array to torch tensor and move to CUDA if needed
        snr_tensor = torch.tensor(snr, dtype=outputs.dtype, device=outputs.device, requires_grad=True)
        
        return -torch.mean(snr_tensor)  # Negative because we want to maximize SNR

    

def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the data...
    if verbose:
        print('Loading the data...')

    classification_images = list() # list of image paths
    classification_labels = list() # list of lists of strings

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        record_parent_folder=os.path.dirname(record)

        # Some images may not be labeled, so we'll exclude those
        labels = load_labels(record)
        if labels:

            # I'm imposing a further condition: the label strings should be nonempty
            nonempty_labels=[l for l in labels if l != '']
            if nonempty_labels != []:

                # Add the first image to the list
                images = get_image_files(record)
                classification_images.append(os.path.join(record_parent_folder, images[0]) )
                classification_labels.append(nonempty_labels)

    # We expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Fix an ordering of the labels
    num_classes=len(LIST_OF_ALL_LABELS)

    # Train the models.
    if verbose:
        print('Training the models on the data...')


    # Split the training set into "training" and "validation" subsets, returning them as DataLoaders
    training_loader, validation_loader \
        = get_training_and_validation_loaders(LIST_OF_ALL_LABELS, classification_images, classification_labels)
    

    #==============================================================================================================================
    # Digitization task
    #==============================================================================================================================

    print("\nTraining The Digitization Model\n")
    
    # print("classification image[0]",classification_images[0])

    base_name = classification_images[0].rsplit('_', 1)[0]

    # Append the '.hea' extension
    signal_header_path = f"{base_name}_hr.hea"
    # print(signal_header_path)

    # Load the dimensions of the signal.
    header_file = get_header_file(signal_header_path)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)
    
    digit_model = DIGITIZATION_MODEL(num_samples=num_samples, 
                                     num_leads=num_signals,
                                     img_size = RESIZE_TEST_IMAGES
                                     ).to(DEVICE)
    # print(digit_model)


    for param in digit_model.parameters(): # fine tune all the layers
        param.requires_grad = True

    criterion = SNRLoss()
    optimizer = optim.Adam(digit_model.parameters(), lr=DIGIT_LR, weight_decay=DIGIT_WEIGHT_DECAY) 
    scheduler_dig = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE_DIG, gamma=SCHEDULER_GAMMA_DIG) 
    

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    save_digitization_model(model_folder, x = (num_samples,num_signals))

    final_dig_model = None

    # Train the model
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(DIGIT_EPOCH):

        # break

        digit_model.train()
        train_loss = 0.0
        for it, (images, signals, _) in enumerate(training_loader):


            images = images.to(DEVICE)
            signals = signals.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = digit_model(images)
            loss = criterion(outputs, signals)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if it%PRINT_AFTER_ITR==0:
                print(f"Training dig-: Epoch: {epoch+1}/{DIGIT_EPOCH}, iteration: {it}/{len(training_loader)} Loss: {loss.item()}.")
            
        
        print()
        digit_model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for jt, (images, signals, _) in enumerate(validation_loader):
                images = images.to(DEVICE)
                signals = signals.to(DEVICE)

                outputs = digit_model(images)
                loss = criterion(outputs, signals)
                val_loss += loss.item()
                if jt%PRINT_AFTER_ITR==0:
                    print(f"val dig -:Epoch: {epoch+1}, Valid Iteration: {jt}/{len(validation_loader)}, Loss: {loss.item()}")
                

        avg_train_loss = train_loss / len(training_loader)
        avg_val_loss = val_loss / len(validation_loader)

        scheduler_dig.step()

        print(f'\nEpoch {epoch+1}/{DIGIT_EPOCH}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')

        no_improve +=1
        
        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            # tem_model_path = os.path.join(model_folder,'digit_tem_model.pth')
            # torch.save(digit_model.state_dict(), tem_model_path)
            final_dig_model = digit_model

            save_digitization_model(model_folder,digit_model_save = final_dig_model)
            print(f'Current best model with validation loss: {(avg_val_loss)}')

        if no_improve>EPOCH_LIMIT_DIG:
            print(f"No Improvement for {EPOCH_LIMIT_DIG} epochs ,so Terminating.")
            break
        
        print()


    
    # save_digitization_model(model_folder,digit_model = final_dig_model)
    
        

    #==============================================================================================================================
    # Classification task
    #==============================================================================================================================

    print("\nTraining The Classification Model")
    # print("Training This Model takes lot of time, Please be patient")
    print()
    

    # Initialize a model
    classification_model = CLASSIFICATION_MODEL(LIST_OF_ALL_LABELS,
                                                signal_len=(num_samples*num_signals),
                                                img_size=RESIZE_TEST_IMAGES).to(DEVICE)
    for param in classification_model.parameters(): # fine tune all the layers
        param.requires_grad = True

    loss = nn.BCELoss() # binary cross entropy loss for multilabel classification
    opt = optim.Adam(classification_model.parameters(), lr=OPTIM_LR, weight_decay=OPTIM_WEIGHT_DECAY) 
    scheduler = StepLR(opt, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA) 

    N_loss = []
    N_loss_valid = []
    train_auprc = []
    valid_auprc = []
    train_auroc= []
    valid_auroc = []
    f1_train = []
    f1_valid = []

    plot_folder=os.path.join(model_folder, "training_figures")
    os.makedirs(plot_folder, exist_ok=True)

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    save_classification_model(model_folder, list_of_classes = LIST_OF_ALL_LABELS)

    # Filename to save the final weights to
    final_weights=None

    best_val_classi = float('inf')
    no_improve_class = 0

    # Now let's train!
    for epoch in range(EPOCHS):

        # Initialization of variables for plotting the progress 
        N_item_sum = 0 
        N_item_sum_valid = 0 
        targets_train = []
        outputs_train = []
        targets_valid = []
        outputs_valid = []
        
        ### Training part
        if verbose:
            print(f"============================[{epoch}]============================")
        classification_model.train()
        for i, (image, signals, label) in enumerate(training_loader):
            opt.zero_grad()

            image = image.float().to(DEVICE)
            signals = signals.to(DEVICE)
            label = label.to(torch.float).to(DEVICE)

        #  print(f'ima shape {image.shape} , signal shape {signals.shape}')
            prediction = classification_model(image,signals)
            
            # print(prediction,prediction.shape)
            # print(label,label.shape)
            # loss
            N = loss(prediction,label) 
            N.backward()
            N_item = N.item()
            N_item_sum += N_item

            # gradient clipping plus optimizer
            torch.nn.utils.clip_grad_norm_(classification_model.parameters(), max_norm=10)
            opt.step()

            if i%PRINT_AFTER_ITR==0:
                print(f"Training cls - Epoch: {epoch+1}, Iteration: {i}/{len(training_loader)}, Loss: {N_item}")
            
            targets_train.append(label.data.cpu().numpy()) #target[:,0]
            outputs_train.append(prediction.data.cpu().numpy())


        ### Validation part
        print()
        classification_model.eval()
        with torch.inference_mode():
            for j, (image, signals, label) in enumerate(validation_loader):
                image = image.float().to(DEVICE)
                signals = signals.to(DEVICE)
                label = label.to(torch.float).to(DEVICE)
                prediction = classification_model(image,signals)
                
                N = loss(prediction,label)
                N_item = N.item()
                N_item_sum_valid += N.item()

                targets_valid.append(label.data.cpu().numpy()) #target[:,0]
                outputs_valid.append(prediction.data.cpu().numpy())

                if j%PRINT_AFTER_ITR==0:
                    print(f"Val cls - Epoch: {epoch+1}, Valid Iteration: {j}, Loss: {N_item}")
        
        avg_train_loss_class = N_item_sum/len(training_loader)
        avg_val_loss_classi = N_item_sum_valid/len(validation_loader)
        print(f'\nEpoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss_class}, Val Loss: {avg_val_loss_classi} ')

        scheduler.step()

        # Logging the outputs and targets to caluclate auprc and auroc
        targets_train = np.concatenate(targets_train, axis=0).T
        outputs_train = np.concatenate(outputs_train, axis=0).T
        targets_valid = np.concatenate(targets_valid, axis=0).T
        outputs_valid = np.concatenate(outputs_valid, axis=0).T

        auprc_t = average_precision_score(y_true=targets_train, y_score=outputs_train)
        auroc_t = roc_auc_score(y_true=targets_train, y_score=outputs_train)
        auprc_v = average_precision_score(y_true=targets_valid, y_score=outputs_valid)
        auroc_v = roc_auc_score(y_true=targets_valid, y_score=outputs_valid)

        train_auprc.append(auprc_t)
        train_auroc.append(auroc_t)
        valid_auprc.append(auprc_v)
        valid_auroc.append(auroc_v)
        
        N_loss.append(N_item_sum/i)
        N_loss_valid.append(N_item_sum_valid/j)
        
        # saving loss function after each epoch so you can look on progress
        fig = plt.figure()
        plt.plot(N_loss, label="train")
        plt.plot(N_loss_valid, label="valid")
        plt.title("Loss function")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(plot_folder, "loss.png"))
        plt.close()

        fig = plt.figure()
        plt.plot(train_auprc, label="train auprc")
        plt.plot(valid_auprc, label="valid auprc")
        plt.plot(train_auroc, label="train auroc")
        plt.plot(valid_auroc, label="valid auroc")
        
        plt.title("AUPRC and AUROC")
        plt.xlabel('epoch')
        plt.ylabel('Performace')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(plot_folder, "auroc_auprc.png"))
        plt.close()

        no_improve_class +=1
        
        if avg_val_loss_classi < best_val_classi:
            best_val_classi = avg_val_loss_classi
            ### save model after each epoch
            # file_path = os.path.join(model_folder, "classi_model_weights.pth")
            # torch.save(classification_model.state_dict(), file_path)
            no_improve_class = 0
            # If this is the last epoch, then the weights of the model will be saved to this file
            # final_weights = classification_model
            # Save the models.
            save_classification_model(model_folder, final_weights = classification_model)
            print(f'Current best model with validation loss: {(avg_val_loss_classi)}')
        
        if no_improve_class > EPOCH_LIMIT_CLASS:
            print(f"No Improvement for {EPOCH_LIMIT_CLASS} epochs ,so Terminating.")
            break
        
        print()

    # # Save the models.
    # save_classification_model(model_folder, final_weights = final_weights)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this
# function to add your code, but do *not* change the arguments of this
# function. If you do not train one of the models, then you can return None for
# the model.
def load_models(model_folder, verbose):

    digit_filename = os.path.join(model_folder, 'digit_info.txt')
    loaded_data = joblib.load(digit_filename)
    loaded_num_samples = int(loaded_data['num_samples'])  # Convert back to integer if needed
    loaded_sum_signals = int(loaded_data['num_signals'])

    digitization_model = DIGITIZATION_MODEL(num_samples=loaded_num_samples, 
                                     num_leads=loaded_sum_signals,
                                     img_size = RESIZE_TEST_IMAGES
                                     ).to(DEVICE)
    digitization_filename = os.path.join(model_folder, "digitization_model.pth")
    digitization_model.load_state_dict(torch.load(digitization_filename))

    classes_filename = os.path.join(model_folder, 'classes.txt')
    classes = joblib.load(classes_filename)

    classification_model = CLASSIFICATION_MODEL(classes,((loaded_num_samples*loaded_sum_signals)),RESIZE_TEST_IMAGES).to(DEVICE) # instantiate a new copy of the model
    classification_filename = os.path.join(model_folder, "classification_model.pth")
    classification_model.load_state_dict(torch.load(classification_filename))

    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):

    # Run the digitization model; if you did not train this model, then you can set signal = None.
    signal = None

    # Run the classification model.
    classes = classification_model.list_of_classes

    # Open the image:
    record_parent_folder=os.path.dirname(record)
    image_files=get_image_files(record)
    image_path=os.path.join(record_parent_folder, image_files[0])
    img = Image.open(image_path)
    # FIXME: repeated code---maybe factor out opening the image from a record
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # transform the image and make it suitable as input
    img = transforms.Resize(RESIZE_TEST_IMAGES)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    img = img.unsqueeze(0)

    # send it to the GPU if necessary
    img = img.float().to(DEVICE)

    if digitization_model is not None:
        digitization_model.eval()
        with torch.inference_mode():
            signal = digitization_model(img).squeeze().cpu().numpy()
            num_samples = digitization_model.num_samples
            num_leads = digitization_model.num_leads
            # print("running model signal shape",signal.shape)
            signal = signal.reshape(num_samples, num_leads)

    # print(f'signal shape {signal.shape}')
    cls_signal = signal.reshape(num_samples*num_leads)
    # print(f'signal shape cls {cls_signal.shape}')
    if isinstance(cls_signal, np.ndarray):
        cls_signal = torch.tensor(cls_signal, dtype=torch.float32)
    
    cls_signal = cls_signal.unsqueeze(0)
    cls_signal = cls_signal.to(DEVICE)
    # print(f'signal shape cls 2 {cls_signal.shape}')

    classification_model.eval()
    with torch.inference_mode():
        probabilities = torch.squeeze(classification_model(img,cls_signal), 0).tolist()
        predictions=list()
        for i in range(len(classes)):
            if probabilities[i] >= CLASSIFICATION_THRESHOLD:
                predictions.append(classes[i])

    # backup if none is over the threshold: use the max
    if predictions==[]:
        highest_probability=max(probabilities)
        for i in range(len(classes)):
            if abs(highest_probability - probabilities[i]) <= CLASSIFICATION_DISTANCE_TO_MAX_THRESHOLD:
                predictions.append(classes[i])


    return signal, predictions

#########################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
#########################################################################################

# Extract features.
def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained models.
def save_classification_model(model_folder,
                list_of_classes=None,
                final_weights=None):
    
    if list_of_classes is not None:

        classes=filename = os.path.join(model_folder, 'classes.txt')
        joblib.dump(list_of_classes, filename, protocol=0)

    if final_weights is not None:
        # copy the file with the final weights to the model path
        model_filename=os.path.join(model_folder, "classification_model.pth")
        # shutil.copyfile(final_weights, model_filename)
        torch.save(final_weights.state_dict(),model_filename)


def save_digitization_model(model_folder,
                            digit_model_save=None,
                            x=None):
    
    if x is not None:
        file_path = os.path.join(model_folder, 'digit_info.txt')
        # Save variables as strings
        data_to_save = {'num_samples': str(x[0]), 'num_signals': str(x[1])}
        joblib.dump(data_to_save, file_path)

    if digit_model_save is not None:
        model_filename = os.path.join(model_folder, 'digitization_model.pth')
        torch.save(digit_model_save.state_dict(), model_filename)