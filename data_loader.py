from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split

from helper_code import *

#===================================
# Parameters (configure stuff here)
#===================================

RESIZE_TO_DIMENSIONS=(425, 650)                          # resize all images to these dimensions during training
IMAGE_MODE='RGB'                                            # set all images to this mode
RANDOM_STATE=42                                             # number for repeatable pseudorandomness

#=========
# Classes
#=========


class ECGImageDataset(Dataset):
    """Map-style dataset that yields (image, label) pairs. In this context,
       each "label" in the (image, label) tuple will really be a vector encoding
       multiple dx labels, while each "image" will be an image path.

       Initialize this with 
        - a list of all possible class labels in a fixed ordering,
        - a boolean value saying whether this is a training set or not,
        - a list of paths (as strings), and
        - a list of lists of labels (as strings).
       """
    
    def __init__(self, list_of_all_classes:list, is_training:bool,
                    image_paths:list, image_labels:list):

        self.list_of_all_classes=list_of_all_classes
        self.num_classes = len(self.list_of_all_classes)
        # Inverse of list_of_all_classes: look up index by name
        self.class_to_index=dict()
        for i in range(self.num_classes):
            self.class_to_index[list_of_all_classes[i]]=i

        self.is_training=is_training

        self.image_paths=image_paths
        self.image_labels=image_labels
        self.signals = {}
           
        # Define transformations
        self.transform_images = transforms.Compose([
            transforms.Resize(RESIZE_TO_DIMENSIONS),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if self.is_training else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if self.is_training else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(degrees=8) if self.is_training else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop(size=RESIZE_TO_DIMENSIONS, scale=(0.8, 1.0), ratio=(0.75, 1.33)) if self.is_training else transforms.Lambda(lambda x: x),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)) if self.is_training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for image_file in self.image_paths:
            signal_id = os.path.basename(image_file).split('_')[0]
            signal_path = os.path.join(os.path.dirname(image_file), f"{signal_id}_hr")
            signals, fields = load_signals(signal_path)
            self.signals[signal_id] = signals  # Assuming single-channel signal

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """index --> (image, label_vector)."""

        image_file = self.image_paths[idx]
        signal_id = os.path.basename(image_file).split('_')[0]
        # First the image
        our_image = Image.open(image_file)
        if our_image.mode != IMAGE_MODE: # e.g. RGB
            our_image = our_image.convert(IMAGE_MODE)
        our_image = self.transform_images(our_image)

        # Next the labels

        #     Our dataset will encode each label not as a list of strings but as a vector,
        #     one index for each class. Thus we have to convert a list like
        #       ["THING1", "THING2", "THING3"]
        #     into a vector like 
        #       [0,1,0,1,1,0].

        label_strings = self.image_labels[idx]
        our_label_vector = [0] * self.num_classes
        for l in label_strings: # labels assigned to this index
            our_label_vector[self.class_to_index[l]] = 1 

        signal = torch.from_numpy(self.signals[signal_id]).float()
        signal = signal.flatten()  # Flatten the signal to one dimension

        return our_image, signal, our_label_vector
    
    @staticmethod
    def collate_fn(batch):
        """Batch of pairs -> pair of tensors representing the batch"""
        images, signals, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        signals = torch.stack(signals, dim=0)
        labels = torch.as_tensor(labels)
        return images, signals, labels


def get_training_and_validation_loaders(list_of_all_classes, image_path_list, label_names_list):
    """
    Given  a list `list_of_all_classes` of all class labels in the dataset,
    `image_path_list` (a list of image paths), and `label_names_list` (a list
    of _lists_ of label names corresponding to the images), return the pair
    `(training_loader, validation_loader)` which can be used to train/validate
    a model.

    """

    # Divide the dataset into training and validation sets

    test_size = 0.2

    training_images, validation_images, \
    training_classes, validation_classes, \
        = train_test_split(image_path_list,
                           label_names_list,
                           test_size=test_size, 
                           random_state=RANDOM_STATE,
                           shuffle=True)

    # Dataset for training
    train_dataset=ECGImageDataset(list_of_all_classes=list_of_all_classes,
                                  is_training=True,
                                  image_paths=training_images,
                                  image_labels=training_classes)

    # Dataset for training
    validation_dataset=ECGImageDataset(list_of_all_classes=list_of_all_classes,
                                       is_training=False,
                                       image_paths=validation_images,
                                       image_labels=validation_classes)
    
    # print("train dataset"+train_dataset[0])

    # Dataloader for training
    batch_size = 4

    training_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=4,
                                                  drop_last=True,
                                                  collate_fn=train_dataset.collate_fn)
    
    # DataLoader for validation
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False, # note this differs
                                                    pin_memory=True,
                                                    num_workers=4,
                                                    drop_last=True,
                                                    collate_fn=validation_dataset.collate_fn)
    
    return training_loader, validation_loader

