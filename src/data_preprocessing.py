# Data loading and preprocessing
from src.utils import load_config

import xml.etree.ElementTree as ET
import pandas as pd
import os
import cv2
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

def parse_xml(file_path):
    """Parses an XML annotation file and returns a list of annotations."""
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        filename = root.find('filename').text

        annotations = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            coordinates = {
                'xmin': int(bbox.find('xmin').text),
                'ymin': int(bbox.find('ymin').text),
                'xmax': int(bbox.find('xmax').text),
                'ymax': int(bbox.find('ymax').text)
            }
            annotations.append({'filename': filename, 'label': label, 'bbox': coordinates})

        return annotations

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")  
        return []  
    except ET.ParseError:
        print(f"Error: XML parse error: {file_path}")  
        return []  
    except AttributeError as e:
        print(f"Error: Missing element in XML: {file_path}. Details: {e}") 
        return []
    

def process_annotations(annotation_path):
    """Parses XML annotations and returns a Pandas DataFrame."""

    data = []
    for filename in os.listdir(annotation_path):
        if filename.endswith('.xml'):  # Check for .xml extension explicitly
            file_path = os.path.join(annotation_path, filename)
            annotations = parse_xml(file_path) 

            for annotation in annotations:
                data.append([annotation['filename'], annotation['label'], annotation['bbox']])

    return pd.DataFrame(data, columns=['filename', 'label', 'bbox'])


def get_augmentation_transform(p=0.5):
    """Creates a PyTorch transformation pipeline for data augmentation."""
    transform_list = []

    transform_list.extend([
        T.RandomApply([
            T.RandomRotation(degrees=25),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0.1,0.1)),
            T.RandomHorizontalFlip(p=0.5)
        ], p=p),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return T.Compose(transform_list)


class FaceMaskDataset(Dataset):
    def __init__(self, df, image_path, target_size=(224, 224), augment=False):
        self.df = df
        self.image_path = image_path
        self.target_size = target_size
        self.augment = augment
        self.transform = get_augmentation_transform() if augment else T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bbox = row['bbox']
        image_path_full = os.path.join(self.image_path, row['filename'])
        image = cv2.imread(image_path_full)

        if image is None:
            print(f"Error: Could not read image {image_path_full}")
            return None

        cropped_image = image[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
        resized_image = cv2.resize(cropped_image, self.target_size)

        pil_image = Image.fromarray(resized_image)
        image_tensor = self.transform(pil_image) 

        config = load_config()
        if not config:
            exit("Failed to load configuration. Exiting.")
        LABEL_MAP = config['label_map']
        label = LABEL_MAP[row['label']]

        return image_tensor, label

