# Data visualizations

import cv2
import matplotlib.pyplot as plt


def visualize_annotations(image_path, annotations):
    """Visualizes annotations on an image.

    Args:
        image_path: Path to the image file.
        annotations: A list of annotation dictionaries, where each dictionary
                     contains 'filename', 'label' and 'bbox' keys.  'bbox' should be a
                     dictionary with 'xmin', 'ymin', 'xmax', and 'ymax' keys.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Check if annotations is empty before looping
        if not annotations:
          print("No annotations found for this image.")
          plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          plt.title(f"Image: {image_path.split('/')[-1]}") # Set title to image name
          plt.axis('off') 
          plt.show()
          return 

        for ann in annotations:
            bbox = ann['bbox']
            label = ann['label']
            filename = ann['filename']

            # Check for valid bounding box coordinates
            if all(coordinates >= 0 for coordinates in bbox.values()) and bbox['xmin'] < bbox['xmax'] and bbox['ymin'] < bbox['ymax']:
                cv2.rectangle(image, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (0, 255, 0), 1)
                cv2.putText(image, label, (bbox['xmin'], bbox['ymin'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                print(f"Warning: Invalid bounding box coordinates for label '{label}' in image {image_path}")

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Image: {filename}") 
        plt.axis('off') 
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")