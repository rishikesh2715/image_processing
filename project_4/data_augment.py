# import cv2
# import numpy as np
# from pathlib import Path
# import random
# from tqdm import tqdm

# def augment_image(img):
#    """Apply random augmentations to the image"""
#    rows, cols = img.shape
   
#    # Random rotation (-15 to +15 degrees)
#    angle = random.uniform(-15, 15)
#    M_rotation = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
#    img = cv2.warpAffine(img, M_rotation, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
   
#    # Random scaling (0.9 to 1.1)
#    scale = random.uniform(0.9, 1.1)
#    new_width = int(cols * scale)
#    new_height = int(rows * scale)
#    img = cv2.resize(img, (new_width, new_height))
   
#    # Pad or crop to original size
#    if scale > 1:
#        # Crop center
#        start_row = (new_height - rows) // 2
#        start_col = (new_width - cols) // 2
#        img = img[start_row:start_row+rows, start_col:start_col+cols]
#    else:
#        # Pad with white
#        pad_height = (rows - new_height) // 2
#        pad_width = (cols - new_width) // 2
#        img = cv2.copyMakeBorder(img, pad_height, pad_height + (rows-new_height)%2,
#                                pad_width, pad_width + (cols-new_width)%2,
#                                cv2.BORDER_CONSTANT, value=255)
   
#    # Small affine transformation
#    pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
#    pts2 = np.float32([[random.uniform(-5,5), random.uniform(-5,5)],
#                       [cols-1+random.uniform(-5,5), random.uniform(-5,5)],
#                       [random.uniform(-5,5), rows-1+random.uniform(-5,5)]])
#    M_affine = cv2.getAffineTransform(pts1, pts2)
#    img = cv2.warpAffine(img, M_affine, (cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
   
#    return img

# def process_directory(base_path, target_per_suit=200):
#    base_path = Path(base_path)
#    suits = ['spade', 'diamond', 'heart', 'club']
   
#    for suit in suits:
#        suit_path = base_path / suit
#        if not suit_path.exists():
#            print(f"Warning: {suit_path} not found, skipping...")
#            continue
           
#        # Get all original images
#        image_files = list(suit_path.glob('*.jpg')) + list(suit_path.glob('*.png'))
#        num_orig_images = len(image_files)
       
#        if num_orig_images == 0:
#            print(f"Warning: No images found in {suit_path}, skipping...")
#            continue
           
#        # Calculate how many augmentations per original image
#        augmentations_per_image = target_per_suit // num_orig_images
       
#        print(f"\nProcessing {suit} images...")
#        pbar = tqdm(total=target_per_suit, desc=f"Generating {suit} images")
       
#        for img_path in image_files:
#            # Read and preprocess original image
#            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
#            _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
           
#            # Save augmented versions
#            for i in range(augmentations_per_image):
#                augmented = augment_image(img)
#                new_path = suit_path / f"{img_path.stem}_aug_{i}{img_path.suffix}"
#                cv2.imwrite(str(new_path), augmented)
#                pbar.update(1)
       
#        pbar.close()

# if __name__ == "__main__":
#    base_path = "dataset/suits"
#    process_directory(base_path, target_per_suit=1000)
#    print("\nAugmentation complete!")




import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

def augment_image(img):
    """Apply random augmentations to the image"""
    rows, cols = img.shape
    
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    M_rotation = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M_rotation, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # Random scaling (0.9 to 1.1)
    scale = random.uniform(0.9, 1.1)
    new_width = int(cols * scale)
    new_height = int(rows * scale)
    img = cv2.resize(img, (new_width, new_height))
    
    # Pad or crop to original size
    if scale > 1:
        # Crop center
        start_row = (new_height - rows) // 2
        start_col = (new_width - cols) // 2
        img = img[start_row:start_row+rows, start_col:start_col+cols]
    else:
        # Pad with white
        pad_height = (rows - new_height) // 2
        pad_width = (cols - new_width) // 2
        img = cv2.copyMakeBorder(img, pad_height, pad_height + (rows-new_height)%2,
                                pad_width, pad_width + (cols-new_width)%2,
                                cv2.BORDER_CONSTANT, value=255)
    
    # Small affine transformation
    pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    pts2 = np.float32([[random.uniform(-5,5), random.uniform(-5,5)],
                       [cols-1+random.uniform(-5,5), random.uniform(-5,5)],
                       [random.uniform(-5,5), rows-1+random.uniform(-5,5)]])
    M_affine = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M_affine, (cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    return img

def process_number_four(base_path, target_images=1000):
    base_path = Path(base_path)
    number_path = base_path / '4'
    
    if not number_path.exists():
        print(f"Warning: {number_path} not found!")
        return
            
    # Get all original images
    image_files = list(number_path.glob('*.jpg')) + list(number_path.glob('*.png'))
    num_orig_images = len(image_files)
    
    if num_orig_images == 0:
        print(f"Warning: No images found in {number_path}")
        return
            
    # Calculate how many augmentations per original image
    augmentations_per_image = target_images // num_orig_images
    
    print("\nProcessing number 4 images...")
    pbar = tqdm(total=target_images, desc="Generating images")
    
    for img_path in image_files:
        # Read and preprocess original image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        
        # Save augmented versions
        for i in range(augmentations_per_image):
            augmented = augment_image(img)
            new_path = number_path / f"{img_path.stem}_aug_{i}{img_path.suffix}"
            cv2.imwrite(str(new_path), augmented)
            pbar.update(1)
    
    pbar.close()

if __name__ == "__main__":
    base_path = "dataset/numbers"
    process_number_four(base_path, target_images=1000)
    print("\nAugmentation complete!")