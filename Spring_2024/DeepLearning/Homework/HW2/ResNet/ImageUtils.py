import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        # Resize the image to add four extra pixels on each side.
        height, width, channels = image.shape
        extra_pixels = 2
        
        # -me Calculate new dimensions with extra pixels
        pad_width = width + 2 * extra_pixels
        pad_height = height + 2 * extra_pixels
        
        # -me Create a new blank image with the new dimensions
        pad_image = np.zeros((pad_height, pad_width, channels), dtype=image.dtype)
        
        # -me Paste the original image onto the new image with extra pixels border
        pad_image[extra_pixels:pad_height-extra_pixels, extra_pixels:pad_width-extra_pixels, :] = image

        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        crop_size = [32, 32]        
        max_y = pad_height - crop_size[0]
        max_x = pad_width - crop_size[1]
        start_y = np.random.randint(0, max_y + 1)
        start_x = np.random.randint(0, max_x + 1)
        
        # -me Crop the image
        image = pad_image[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]]

        # Randomly flip the image horizontally.
        flip = np.random.rand() < 0.5
    
        # -me If flip is True, horizontally flip the image
        if flip:
            image = np.fliplr(image)

    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    image = (image - mean) / (std + 1e-7)   # Adding a small value to avoid division by zero

    return image