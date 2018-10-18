#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def define_args():
    """ Define arguments.

    Defines the arguments required to run the program. Call first in main().
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", nargs=2, required=True,
                    help="Path to input images (2 required).")
    ap.add_argument("-k", "--kernel", required=True,
                    help="Kernal size, e.g. 5x7.")
    ap.add_argument("-o", "--output", required=True,
                    help="Path to output image file.")
    ap.add_argument("-v", "--visual", required=False,
                    help="Path to output visualisation file.")
    # Return arguments.
    return vars(ap.parse_args())


def convolution(img, kernel):
    """ Run convolution.

    This function executes the convolution between `img` and `kernel`.
    Note: `None` indicates that there is content yet to complete.
    """
    # Display the image.
    image = cv2.imread(img)
    cv2.imshow('Input image', image)
    # Get size of image and kernel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    (pad_h, pad_w) = (kernel_h // 2, kernel_w // 2)
    # Create image to write to.
    output = np.zeros(image.shape)
    # Slide kernel across every pixel.
    for y in range(pad_h, image_h-pad_h):
        for x in range(pad_w, image_w-pad_w):
            for colour in range(3):
                # Get center pixel.
                center = image[y - pad_h:y + pad_h + 1,
                               x - pad_w:x + pad_w + 1,
                               colour]
                # Perform convolution.
                conv = (center * kernel).sum()
                # Write back value to output image.
                output[y, x, colour] = conv

    # Return the result of the convolution.
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


def construct_kernels(size):
    """ Build kernels.

    Builds a dictionary of kernels based on the dimentions passed when the
    program is run. Kernels are determines by the `size` argument, which is a
    tuple from the program arguments.
    """
    kernels = {
        "smallBlur" : np.ones(size, dtype="float")
                              * (1.0 / (size[0] * size[1]))
    }
    # Return kernel dictionary.
    return kernels


def low_pass(image):
    """ Perform LPF.

    Returns low pass version of the input `image`.
    """
    # Perform convolution with bluring kernel.
    img = convolution(image, blur)
    # Return result.
    return img


def high_pass(image):
    """ Perform HPF.

    Returns low pass version of the input `image`.
    """
    # Compute low pass of image.
    img = low_pass(image)
    # Subtract from starting image.
    img = image - img
    # Return result.
    return img


def output_vis(image):
    """ Produce visualisation.

    Display hybrid image comparison for report. Visualisation shows 5 images
    reducing in size to simulate viewing the image from a distance.
    """

    # Return the output visualisation.
    return output


def main():
    """ Main function.

    This is the main function. Here the image is loaded into an array, along
    with the kernel size argument, the convolution is performed and the
    resulting image is displayed.
    """
    # Get arguments.
    args = define_args()
    # Split kernel size.
    (kW, kH) = tuple(map(int, args["kernel"].split('x')))
    # Run convolution.
    images = args["image"]
    # Create kernels.
    kernels = construct_kernels((kW, kH))
    # Show convolution result.
    cv2.imshow('Result of convolution',
               convolution(images[0], kernels["smallBlur"]))
    # Hold image on display.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the main function.
if __name__ == "__main__":
    main()
