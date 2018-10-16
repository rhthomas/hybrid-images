#!/usr/bin/env python3

import argparse
import cv2
import numpy as np


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

    return vars(ap.parse_args())


def convolution(image, kernel):
    """ Run convolution.

    This function executes the convolution between `image` and `kernel`.
    Note: `None` indicates that there is content yet to complete.
    """
    # Get size of image and kernel.

    # Create image to write to.
    output = None
    # Slide kernel across every pixel.
    for y in None:
        for x in None:
            # Get center pixel.
            center = None
            # Perform convolution.
            conv = (center * kernel).sum()
            # Write back value to output image.
            output[None] = k

    # Return the result of the convolution.
    return output


def construct_kernels(size):
    """ Build kernels.

    Builds a dictionary of kernels based on the dimentions passed when the
    program is run. Kernels are determines by the `size` argument, which is a
    tuple from the program arguments.
    """
    # Return kernel dictionary.
    return kernels


def low_pass(image):
    """ Perform LPF.

    Returns low pass version of the input `image`.
    """

    # Return result.
    return img


def low_pass(image):
    """ Perform HPF.

    Returns low pass version of the input `image`.
    """
    # Compute low pass of image.

    # Subtract from starting image.

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
    args = define_args()
    print(args)


# Call the main function.
if __name__ == "__main__":
    main()
