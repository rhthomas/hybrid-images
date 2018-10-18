#!/usr/bin/env python3

import argparse
import cv2
import numpy as np


def define_args():
    """ Defines the arguments required to run the program.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", nargs=2, required=True,
                    help="Path to input images.")
    ap.add_argument("-k", "--kernel", nargs=2, type=int,
                    help="Kernal size, e.g. 5 7.")
    ap.add_argument("-c", "--cutoff", nargs=2, type=int, required=False,
                    help="Gaussian cutoff frequencies, e.g. 5 5.")
    ap.add_argument("-o", "--output", required=True,
                    help="Path to output image file.")
    ap.add_argument("-v", "--visual", required=False,
                    help="Path to output visualisation file.")

    # Return arguments.
    return vars(ap.parse_args())


def convolution(img, kernel):
    """ This function executes the convolution between `img` and `kernel`.
    Note: `None` indicates that there is content yet to complete.
    """
    # Load the image.
    image = cv2.imread(img)
    # Get size of image and kernel. 3rd value of shape is colour channel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    (pad_h, pad_w) = (kernel_h // 2, kernel_w // 2)
    # Create image to write to.
    output = np.zeros(image.shape)
    # Slide kernel across every pixel.
    for y in range(pad_h, image_h-pad_h):
        for x in range(pad_w, image_w-pad_w):
            # If coloured, loop for colours.
            for colour in range(image.shape[2]):
                # Get center pixel.
                center = image[y - pad_h:y + pad_h + 1,
                               x - pad_w:x + pad_w + 1,
                               colour]
                # Perform convolution and map value to [0, 255].
                # Write back value to output image.
                output[y, x, colour] = (center * kernel).sum()/255

    # Return the result of the convolution.
    return output


def fourier(img, kernel):
    # Load the image.
    image = cv2.imread(img)
    # Get size of image and kernel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    # Apply padding to the kernel.
    padded_kernel = cv2.copyMakeBorder(kernel,
                                       top=(int)(image_h-kernel_h)//2,
                                       bottom=(image_h-kernel_h)//2,
                                       left=(image_w-kernel_w)//2+1,
                                       right=(image_w-kernel_w)//2,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=[0, 0, 0])
    # Create image to write to.
    output = np.zeros(image.shape)
    # Run FFT on all 3 channels.
    for colour in range(3):
        Fi = np.fft.fft2(image[:, :, colour])
        Fk = np.fft.fft2(padded_kernel)
        # Inverse fourier.
        output[:, :, colour] = np.fft.ifft2(Fi * Fk)/255

    # Return the result of convolution.
    return output


def construct_kernels(size):
    """ Builds a dictionary of kernels based on the dimentions passed when the
    program is run. Kernels are determines by the `size` argument, which is a
    tuple from the program arguments.
    """
    kernels = {
        "smallBlur": np.ones(size, dtype="float") * (1.0 / (size[0] * size[1]))
    }
    # Return kernel dictionary.
    return kernels


def gaussian_blur(image, sigma, high):
    """ Builds a Gaussian kernel used to perform the LPF on an image.
    """
    # Calculate size of filter.
    size = 8 * sigma + 1
    if not size % 2:
        size = size + 1

    center = size//2
    kernel = np.zeros((size, size))

    # Generate Gaussian blur.
    for y in range(size):
        for x in range(size):
            diff = np.sqrt((y-center) ** 2
                           + (x - center) ** 2)
            kernel[y, x] = np.exp(-(diff ** 2)
                                  / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    if high:
        return (cv2.imread(image)/255) - convolution(image, kernel)
    else:
        return convolution(image, kernel)


def hybrid_image(image, cutoff):
    # TODO Image writes.
    # Perform low pass filter
    low = gaussian_blur(image[0], cutoff[0], 0)
    cv2.imshow('low', low)

    # Perform high pass filter
    high = gaussian_blur(image[1], cutoff[1], 1)
    cv2.imshow('high', high)

    # Return hybrid image.
    return low + high


def output_vis(image):
    """ Display hybrid image comparison for report. Visualisation shows 5 images
    reducing in size to simulate viewing the image from a distance.
    """
    for n in range(1, 5):
        None
        # Half the image each time.

    # Return the output visualisation.
    return output


def main():
    """ Main function handles execution of algorithms.
    """
    # Get arguments.
    args = define_args()
    images = args["image"]

    # Decide which algorithm to run.
    if args["kernel"] is not None:
        kSize = args["kernel"]
        # TODO Run a general algorithm.
    elif args["cutoff"] is not None:
        cutoff = args["cutoff"]
        hybrid = hybrid_image(images, cutoff)[4 * max(cutoff):-4 * max(cutoff),
                                              4 * max(cutoff):-4 * max(cutoff)]
        cv2.imshow('Hybrid', hybrid)
    else:
        print("No operation defined")

    # Hold image on display.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the main function.
if __name__ == "__main__":
    main()
