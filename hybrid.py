#!/usr/bin/env python3
# pip install -r requirements.txt
# ./hybrid.py -i data/dog.bmp data/cat.bmp -c 4 4 -o hybrid.jpg -v visual.jpg

import argparse
import cv2
import numpy as np
import os


def define_args():
    """ Defines the arguments required to run the program.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", nargs=2, required=True,
                    help="Path to input images.")
    ap.add_argument("-k", "--kernel", nargs=2, type=int,
                    help="Kernal size, e.g. 5 7. Note: first image in list will\
                    be used.")
    ap.add_argument("-c", "--cutoff", nargs=2, type=int,
                    help="Gaussian cutoff frequencies, e.g. 5 5.")
    ap.add_argument("-o", "--output", required=False,
                    help="Path to output image file.")
    ap.add_argument("-v", "--visual", required=False,
                    help="Path to output visualisation file.")
    ap.add_argument("-f", "--fourier", default=False, action='store_true',
                    help="Use Fourier convolution.")
    ap.add_argument("-s", "--sobel", default=False, action='store_true',
                    help="Run Sobel edge detection on the first image.")

    # Return arguments.
    return vars(ap.parse_args())


def convolution(img, kernel):
    """ This function executes the convolution between `img` and `kernel`.
    """
    print("[" + img + "]\tRunning convolution...\n")
    # Load the image.
    image = cv2.imread(img)
    # Flip template before convolution.
    kernel = cv2.flip(kernel, -1)
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
                output[y, x, colour] = (center * kernel).sum() / 255

    # Return the result of the convolution.
    return output


def fourier(img, kernel):
    # Load the image.
    image = cv2.imread(img)
    # Get size of image and kernel.
    (image_h, image_w) = image.shape[:2]
    (kernel_h, kernel_w) = kernel.shape[:2]
    # Apply padding to the kernel.
    padded_kernel = np.zeros(image.shape[:2])
    start_h = (image_h - kernel_h) // 2
    start_w = (image_w - kernel_w) // 2
    padded_kernel[start_h:start_h + kernel_h,
                  start_w:start_w + kernel_w] = kernel
    # Create image to write to.
    output = np.zeros(image.shape)
    # Run FFT on all 3 channels.
    for colour in range(3):
        Fi = np.fft.fft2(image[:, :, colour])
        Fk = np.fft.fft2(padded_kernel)
        # Inverse fourier.
        output[:, :, colour] = np.fft.fftshift(np.fft.ifft2(Fi * Fk)) / 255

    # Return the result of convolution.
    return output


def gaussian_blur(image, sigma):
    """ Builds a Gaussian kernel used to perform the LPF on an image.
    """
    print("[" + image + "]\tCalculating Gaussian kernel...")
    # Calculate size of filter.
    size = 8 * sigma + 1
    if not size % 2:
        size = size + 1

    center = size // 2
    kernel = np.zeros((size, size))

    # Generate Gaussian blur.
    for y in range(size):
        for x in range(size):
            diff = (y - center) ** 2 + (x - center) ** 2
            kernel[y, x] = np.exp(-diff / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)

    if use_f:
        return fourier(image, kernel)
    else:
        return convolution(image, kernel)


def low_pass(image, cutoff):
    """ Generate low pass filter of image.
    """
    return gaussian_blur(image, cutoff)


def high_pass(image, cutoff):
    """ Generate high pass filter of image. This is simply the image minus its
    low passed result.
    """
    return (cv2.imread(image)/255) - low_pass(image, cutoff)


def hybrid_image(image, cutoff):
    """ Create a hybrid image by summing together the low and high freqency
    images.
    """
    # Perform low pass filter and export.
    print("[" + image[0] + "]\tGenerating low pass image...")
    low = low_pass(image[0], cutoff[0])
    cv2.imwrite('low.jpg', low * 255)
    # Perform high pass filter and export.
    print("[" + image[1] + "]\tGenerating high pass image...")
    high = high_pass(image[1], cutoff[1])
    cv2.imwrite('high.jpg', (high + 0.5) * 255)
    # Return hybrid image.
    print("Creating hybrid image...")
    return low + high


def output_vis(image):
    """ Display hybrid image comparison for report. Visualisation shows 5 images
    reducing in size to simulate viewing the image from a distance.
    """
    print("Creating visualisation...")
    # Local variables.
    num = 5  # Number of images to display.
    gap = 2  # Gap between images (px).

    # Create list of images.
    image_list = [image]
    max_height = image.shape[0]
    max_width = image.shape[1]
    # Add images to list and increase max width.
    for i in range(1, num):
        tmp = cv2.resize(image, (0, 0), fx=0.5 ** i, fy=0.5 ** i)
        max_width += tmp.shape[1] + gap
        image_list.append(tmp)

    # Create space for image stack.
    stack = np.ones((max_height, max_width, 3)) * 255
    # Add images to stack.
    current_x = 0
    for img in image_list:
        stack[max_height-img.shape[0]:,
              current_x:img.shape[1] + current_x, :] = img
        current_x += img.shape[1] + gap

    # Return the result.
    return stack


def main():
    """ Main function handles execution of algorithms.
    """
    # Get arguments.
    args = define_args()
    images = args["image"]
    # Create global flag to set convolution method.
    global use_f
    use_f = args["fourier"]

    # Decide which algorithm to run.
    if args["kernel"] is not None:
        kSize = args["kernel"]
        if any(s % 2 == 0 for s in kSize):
            print("Kernel dimentions must be odd!")
            exit()
        kernel = np.ones(kSize, dtype="float") * (255.0 / (kSize[0] * kSize[1]))
        result = convolution(images[0], kernel)
        cv2.imwrite(args["output"], result)
    elif args["cutoff"] is not None:
        cutoff = args["cutoff"]
        if use_f:
            # No need to crop the fourier image.
            hybrid = hybrid_image(images, cutoff)
        else:
            hybrid = hybrid_image(images,
                                  cutoff)[4 * max(cutoff):-4 * max(cutoff),
                                          4 * max(cutoff):-4 * max(cutoff)]
        # Save resulting images.
        cv2.imwrite(args["output"], hybrid * 255)
        if args["visual"] is not None:
            # Only save visualisation if requested.
            cv2.imwrite(args["visual"], output_vis(hybrid) * 255)
    elif args["sobel"]:
        # Run Sobel edge detection.
        sobel_x = fourier(images[0], 255 * np.array([[1, 0, -1],
                                               [2, 0, -2],
                                               [1, 0, -1]]))
        sobel_y = fourier(images[0], 255 * np.array([[1, 2, 1],
                                               [0, 0, 0],
                                               [-1, -2, -1]]))
        # Save resulting images.
        cv2.imwrite("sobel_x.jpg", sobel_x)
        cv2.imwrite("sobel_y.jpg", sobel_y)
        cv2.imwrite("sobel_xy.jpg", sobel_x + sobel_y)
    else:
        print("No operation defined")
        exit()

    print("Done.")


# Call the main function.
if __name__ == "__main__":
    main()
