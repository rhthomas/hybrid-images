# Hybrid Images

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Hybrid images demonstrate how certain frequencies dominate at certain distances.
Most notably, high frequencies dominate when the object is close you you, while lower frequencies dominate when it is further away.

We can demonstrate the effect of hybrid images by applying a Gaussian blur to an image in order to produce a low-pass filtered equivalent.

![cat](./data/dog.bmp) ![cat hpf](./Output/low.jpg)

For a different (but aligned) image we again Gaussian blur, but then subtract the low-pass image from the original in order to produce a high pass image.

![cat](./data/cat.bmp) ![cat hpf](./Output/high.jpg)

These two images can then be summed together to produce a hybrid image as shown below.
The image is downsampled to give the effect of moving the object closer/further away from your view point.

![Hybrid effect](./Output/visual.jpg)

## Installation

(Optional) Setup a virtual environment to install the necessary packages.

```bash
virtualenv .venv
source .venv/bin/activate
```

Install the packages listed in the requirements file.

```bash
pip install -r requirements.txt
```

Run the program!

## Usage

This program has a command-line interface built up using [click](https://github.com/pallets/click).
An auto-generated help message is printed with the `-h/--help` flag.

```
> python hybrid.py -h
Usage: hybrid.py [OPTIONS] COMMAND [ARGS]...

  Hybrid image demonstration program.

Options:
  -h, --help  Show this message and exit.

Commands:
  hybrid  Create hybrid image from two source images.
  kernel  Demonstrate the effect of kernel size.
  sobel   Perform sobel edge detection.
```

```
> python hybrid.py hybrid -h
Usage: hybrid.py hybrid [OPTIONS] IMAGES...

  Create hybrid image from two source images.

Options:
  -o, --output TEXT        Output file.
  -c, --cutoff INTEGER...  High/low cutoff frequencies.
  -v, --visual             Generate visualisation.
  -f, --fourier            Use fourier convolution.
  -h, --help               Show this message and exit.
```

```
> python hybrid.py kernel -h
Usage: hybrid.py kernel [OPTIONS] IMAGE

  Demonstrate the effect of kernel size.

Options:
  -o, --output TEXT      Output file.
  -s, --size INTEGER...  Kernel dimensions.
  -h, --help             Show this message and exit.
```

```
> python hybrid.py sobel -h
Usage: hybrid.py sobel [OPTIONS] IMAGE

  Perform sobel edge detection.

Options:
  -o, --output TEXT  Output file.
  -h, --help         Show this message and exit.
```
