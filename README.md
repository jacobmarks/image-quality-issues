## Image Quality Issues Plugin

This plugin is a Python plugin that allows you to find common issues in your
image datasets.

With this plugin, you can:

-   Find bright images
-   Find dark images
-   Find weird aspect ratios
-   Find blurry images
-   Find low entropy images
-   Find underexposed images
-   Find overexposed images

It is straightforward to add support for other types of issues!

## Installation

```shell
fiftyone plugins download \
    https://github.com/jacobmarks/image-quality-issues/ \
    --plugin-names image_issues
```

## Operators

### `compute_brightness`

Computes the brightness of all images in the dataset.

### `compute_aspect_ratio`

Computes the aspect ratio of all images in the dataset.

### `compute_blurriness`

Computes the blurriness of all images in the dataset.

### `compute_entropy`

Computes the entropy of all images in the dataset.

### `compute_exposure`

Computes the exposure of all images in the dataset.

### `find_issues`

Finds images with brightness, aspect ratio, or entropy issues. You can specify
the threshold for each issue type, and which issue types to check for.

## See Also
Here are some related projects that this plugin is inspired by and adapted from:

- CleanLab's [CleanVision](https://github.com/cleanlab/cleanvision/tree/main)
- Lakera's [MLTest](https://docs.lakera.ai/)
- [BlurDetection2](https://github.com/WillBrennan/BlurDetection2) with OpenCV
- [Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction)
