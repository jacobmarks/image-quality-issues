## Image Quality Issues Plugin

![image_issues](https://github.com/jacobmarks/image-quality-issues/assets/12500356/d69a14f7-011d-471a-bc17-503eff0f3f6e)

This plugin is a Python plugin that allows you to find common issues in your
image datasets.

With this plugin, you can find the following issues:

-   📏 Aspect ratio: find images with weird aspect ratios
-   🌫️ Blurriness: find blurry images
-   ☀️ Brightness: find bright and dark images
-   🌓 Contrast: find images with high or low contrast
-   🔀 Entropy: find images with low entropy
-   📸 Exposure: find overexposed and underexposed images
-   🕯️ Illumination: find images with uneven illumination
-   🧂 Noise: find images with high salt and pepper noise
-   🌈 Saturation: find images with low and high saturation


## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/image-quality-issues/
```

## Operators

### `compute_aspect_ratio`

Computes the aspect ratio of all images in the dataset.

### `compute_blurriness`

Computes the blurriness of all images in the dataset.


### `compute_brightness`

Computes the brightness of all images in the dataset.

### `compute_contrast`

Computes the contrast of all images in the dataset.

### `compute_entropy`

Computes the entropy of all images in the dataset.

### `compute_exposure`

Computes the exposure of all images in the dataset.


### `compute_salt_and_pepper`

Computes the salt and pepper noise of all images in the dataset.

### `compute_saturation`

Computes the saturation of all images in the dataset.


### `compute_vignetting`

Computes the vignetting of all images in the dataset.

### `find_issues`

Finds images with brightness, aspect ratio, or entropy issues. You can specify
the threshold for each issue type, and which issue types to check for.

## See Also
Here are some related projects that this plugin is inspired by and adapted from:

- [BlurDetection2](https://github.com/WillBrennan/BlurDetection2) with OpenCV
- CleanLab's [CleanVision](https://github.com/cleanlab/cleanvision/tree/main)
- [Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction)
- Lakera's [MLTest](https://docs.lakera.ai/)
