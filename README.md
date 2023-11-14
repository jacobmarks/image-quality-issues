## Image Quality Issues Plugin

![common_image_issues_updated](https://github.com/jacobmarks/image-quality-issues/assets/12500356/4f3b89c9-58b6-4404-a9da-8cd6570a1793)

This plugin is a Python plugin that allows you to find common issues in your
image datasets.

With this plugin, you can find the following issues:

- 📏 Aspect ratio: find images with weird aspect ratios
- 🌫️ Blurriness: find blurry images
- ☀️ Brightness: find bright and dark images
- 🌓 Contrast: find images with high or low contrast
- 🔀 Entropy: find images with low entropy
- 📸 Exposure: find overexposed and underexposed images
- 🕯️ Illumination: find images with uneven illumination
- 🧂 Noise: find images with high salt and pepper noise
- 🌈 Saturation: find images with low and high saturation

For any of these properties, the plugin allows you to compute with the following options:
- In real-time, or in delegated execution mode
- For full images, or for any object patches, specified by a `Detections` field
- for the entire dataset, the current view, or for currently selected samples

These computations are all united via a `find_issues` operator, which allows you to designate images (or detections) as plagued by specific issues. You can run the issue-finding operator in single-issue or multi-issue mode, and can specify the threshold for each issue at the time of execution. All necessary computations which have not yet been run will be run.

### Updates

- **2021-11-13**: Version 2.0.1 supports [calling the compute methods from the Python SDK](#python-sdk)!
- **2021-11-10**: Version 2.0.0 adds support for object patches, lots of refactoring, and more robustness!

## Watch On Youtube

[![Video Thumbnail](https://img.youtube.com/vi/0Kkzx0nEXEo/0.jpg)](https://www.youtube.com/watch?v=0Kkzx0nEXEo&list=PLuREAXoPgT0RZrUaT0UpX_HzwKkoB-S9j&index=14)

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/image-quality-issues/
```


## Properties

### Aspect Ratio
Relatively self-explanatory, this is the raio of image (or patch) width to height. The minimum of `width/height` and `height/width` is computed.

Operator: `compute_aspect_ratio`

Compute aspect ratio of images:
![image_aspect_ratio_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/fe052278-7a64-4b39-b22f-240f0f12ed2e)

Compute aspect ratio of detection patches:
![patch_aspect_ratio_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/549fd7d8-b338-44d8-a401-6a03fe1693db)

### Blurriness

Blurriness measures the lack of sharpness, or clarity in the image. It encapsulates multiple sources, including motion blur, defocus blur, and low-quality imaging. You can compute blurriness with the `compute_blurriness` operator.

![blurriness_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/c6cc790c-ddcc-43d8-9a42-8b118f470b14)


### Brightness

Brightness attempts to quantify the amount of light in an image, as it is perceived by the viewer. For RGB images, it is computed by taking a weighted average of the RGB channels for each pixel and computing a luminance score. You can run the computation with `compute_brightness`.

![brightness_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/824e1972-9878-4c0c-8ccc-5de03c0275fa)

This brightness score allows you to identify dark images and bright images.

### Contrast

Contrast measures the difference in brightness between the darkest and brightest parts of an image. Low contrast means high uniformity across the image. You can compute the contrast for your images or patches with `compute_contrast`.

![contrast_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/b2767143-e436-4dc2-8b30-665820a59fb3)

### Entropy

Entropy quantifies the information content of the pixels in an image. The more complex the visual structure, the higher the entropy. Low entropy images typically provide little information to a model, so it may be desired to remove them from the dataset before training. Compute entropy with `compute_entropy`.

![entropy_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/4a39fda5-f233-4b16-909a-cfece9edbbf6)

### Exposure

Exposure measures the amount of light per unit area reaching the image sensor or film. It is related to but distinct from brightness. Low exposure and high exposure can both be issues that plague images captured by cameras. Compute exposure with `compute_exposure`.

![exposure_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/df42beeb-e086-4ed3-9ab6-b5a2cbbcd092)

### Salt 'n Pepper Noise

Salt and pepper noise is a variety of noise in images that comes in the form of black and white pixels. It can result in grainy images, or images which are hard to make accurate predictions on. Compute the salt and pepper noise with `compute_salt_and_pepper`.

![salt_and_pepper_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/2a2926b3-d784-4ec3-a961-9ef9bb624379)

### Saturation

Saturation measures the intensity or purity of color in an image. Grayscale images have low saturation, whereas vivid, rich images have high saturation. Compute saturation with `compute_saturation`.

![saturation_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/90b42694-cdea-42f7-b7a0-9e6c464370ee)


### Vignetting

Vignetting measures the dropoff in brightness and saturation near the periphery of an image compared to at its center. This can be an undesired consequence of camera settings. Compute vignetting with `compute_vignetting`.
![vignetting_compressed](https://github.com/jacobmarks/image-quality-issues/assets/12500356/0a35c03a-db08-44c8-a51b-48a868c19d88)


### `find_issues`

Finds images with brightness, aspect ratio, or entropy issues. You can specify
the threshold for each issue type, and which issue types to check for.

## Python SDK

You can also use the compute operators from the Python SDK!

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

dataset = fo.load_dataset("quickstart")

## Access the operator via its URI (plugin name + operator name)
compute_brightness = foo.get_operator(
    "@jacobmarks/image_issues/compute_brightness"
)

## Compute the brightness of all images in the dataset
compute_brightness(dataset)

## Compute brightness of a view
view = dataset.take(10)
compute_brightness(view)

## Compute brightness of detection patches
compute_brightness(dataset, patches_field="ground_truth")
```

**Note**: The `find_issues` operator is not yet supported in the Python SDK.

**Note**: You may have trouble running these within a Jupyter notebook. If so, try running them in a Python script.

## See Also

Here are some related projects that this plugin is inspired by and adapted from:

- [BlurDetection2](https://github.com/WillBrennan/BlurDetection2) with OpenCV
- CleanLab's [CleanVision](https://github.com/cleanlab/cleanvision/tree/main)
- [Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction)
- Lakera's [MLTest](https://docs.lakera.ai/)
