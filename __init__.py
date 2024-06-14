"""Common Issues plugin.

This plugin provides operators to compute common issues in image datasets.
It is open source, and is adapted heavily from leading open source projects
listed in the README.
|
"""

import numpy as np


import cv2
from PIL import Image


import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F


######## HELPER FUNCTIONS ########


def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )


def _crop_pillow_image(pillow_img, detection):
    img_w, img_h = pillow_img.width, pillow_img.height

    bounding_box = detection.bounding_box
    left, top, width, height = bounding_box
    left *= img_w
    top *= img_h
    right = left + width * img_w
    bottom = top + height * img_h

    return pillow_img.crop((left, top, right, bottom))


def _get_pillow_patch(sample, detection):
    img = Image.open(get_filepath(sample))
    return _crop_pillow_image(img, detection)


def _convert_pillow_to_opencv(pillow_img):
    # pylint: disable=no-member
    return cv2.cvtColor(np.array(pillow_img), cv2.COLOR_RGB2BGR)


def _convert_opencv_to_pillow(opencv_image):
    # pylint: disable=no-member
    return Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))


def _get_opencv_grayscale_image(sample):
    # pylint: disable=no-member
    return cv2.imread(get_filepath(sample), cv2.IMREAD_GRAYSCALE)


######## CONTEXT & INPUT MANAGEMENT ########


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


def _handle_patch_inputs(ctx, inputs):
    target_view = ctx.target_view()
    patch_types = (fo.Detection, fo.Detections, fo.Polyline, fo.Polylines)
    patches_fields = list(
        target_view.get_field_schema(embedded_doc_type=patch_types).keys()
    )

    if patches_fields:
        patches_field_choices = types.DropdownView()
        for field in sorted(patches_fields):
            patches_field_choices.add_choice(field, label=field)

        inputs.str(
            "patches_field",
            default=None,
            required=False,
            label="Patches field",
            description=(
                "An optional sample field defining image patches in each "
                "sample to run the computation on. If omitted, the full images "
                "will be used."
            ),
            view=patches_field_choices,
        )


######## COMPUTATION FUNCTIONS ########

#### ASPECT RATIO ####
def _compute_aspect_ratio(width, height):
    ratio = width / height
    return min(ratio, 1 / ratio)


def compute_sample_aspect_ratio(sample):
    width, height = sample.metadata.width, sample.metadata.height
    return _compute_aspect_ratio(width, height)


def compute_patch_aspect_ratio(sample, detection):
    img_width, img_height = sample.metadata.width, sample.metadata.height
    bbox_width, bbox_height = detection.bounding_box[2:]
    width, height = bbox_width * img_width, bbox_height * img_height
    return _compute_aspect_ratio(width, height)


#### BLURRINESS ####
def _compute_blurriness(cv2_img):
    # pylint: disable=no-member
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    # pylint: disable=no-member
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def compute_sample_blurriness(sample):
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample))
    return _compute_blurriness(image)


def compute_patch_blurriness(sample, detection):
    patch = _get_pillow_patch(sample, detection)
    patch = _convert_pillow_to_opencv(patch)
    return _compute_blurriness(patch)


#### BRIGHTNESS ####
def _compute_brightness(pillow_img):
    pixels = np.array(pillow_img)
    if pixels.ndim == 3 and pixels.shape[-1] == 3:
        r, g, b = pixels.mean(axis=(0, 1))
    else:
        mean = pixels.mean()
        r, g, b = (
            mean,
            mean,
            mean,
        )

    ## equation from here:
    ## https://www.nbdtech.com/Blog/archive/2008/04/27/calculating-the-perceived-brightness-of-a-color.aspx
    ## and here:
    ## https://github.com/cleanlab/cleanvision/blob/72a1535019fe7b4636d43a9ef4e8e0060b8d66ec/src/cleanvision/issue_managers/image_property.py#L95
    brightness = (
        np.sqrt(0.241 * r**2 + 0.691 * g**2 + 0.068 * b**2) / 255
    )
    return brightness


def compute_sample_brightness(sample):
    filepath = get_filepath(sample)
    with Image.open(filepath) as image:
        return _compute_brightness(image)


def compute_patch_brightness(sample, detection):
    patch = _get_pillow_patch(sample, detection)
    return _compute_brightness(patch)


#### CONTRAST ####
def _compute_contrast(cv2_image):
    # Calculate the histogram
    histogram, _ = np.histogram(cv2_image, bins=256, range=(0, 256))
    min_intensity = np.min(np.where(histogram > 0))
    max_intensity = np.max(np.where(histogram > 0))
    contrast_range = max_intensity - min_intensity
    return contrast_range


def compute_sample_contrast(sample):
    image = _get_opencv_grayscale_image(sample)
    return _compute_contrast(image)


def compute_patch_contrast(sample, detection):
    cv2_image = _get_opencv_grayscale_image(sample)
    pillow_image = _convert_opencv_to_pillow(cv2_image)
    patch = _crop_pillow_image(pillow_image, detection)
    patch = _convert_pillow_to_opencv(patch)
    return _compute_contrast(patch)


#### ENTROPY ####
def _compute_entropy(pillow_img):
    return pillow_img.entropy()


def compute_sample_entropy(sample):
    filepath = get_filepath(sample)
    with Image.open(filepath) as image:
        return _compute_entropy(image)


def compute_patch_entropy(sample, detection):
    patch = _get_pillow_patch(sample, detection)
    return _compute_entropy(patch)


#### EXPOSURE ####
def _compute_exposure(opencv_gray_img):
    # pylint: disable=no-member
    histogram = cv2.calcHist([opencv_gray_img], [0], None, [256], [0, 256])
    normalized_histogram = histogram.ravel() / histogram.max()
    min_exposure = normalized_histogram[0]
    max_exposure = normalized_histogram[-1]
    return min_exposure, max_exposure


def compute_sample_exposure(sample):
    gray_img = _get_opencv_grayscale_image(sample)
    return _compute_exposure(gray_img)


def compute_patch_exposure(sample, detection):
    gray_img = _get_opencv_grayscale_image(sample)
    pillow_image = _convert_opencv_to_pillow(gray_img)
    patch = _crop_pillow_image(pillow_image, detection)
    patch = _convert_pillow_to_opencv(patch)
    return _compute_exposure(patch)


#### SALT AND PEPPER ####
def _compute_salt_and_pepper(opencv_gray_img):
    SALT_THRESHOLD = 244
    PEPPER_THRESHOLD = 10

    # Identifying salt-and-pepper pixels
    salt_pixels = opencv_gray_img >= SALT_THRESHOLD
    pepper_pixels = opencv_gray_img <= PEPPER_THRESHOLD

    # Morphological operations to exclude larger contiguous regions
    kernel = np.ones((2, 2), np.uint8)

    # Dilate and then erode (Opening operation)
    # pylint: disable=no-member
    salt_opening = cv2.morphologyEx(
        salt_pixels.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )  # pylint: disable=no-member
    pepper_opening = cv2.morphologyEx(
        pepper_pixels.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )

    # Identify isolated salt and pepper pixels
    salt_isolated = salt_pixels & ~salt_opening
    pepper_isolated = pepper_pixels & ~pepper_opening

    # Calculate the percentage of isolated salt-and-pepper pixels
    total_isolated_salt_pepper_pixels = np.sum(salt_isolated) + np.sum(
        pepper_isolated
    )
    total_pixels = opencv_gray_img.size
    noise_percentage = total_isolated_salt_pepper_pixels / total_pixels * 100

    return noise_percentage


def compute_sample_salt_and_pepper(sample):
    gray_img = _get_opencv_grayscale_image(sample)
    return _compute_salt_and_pepper(gray_img)


def compute_patch_salt_and_pepper(sample, detection):
    gray_img = _get_opencv_grayscale_image(sample)
    pillow_image = _convert_opencv_to_pillow(gray_img)
    patch = _crop_pillow_image(pillow_image, detection)
    patch = _convert_pillow_to_opencv(patch)
    return _compute_salt_and_pepper(patch)


def _compute_saturation(open_cv_image):
    # pylint: disable=no-member
    hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return np.mean(saturation)


def compute_sample_saturation(sample):
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample))
    return _compute_saturation(image)


def compute_patch_saturation(sample, detection):
    # pylint: disable=no-member
    opencv_image = cv2.imread(get_filepath(sample))
    pillow_image = _convert_opencv_to_pillow(opencv_image)
    patch = _crop_pillow_image(pillow_image, detection)
    patch = _convert_pillow_to_opencv(patch)
    return _compute_saturation(patch)


#### VIGNETTING ####
def _compute_vignetting(opencv_gray_img):
    # Get the image center
    size_y, size_x = np.array(opencv_gray_img).shape[:2]
    center_y, center_x = size_y / 2, size_x / 2

    # Calculate the maximum radius
    max_radius = np.min([center_x, center_y])

    # Create a meshgrid for calculating distances
    y, x = np.ogrid[
        -center_y : opencv_gray_img.shape[0] - center_y,
        -center_x : opencv_gray_img.shape[1] - center_x,
    ]
    distances = np.sqrt(x**2 + y**2)

    # Calculate the radial intensity profile
    radial_profile = []
    for r in range(int(max_radius)):
        mask = distances < r
        if np.any(mask):
            radial_profile.append(np.mean(opencv_gray_img[mask]))
        else:
            radial_profile.append(np.nan)  # Append NaN if the mask is empty

    radial_profile = np.array(radial_profile)

    # Filter out NaN values before calculating the drop-off
    radial_profile = radial_profile[~np.isnan(radial_profile)]

    # Analyze the profile for a drop-off, if there are any values
    if len(radial_profile) > 0:
        drop_off_percentage = (
            (radial_profile[0] - radial_profile[-1]) / radial_profile[0] * 100
        )
    else:
        drop_off_percentage = np.nan

    return drop_off_percentage


def compute_sample_vignetting(sample):
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample), cv2.IMREAD_GRAYSCALE)
    return _compute_vignetting(image)


def compute_patch_vignetting(sample, detection):
    # pylint: disable=no-member
    gray_image = cv2.imread(get_filepath(sample), cv2.IMREAD_GRAYSCALE)
    pillow_image = _convert_opencv_to_pillow(gray_image)
    patch = _crop_pillow_image(pillow_image, detection)
    patch = _convert_pillow_to_opencv(patch)
    return _compute_vignetting(patch)


################################################################
################################################################

PROP_SAMPLE_COMPUTE_FUNCTIONS = {
    "aspect_ratio": compute_sample_aspect_ratio,
    "blurriness": compute_sample_blurriness,
    "brightness": compute_sample_brightness,
    "contrast": compute_sample_contrast,
    "entropy": compute_sample_entropy,
    "exposure": compute_sample_exposure,
    "salt_and_pepper": compute_sample_salt_and_pepper,
    "saturation": compute_sample_saturation,
    "vignetting": compute_sample_vignetting,
}


PROP_PATCH_COMPUTE_FUNCTIONS = {
    "aspect_ratio": compute_patch_aspect_ratio,
    "blurriness": compute_patch_blurriness,
    "brightness": compute_patch_brightness,
    "contrast": compute_patch_contrast,
    "entropy": compute_patch_entropy,
    "exposure": compute_patch_exposure,
    "salt_and_pepper": compute_patch_salt_and_pepper,
    "saturation": compute_patch_saturation,
    "vignetting": compute_patch_vignetting,
}


def compute_dataset_property(property, dataset, view=None, patches_field=None):
    if view is None:
        view = dataset
    if patches_field is None:
        dataset.add_sample_field(property, fo.FloatField)
        for sample in view.iter_samples(autosave=True, progress=True):
            prop_value = PROP_SAMPLE_COMPUTE_FUNCTIONS[property](sample)
            if property == "exposure":
                sample["min_exposure"] = prop_value[0]
                sample["max_exposure"] = prop_value[1]
            else:
                sample[property] = prop_value
    else:
        for sample in view.iter_samples(autosave=True, progress=True):
            if sample[patches_field] is None:
                continue
            for detection in sample[patches_field].detections:
                prop_value = PROP_PATCH_COMPUTE_FUNCTIONS[property](
                    sample, detection
                )
                if property == "exposure":
                    detection["min_exposure"] = prop_value[0]
                    detection["max_exposure"] = prop_value[1]
                else:
                    detection[property] = prop_value
        dataset.add_dynamic_sample_fields()


################################################################
################################################################


##### UNIFIED INTERFACE #####
def _handle_config(property_name):
    _config = foo.OperatorConfig(
        name=f"compute_{property_name}",
        label=f"Common Issues: compute {property_name.replace('_', ' ')}",
        dynamic=True,
    )
    _config.icon = "/assets/icon.svg"
    return _config


def _handle_inputs(ctx, property_name):
    inputs = types.Object()
    label = "compute " + property_name.replace("_", " ")
    inputs.message(label, label=label)
    _execution_mode(ctx, inputs)
    inputs.view_target(ctx)
    _handle_patch_inputs(ctx, inputs)
    return types.Property(inputs)


def _handle_execution(ctx, property_name):
    view = ctx.target_view()
    patches_field = ctx.params.get("patches_field", None)
    compute_dataset_property(
        property_name, ctx.dataset, view=view, patches_field=patches_field
    )
    ctx.ops.reload_dataset()


def _handle_calling(
    uri, sample_collection, patches_field=None, delegate=False
):
    ctx = dict(view=sample_collection.view())
    params = dict(
        target="CURRENT_VIEW",
        patches_field=patches_field,
        delegate=delegate,
    )
    return foo.execute_operator(uri, ctx, params=params)


class ComputeAspectRatio(foo.Operator):
    @property
    def config(self):
        return _handle_config("aspect_ratio")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "aspect_ratio")

    def execute(self, ctx):
        _handle_execution(ctx, "aspect_ratio")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeBlurriness(foo.Operator):
    @property
    def config(self):
        return _handle_config("blurriness")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "blurriness")

    def execute(self, ctx):
        _handle_execution(ctx, "blurriness")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeBrightness(foo.Operator):
    @property
    def config(self):
        return _handle_config("brightness")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "brightness")

    def execute(self, ctx):
        _handle_execution(ctx, "brightness")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeContrast(foo.Operator):
    @property
    def config(self):
        return _handle_config("contrast")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "contrast")

    def execute(self, ctx):
        _handle_execution(ctx, "contrast")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeEntropy(foo.Operator):
    @property
    def config(self):
        return _handle_config("entropy")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "entropy")

    def execute(self, ctx):
        _handle_execution(ctx, "entropy")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeExposure(foo.Operator):
    @property
    def config(self):
        return _handle_config("exposure")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "exposure")

    def execute(self, ctx):
        _handle_execution(ctx, "exposure")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeSaltAndPepper(foo.Operator):
    @property
    def config(self):
        return _handle_config("salt_and_pepper")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "salt_and_pepper")

    def execute(self, ctx):
        _handle_execution(ctx, "salt_and_pepper")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeSaturation(foo.Operator):
    @property
    def config(self):
        return _handle_config("saturation")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "saturation")

    def execute(self, ctx):
        _handle_execution(ctx, "saturation")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


class ComputeVignetting(foo.Operator):
    @property
    def config(self):
        return _handle_config("vignetting")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        return _handle_inputs(ctx, "vignetting")

    def execute(self, ctx):
        _handle_execution(ctx, "vignetting")

    def __call__(self, sample_collection, patches_field=None, delegate=False):
        return _handle_calling(
            self.uri,
            sample_collection,
            patches_field=patches_field,
            delegate=delegate,
        )


def _need_to_compute(dataset, field_name, patches_field=None):
    if patches_field is not None:
        i = 0
        sample = dataset.skip(i).first()
        while (
            "detections" not in sample[patches_field]
            or len(sample[patches_field].detections) == 0
        ):
            i += 1
            sample = dataset.skip(i).first()
        detection = sample[patches_field].detections[0]
        if field_name not in detection:
            return True
        else:
            return False
    else:
        if field_name in list(dataset.get_field_schema().keys()):
            return False
        else:
            return field_name not in dataset.first()


def _run_computation(dataset, issue_name, patches_field=None):
    compute_dataset_property(issue_name, dataset, patches_field=patches_field)


######## ISSUE FUNCTIONS ########

ISSUE_MAPPING = {
    "bright": {
        "label": "Bright",
        "base_property": "brightness",
        "threshold": 0.55,
        "lt": False,
        "description": "Find bright images in the dataset",
    },
    "dark": {
        "label": "Dark",
        "base_property": "brightness",
        "threshold": 0.1,
        "lt": True,
        "description": "Find dark images in the dataset",
    },
    "weird_aspect_ratio": {
        "label": "Weird Aspect Ratio",
        "base_property": "aspect_ratio",
        "threshold": 0.5,
        "lt": True,
        "description": "Find weird aspect ratio images in the dataset",
    },
    "blurry": {
        "label": "Blurry",
        "base_property": "blurriness",
        "threshold": 100.0,
        "lt": True,
        "description": "Find blurry images in the dataset",
    },
    "low_entropy": {
        "label": "Low Entropy",
        "base_property": "entropy",
        "threshold": 5.0,
        "lt": True,
        "description": "Find low entropy images in the dataset",
    },
    "low_exposure": {
        "label": "Low Exposure",
        "base_property": "min_exposure",
        "threshold": 0.1,
        "lt": True,
        "description": "Find low exposure images in the dataset",
    },
    "high_exposure": {
        "label": "High Exposure",
        "base_property": "max_exposure",
        "threshold": 0.7,
        "lt": False,
        "description": "Find high exposure images in the dataset",
    },
    "low_contrast": {
        "label": "Low Contrast",
        "base_property": "contrast",
        "threshold": 50.0,
        "lt": True,
        "description": "Find low contrast images in the dataset",
    },
    "high_contrast": {
        "label": "High Contrast",
        "base_property": "contrast",
        "threshold": 200.0,
        "lt": False,
        "description": "Find high contrast images in the dataset",
    },
    "low_saturation": {
        "label": "Low Saturation",
        "base_property": "saturation",
        "threshold": 40.0,
        "lt": True,
        "description": "Find low saturation images in the dataset",
    },
    "high_saturation": {
        "label": "High Saturation",
        "base_property": "saturation",
        "threshold": 200.0,
        "lt": False,
        "description": "Find high saturation images in the dataset",
    },
}


def find_issue_images(
    dataset,
    threshold,
    field_name,
    issue_name,
    lt=True,
    patches_field=None,
    view=None,
):
    if _need_to_compute(dataset, field_name, patches_field=patches_field):
        _run_computation(dataset, field_name, patches_field=patches_field)

    if view is None:
        view = dataset

    if patches_field is None:
        dataset.add_sample_field(issue_name, fo.BooleanField)

        if lt:
            view = view.set_field(issue_name, F(field_name) < threshold)
        else:
            view = view.set_field(issue_name, F(field_name) > threshold)
        view.save()
        view = view.match(F(issue_name))
        view.tag_samples(issue_name)
        view.tag_samples("issue")
        view.save()
    else:
        embedded_field_name = f"{patches_field}.detections.{field_name}"
        embedded_issue_name = f"{patches_field}.detections.{issue_name}"
        if lt:
            values = view.values(F(embedded_field_name) < threshold)
        else:
            values = view.values(F(embedded_field_name) > threshold)
        view.set_values(embedded_issue_name, values, dynamic=True)
        view = view.filter_labels(patches_field, filter=F(issue_name) == True)
        view.tag_labels(issue_name, label_fields=patches_field)
        view.tag_labels("issue", label_fields=patches_field)
        dataset.add_dynamic_sample_fields()


def _find_issue_type_images(
    dataset, issue_type, threshold=None, patches_field=None, view=None
):
    issue = ISSUE_MAPPING[issue_type]
    if threshold is None:
        threshold = issue["threshold"]
    find_issue_images(
        dataset,
        threshold,
        issue["base_property"],
        issue_type,
        lt=issue["lt"],
        patches_field=patches_field,
        view=view,
    )


def _single_or_multi_mode(inputs):
    mode = types.RadioGroup()
    mode.add_choice(
        "SINGLE",
        label="SINGLE",
        description="Find a single type of issue",
    )
    mode.add_choice(
        "MULTI",
        label="MULTI",
        description="Find multiple types of issues",
    )
    inputs.enum(
        "issue_mode",
        mode.values(),
        default="SINGLE",
        description="Find a single type of issue or multiple types of issues",
        view=types.TabsView(),
    )


class FindIssues(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="find_issues",
            label="Common Issues: find issues",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(label="Find Common Issues")
        if ctx.dataset.media_type != "image":
            warning = types.Warning(
                label="This operator is only available for image datasets!"
            )
            inputs.view("warning", warning)
            return types.Property(inputs)

        threshold_view = types.TextFieldView(
            componentsProps={
                "textField": {
                    "step": "0.01",
                    "inputMode": "numeric",
                    "pattern": "[0-9]*",
                },
            }
        )

        _single_or_multi_mode(inputs)

        mode = ctx.params.get("issue_mode", "SINGLE")
        inputs.view_target(ctx)
        _handle_patch_inputs(ctx, inputs)

        if mode == "SINGLE":
            issue_choices = types.Dropdown(multiple=False)
            for issue in ISSUE_MAPPING:
                issue_choices.add_choice(
                    issue,
                    label=ISSUE_MAPPING[issue]["label"],
                    description=ISSUE_MAPPING[issue]["description"],
                )
            inputs.enum(
                "issue",
                issue_choices.values(),
                required=True,
                label="Issue Type",
                view=issue_choices,
            )

            for issue in ISSUE_MAPPING:
                if ctx.params.get("issue", False) == issue:
                    inputs.float(
                        issue + "_threshold",
                        default=ISSUE_MAPPING[issue]["threshold"],
                        label=ISSUE_MAPPING[issue]["description"],
                        view=threshold_view,
                    )
        else:
            for issue in ISSUE_MAPPING:
                inputs.bool(
                    issue,
                    default=True,
                    label=ISSUE_MAPPING[issue]["label"],
                    view=types.CheckboxView(),
                )

                if ctx.params.get(issue, False) == True:
                    inputs.float(
                        issue + "_threshold",
                        default=ISSUE_MAPPING[issue]["threshold"],
                        label=ISSUE_MAPPING[issue]["description"],
                        view=threshold_view,
                    )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        single_mode = ctx.params.get("issue_mode", "SINGLE")
        view = ctx.target_view()
        patches_field = ctx.params.get("patches_field", None)

        for issue in ISSUE_MAPPING.keys():
            if (
                ctx.params.get(issue, False) == True
                and single_mode == "MULTI"
                or ctx.params.get("issue", False) == issue
                and single_mode == "SINGLE"
            ):
                threshold_key = ISSUE_MAPPING[issue]["threshold"]
                threshold = ctx.params.get(threshold_key, None)
                _find_issue_type_images(
                    ctx.dataset,
                    issue,
                    threshold=threshold,
                    patches_field=patches_field,
                    view=view,
                )

        ctx.ops.reload_dataset()


def register(plugin):
    plugin.register(ComputeAspectRatio)
    plugin.register(ComputeBlurriness)
    plugin.register(ComputeBrightness)
    plugin.register(ComputeContrast)
    plugin.register(ComputeEntropy)
    plugin.register(ComputeExposure)
    plugin.register(ComputeSaltAndPepper)
    plugin.register(ComputeSaturation)
    plugin.register(ComputeVignetting)
    plugin.register(FindIssues)
