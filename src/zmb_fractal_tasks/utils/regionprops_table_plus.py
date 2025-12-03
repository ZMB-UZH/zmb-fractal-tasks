import numpy as np
from skimage.measure import regionprops, regionprops_table
from skimage.measure._regionprops import COL_DTYPES


def most_frequent_value(mask, img):  # noqa: D103
    masked_img = img[mask].astype(int)
    return np.bincount(masked_img).argmax()


def intensity_std(mask, img):  # noqa: D103
    masked_img = img[mask]
    return np.std(masked_img)


def intensity_total(mask, img):  # noqa: D103
    masked_img = img[mask]
    return np.sum(masked_img)


FUNS_PLUS = {
    "most_frequent_value": most_frequent_value,
    "intensity_std": intensity_std,
    "intensity_total": intensity_total,
}


def regionprops_plus(
    label_image,
    intensity_image=None,
    cache=True,
    *,
    extra_properties=None,
    spacing=None,
    offset=None,
):
    """Wrapper around regionprops, to integrate extra properties.

    The additional properties are:
    - most_frequent_value
        returns most frequent value of img inside mask
        (mainly used for annotating labels, where img are annotation labels)
    - intensity_std: float
        standard deviation of pixel values
    - intensity_total:
        sum of pixel values
    """
    properties_plus = [most_frequent_value, intensity_std, intensity_total]
    if extra_properties is not None:
        extra_properties = extra_properties + properties_plus
    else:
        extra_properties = properties_plus
    return regionprops(
        label_image=label_image,
        intensity_image=intensity_image,
        cache=cache,
        extra_properties=extra_properties,
        spacing=spacing,
        offset=offset,
    )


def regionprops_table_plus(
    label_image,
    intensity_image=None,
    properties=("label", "bbox"),
    *,
    cache=True,
    separator="-",
    extra_properties=None,
    spacing=None,
):
    """Like skimage.measure.regionprops_table(), but with some additional properties:

    - most_frequent_value
        returns most frequent value of img inside mask
        (mainly used for annotating labels, where img are annotation labels)
    - intensity_std: float
        standard deviation of pixel values
    - intensity_total:
        sum of pixel values
    """
    properties_org = []
    properties_plus = []
    for prop in properties:
        if prop in COL_DTYPES.keys():
            properties_org.append(prop)
        elif prop in FUNS_PLUS.keys():
            properties_plus.append(FUNS_PLUS[prop])
    rpt = regionprops_table(
        label_image,
        intensity_image,
        properties=properties_org,
        cache=cache,
        separator=separator,
        extra_properties=properties_plus,
        spacing=spacing,
    )
    return {
        prop: rpt[prop] for prop in properties
    }  # sort table according to input properties
