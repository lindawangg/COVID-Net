import tensorflow as tf
from matplotlib import cm


def heatmap_overlay_summary_op(name, image_tsr, mask_tsr, alpha=0.2, cmap='jet', max_outputs=5, masked=False):
    """Makes an image summary op which overlays a mask on an image using a heatmap overlay"""
    image_tsr = normalize_minmax(image_tsr)
    heatmap = make_heatmap(mask_tsr, cmap=cmap)
    overlay = alpha_blend(heatmap, image_tsr, alpha, mask_tsr=(mask_tsr if masked else None))
    summary_op = tf.summary.image(name, overlay, max_outputs=max_outputs)
    return summary_op


def colour_overlay_summary_op(name, image_tsr, mask_tsr, alpha=0.2, colour=(1, 0, 0), max_outputs=5):
    """Makes an image summary op which overlays a mask on an image using a solid colour overlay"""
    image_tsr = normalize_minmax(image_tsr)
    colour_tsr = make_rgb_tsr(mask_tsr, colour)
    overlay = alpha_blend(colour_tsr, image_tsr, alpha, mask_tsr)
    summary_op = tf.summary.image(name, overlay, max_outputs=max_outputs)
    return summary_op


def normalize_minmax(tsr):
    """Normalizes a tensor to the range [0, 1]"""
    tsr_min = tf.reduce_min(tsr)
    tsr_max = tf.reduce_max(tsr)
    tsr_normed = (tsr - tsr_min)/(tsr_max - tsr_min)
    return tsr_normed


def alpha_blend(image1_tsr, image2_tsr, alpha, mask_tsr=None):
    """Performs alpha blending of two image tensors"""
    if mask_tsr is None:
        alpha_mask = alpha
    else:
        alpha_mask = alpha*mask_tsr
    blend_tsr = alpha_mask*image1_tsr + (1. - alpha_mask)*image2_tsr
    blend_tsr = tf.cast(255.*blend_tsr, tf.uint8)
    return blend_tsr


def make_heatmap(mask_tsr, cmap='jet'):
    """Converts an image with values in [0, 1] to a heatmap"""
    # Get colormap indices
    indices = tf.to_int32(tf.round(255.*tf.squeeze(mask_tsr)))

    # Get colourmap values
    levels = list(range(256))
    cm_func = cm.get_cmap(cmap)
    cmap_vals = tf.constant(cm_func(levels)[:, :3], dtype=tf.float32)

    # Gather colourmap values at indices
    heatmap = tf.gather(cmap_vals, indices)

    return heatmap


def make_rgb_tsr(mask_tsr, colour):
    """Makes a solid colour rgb tensor in the given shape"""
    ones = tf.ones_like(mask_tsr)
    channels = [ones*c for c in colour]
    rgb_tsr = tf.concat(channels, axis=-1)
    return rgb_tsr


def scalar_summary(tag_to_value, tag_prefix=''):
    """Summary object for a dict of scalars"""
    return tf.Summary(value=[tf.Summary.Value(tag=tag_prefix + tag, simple_value=value)
                             for tag, value in tag_to_value.items() if isinstance(value, (int, float))])
