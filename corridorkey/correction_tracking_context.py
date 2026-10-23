"""Explicit source interpretation bound to derived final correction masks."""
from pathlib import Path


def tracking_decode(config, source):
    return {'contract': 'corridorkey-native-rgb-v1',
            'source_kind': 'frames' if Path(source).is_dir() else 'video',
            'frame_dir_linear': config.frame_dir_linear,
            'frame_dir_alpha': config.frame_dir_alpha,
            'source_colour_profile': config.source_colour_profile,
            'linear_brightness': config.linear_brightness,
            'linear_contrast': config.linear_contrast,
            'linear_contrast_pivot': config.linear_contrast_pivot}


def tracking_cuts(config):
    # Cuts and explicit temporal resets both bound source motion propagation.
    return sorted(set(config.quick_cut_frames) | set(config.quick_reset_frames))
