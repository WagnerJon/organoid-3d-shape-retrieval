"""Explicit geometry estimate for imports without usable physical calibration."""

ESTIMATED_SOURCE = ('Missing calibration; assume cubic stack (estimated): X = Y = 1 nominal µm, '
                    'Z = longest XY dimension / slice count. Absolute physical scale is unknown (editable).')


def assumed_cubic_spacing(shape):
    """Return Z/Y/X spacing, retaining square XY pixels and native XY aspect ratio.

    Depth matches the longer lateral extent. Non-square images therefore keep
    their rectangular field of view; they are not stretched to force a cube.
    """
    if len(shape) != 3 or any(int(n) != n or n < 1 for n in shape):
        raise ValueError('Cubic-stack estimation requires positive Z/Y/X dimensions.')
    z, y, x = shape
    return [float(max(y, x)) / z, 1., 1.]
