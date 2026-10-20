"""FOCUS-3D upstream PyTorch inference adapter for native-grid comparisons."""
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOCUS_ROOT = ROOT / 'vendor/focus3d'
FOCUS_COMMIT = '033017f20234f33722baca7b94679283e8fa6d47'
DEFAULT_CONFIG = FOCUS_ROOT / 'src/focus3d/segmentation/FOCUS3D/configs/3d_test.yaml'


def inference_options(settings, spacing):
    """Validate parameters before loading the multi-GB model."""
    if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
        raise ValueError('FOCUS-3D requires positive ZYX voxel spacing.')
    if not math.isclose(spacing[1], spacing[2], rel_tol=1e-4):
        raise ValueError('FOCUS-3D assumes equal X/Y voxel spacing. Resample to square XY pixels before inference.')
    def number(key, default, low, high, integer=False):
        value = float(settings.get(key, default))
        if not math.isfinite(value) or not low <= value <= high or (integer and value != int(value)):
            raise ValueError(f'Invalid FOCUS-3D {key}: expected {low}–{high}.')
        return int(value) if integer else value
    low = number('lower_percentile', 1, 0, 100)
    high = number('upper_percentile', 99, 0, 100)
    if low >= high:
        raise ValueError('Normalization lower percentile must be below upper percentile.')
    stride = tuple(number(k, d, 1, p-1, True) for k, d, p in zip(
        ('stride_z', 'stride_y', 'stride_x'), (24, 64, 64), (32, 96, 96)))
    minimum = number('min_size', 0, 0, 2**31-1, True)
    maximum = number('max_size', 0, 0, 2**31-1, True)
    if maximum and maximum < minimum:
        raise ValueError('Maximum size must be zero (unlimited) or at least minimum size.')
    background = None if settings.get('automatic_background', True) else number('background_threshold', 0, -1e30, 1e30)
    device = settings.get('device', 'auto')
    if device not in ('cpu', 'auto', 'mps', 'cuda:0'):
        raise ValueError('FOCUS-3D device must be Auto, CPU, Apple MPS or CUDA 0.')
    return dict(device=device, z_ratio=spacing[0]/spacing[1],
                cell_radius=number('cell_radius', 15, 1, 1000),
                lower_percentile=low, upper_percentile=high, background_threshold=background,
                batch_size=number('batch_size', 1, 1, 16, True), data_loader_num_workers=0,
                patch_size=(32, 96, 96), stride=stride, restore_resampled_mask=True,
                score_thresh=number('score_threshold', .5, 0, 1), mask_thresh=number('mask_threshold', .5, 0, 1),
                min_edge_area=number('min_edge_area', 20, 0, 2**31-1, True),
                size_filter_min_size=minimum, size_filter_max_size=maximum or None,
                stitch_face_iom_thresh=number('stitch_iom', .2, 0, 1),
                stitch_min_core_voxels=number('stitch_core', 64, 1, 2**31-1, True),
                stitch_min_contact_cc_pixels=number('stitch_contact', 10, 1, 2**31-1, True),
                stitch_min_contact_cc_ratio=number('stitch_ratio', .5, 0, 1),
                stitch_contact_connectivity=2, save_intermediate=True)


def load_backend():
    sys.path.insert(0, str(FOCUS_ROOT / 'src'))
    backend = importlib.import_module('focus3d.segmentation.FOCUS3D.inference_win')
    # Upstream's strict=False loader suppresses missing-key diagnostics. Reject
    # missing learned parameters rather than producing partial/random inference.
    backend.build_predictor = lambda cfg: checked_predictor(backend, cfg)
    # The upstream selector returns CPU before considering an explicit MPS
    # request when CUDA is absent. Adapt selection without changing model math.
    backend.resolve_inference_device = resolve_device
    backend.activate_inference_device = activate_device
    return backend


def resolve_device(device=None):
    import torch
    device = device or 'auto'
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Apple MPS GPU is unavailable. Select CPU, or run outside a sandbox that hides Metal.')
    if device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable. Select Apple MPS or CPU on this Mac.')
    if device not in ('cpu', 'mps', 'cuda:0'):
        raise ValueError(f'Unsupported FOCUS-3D device: {device}')
    return device


def activate_device(device=None):
    import torch
    device = resolve_device(device)
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))
    print(f'[Atlas FOCUS-3D] inference device: {device}',flush=True)
    return device


def checked_predictor(backend, cfg):
    import torch
    model = backend.build_maskformer_model_from_cfg(cfg)
    checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu', weights_only=False)
    state = backend._clean_state_dict_keys(backend._extract_model_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_parameters = set(missing) & dict(model.named_parameters()).keys()
    if missing_parameters:
        raise ValueError('FOCUS-3D checkpoint/config mismatch: missing learned parameters: ' + ', '.join(sorted(missing_parameters)[:8]))
    backend.atlas_model_load = dict(missingBuffers=list(missing), unexpectedKeys=list(unexpected))
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    # Release the separate CPU checkpoint tensors before inference, especially
    # valuable with this multi-GB model on unified-memory Macs.
    del checkpoint, state
    return model


def run_focus3d(raw, source, settings, folder, progress, fingerprint):
    options = inference_options(settings, source['spacing'])
    checkpoint = Path(settings.get('checkpoint', '')).expanduser()
    config = Path(settings.get('config', '') or DEFAULT_CONFIG).expanduser()
    if not checkpoint.is_file():
        raise ValueError('Select a downloaded FOCUS-3D .pth checkpoint. Official weights require Hugging Face access approval.')
    if not config.is_file():
        raise ValueError(f'FOCUS-3D configuration not found: {config}. Run zsh scripts/setup_focus3d.sh.')
    options['device'] = resolve_device(options['device'])
    backend = load_backend()
    checkpoint_hash = fingerprint(checkpoint)
    config_hash = fingerprint(config)
    progress('FOCUS-3D: loading model', .05)
    fraction = .05
    def report(value, message):
        nonlocal fraction
        if value >= 0:
            fraction = max(fraction, .05 + .85 * min(100, value) / 100)
        progress('FOCUS-3D: ' + message, fraction)
    result = backend.infer_volume(image_path=source['path'], config_file=str(config.resolve()),
                                  weights_path=str(checkpoint.resolve()), output_dir=str(folder / 'focus3d'),
                                  progress_callback=report, **options)
    labels = np.asarray(result['instance_map'])
    if labels.shape != raw.shape or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError('FOCUS-3D did not restore integer labels to the original raw grid.')
    if fingerprint(config) != config_hash or fingerprint(checkpoint) != checkpoint_hash:
        raise ValueError('FOCUS-3D configuration or checkpoint changed while inference was running.')
    commit = subprocess.check_output(['git', '-C', str(FOCUS_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    saved = dict(settings, resolvedOptions=options, checkpoint=str(checkpoint.resolve()), config=str(config.resolve()),
                 checkpointSHA256=checkpoint_hash, configSHA256=config_hash, upstreamCommit=commit,
                 adapterSHA256=fingerprint(__file__), upstreamLog=result['log_info'],
                 modelLoad=getattr(backend, 'atlas_model_load', {}),
                 mpsCPUFallbackEnabled=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1',
                 patches=result['num_infer_patches'], skippedPatches=result['num_skipped_background_patches'])
    return labels, saved, 'FOCUS-3D upstream no-Detectron2 PyTorch backend; raw intensities only. Cell-radius/ZYX resampling restored to the original grid; no reference masks.'
