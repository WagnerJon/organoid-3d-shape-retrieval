"""Cellpose's native orthoview prediction and 3D dynamics on an isotropic grid."""
import math
import logging


def settings(values):
    mode = values.get('segmentationMode', 'usegment3d')
    if mode not in ('usegment3d', 'cellpose3d'):
        raise ValueError('Choose u-Segment3D or native Cellpose 3D.')
    result = dict(segmentationMode=mode, flow3DSmooth=float(values.get('flow3DSmooth', 0.)),
                  cellprobThreshold=float(values.get('cellprobThreshold', 0.)), minSize3D=values.get('minSize3D', 15))
    if not math.isfinite(result['flow3DSmooth']) or not 0 <= result['flow3DSmooth'] <= 20:
        raise ValueError('3D flow smoothing must be between 0 and 20 processed pixels.')
    if not math.isfinite(result['cellprobThreshold']):
        raise ValueError('Cell probability threshold must be finite.')
    if type(result['minSize3D']) is not int or not 0 <= result['minSize3D'] <= 10_000_000:
        raise ValueError('Minimum 3D size must be an integer between 0 and 10000000 processed voxels.')
    return result


def stage_sequence(config):
    return ['preprocess', 'aggregate', 'export'] if config.get('segmentationMode') == 'cellpose3d' else ['preprocess','xy','xz','yz','aggregate','export']


def predict(processed, values, progress=None, model_factory=None):
    import numpy as np
    options = settings(values)
    report = progress or (lambda *args: None)
    if processed.ndim != 3 or min(processed.shape) < 2:
        raise ValueError('Native Cellpose 3D requires a 3D Z/Y/X processing grid.')
    if any(not np.isfinite(processed[z]).all() for z in range(len(processed))):
        raise ValueError('Nonfinite intensities on the processing grid.')
    if model_factory is None:
        from atlas_cellpose_models import make_model
        model_factory = make_model
    report('Cellpose 3D · loading model', .05)
    from atlas_cellpose_models import factory_arguments
    model = model_factory(**factory_arguments(values))
    network_device = str(model.device)
    dynamics_device = 'cpu' if network_device.startswith('mps') else network_device
    report(f'Cellpose 3D · orthoview prediction ({network_device}) and 3D masks ({dynamics_device})', .15)
    # Atlas preprocessing already corrects physical anisotropy. Applying the raw
    # Z/XY ratio again here would stretch the volume twice.
    class PhaseProgress(logging.Handler):
        def emit(self, record):
            message=record.getMessage()
            for view, fraction in [('YX',.2),('ZY',.4),('ZX',.6)]:
                if message.startswith('running '+view+':'):
                    report('Cellpose 3D · '+message,fraction)
            if 'MPS does not support 3D post-processing' in message:
                report('Cellpose 3D · forming masks on CPU',.8)
    loggers=[logging.getLogger(name) for name in ('cellpose','_atlas_cellpose3')]
    previous_levels=[logger.level for logger in loggers];listener=PhaseProgress()
    for logger in loggers:logger.addHandler(listener);logger.setLevel(logging.INFO)
    try:
        masks, fields, _ = model.eval(np.array(processed, dtype=np.float32, copy=True)[..., None],
                                     channel_axis=3, z_axis=0, do_3D=True, anisotropy=1., batch_size=4,
                                     diameter=float(values['diameter']), compute_masks=True,
                                     flow3D_smooth=options['flow3DSmooth'],
                                     cellprob_threshold=options['cellprobThreshold'], min_size=options['minSize3D'])
    finally:
        for logger,level in zip(loggers,previous_levels):logger.removeHandler(listener);logger.setLevel(level)
    if masks.shape != processed.shape or not np.issubdtype(masks.dtype, np.integer) or np.any(masks < 0) or masks.max() > np.iinfo(np.uint32).max:
        raise ValueError('Cellpose returned invalid native 3D masks.')
    runtime=getattr(model,'atlas_runtime',None)
    del fields, model
    report('Cellpose 3D · masks ready', 1.)
    return masks.astype(np.uint32, copy=False), dict(**options, anisotropy=1., networkDevice=network_device,
                                                   dynamicsDevice=dynamics_device, cells=int(np.count_nonzero(np.unique(masks))),
                                                   modelRuntime=runtime,
                                                   note='Native do_3D=True on Atlas isotropic preprocessing grid; 2D flow threshold is not used.')
