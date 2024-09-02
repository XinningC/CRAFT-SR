from basicsr.utils import get_root_logger, scandir
from copy import deepcopy
from basicsr.utils.registry import LOSS_REGISTRY

def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    logger = get_root_logger()
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss

opt =  {"type": "L1Loss","loss_weight": 1.0,"reduction":"mean"}
cri = build_loss(opt)