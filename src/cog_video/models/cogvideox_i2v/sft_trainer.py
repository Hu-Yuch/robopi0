from ..cogvideox_i2v.bridge_trainer import CogVideoXBridgeLoraTrainer
from ..utils import register


class CogVideoXI2VSftTrainer(CogVideoXBridgeLoraTrainer):
    pass


register("cogvideoxbridge", "sft", CogVideoXI2VSftTrainer)
