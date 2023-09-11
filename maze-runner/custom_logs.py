
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def append(info, key, value):
    if key not in info:
        info[key] = []
    info[key].append(value)


class CustomTensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        venv = self.training_env
        info = {}

        for i in range(len(venv.reset_infos)):
            for key, value in venv.reset_infos[i].items():
                append(info, key, value)
        for key, value in info.items():
            self.logger.record(f"info/{key}:", np.mean(value))
        return True
