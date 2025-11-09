from .clip_mora import ClipMoRA


def build_model(cfg):
    if cfg.NAME == "ClipMoRA":
        model = ClipMoRA(cfg)
    else:
        raise ValueError(f"Cannot find the model named {cfg.NAME}.")
    return model