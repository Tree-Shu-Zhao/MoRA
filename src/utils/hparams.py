from loguru import logger


def reset_cfg(cfg, params):
    # Optimizer
    if "LR" in params.keys():
        cfg.train.LR = params["LR"]
        logger.info(f"Reset LR to {params['LR']}")
    if "WEIGHT_DECAY" in params.keys():
        cfg.train.WEIGHT_DECAY = params["WEIGHT_DECAY"]
        logger.info(f"Reset WEIGHT_DECAY to {params['WEIGHT_DECAY']}")
    if "BATCH_SIZE" in params.keys():
        cfg.train.BATCH_SIZE = params["BATCH_SIZE"]
        logger.info(f"Reset BATCH_SIZE to {params['BATCH_SIZE']}")
    if "EPOCHS" in params.keys():
        cfg.train.EPOCHS = params["EPOCHS"]
        logger.info(f"Reset EPOCHS to {params['EPOCHS']}")
    if "WARMUP_RATIO" in params.keys():
        cfg.train.WARMUP_RATIO = params["WARMUP_RATIO"]
        logger.info(f"Reset WARMUP_RATIO to {params['WARMUP_RATIO']}")
    if "REC_LOSS_WEIGHT" in params.keys():
        cfg.train.REC_LOSS_WEIGHT = params["REC_LOSS_WEIGHT"]
        logger.info(f"Reset REC_LOSS_WEIGHT to {params['REC_LOSS_WEIGHT']}")
    
    # Model
    if "START_LAYER_INDEX" in params.keys():
        cfg.model.lora.START_LAYER_INDEX = params["START_LAYER_INDEX"]
        logger.info(f"Reset START_LAYER_INDEX to {params['START_LAYER_INDEX']}")
    if "RANK" in params.keys():
        cfg.model.lora.RANK = params["RANK"]
        logger.info(f"Reset RANK to {params['RANK']}")
    if "SHARED_RANK" in params.keys():
        cfg.model.lora.SHARED_RANK = params["SHARED_RANK"]
        logger.info(f"Reset SHARED_RANK to {params['SHARED_RANK']}")
    if "VISION_RANK" in params.keys():
        cfg.model.lora.VISION_RANK = params["VISION_RANK"]
        logger.info(f"Reset VISION_RANK to {params['VISION_RANK']}")
    if "TEXT_RANK" in params.keys():
        cfg.model.lora.TEXT_RANK = params["TEXT_RANK"]
        logger.info(f"Reset TEXT_RANK to {params['TEXT_RANK']}")
    if "ALPHA" in params.keys():
        cfg.model.lora.ALPHA = params["ALPHA"]
        logger.info(f"Reset ALPHA to {params['ALPHA']}")
    if "DROPOUT" in params.keys():
        cfg.model.lora.DROPOUT = params["DROPOUT"]
        logger.info(f"Reset DROPOUT to {params['DROPOUT']}")
    if "TARGET_MODULES" in params.keys():
        tm = params["TARGET_MODULES"]
        if tm == "q":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
            ]
        elif tm == "k":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
            ]
        elif tm == "v":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
            ]
        elif tm == "o":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        elif tm == "qk":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
            ]
        elif tm == "qv":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
            ]
        elif tm == "qo":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        elif tm == "kv":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
            ]
        elif tm == "ko":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        elif tm == "vo":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        elif tm == "qkv":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
            ]
        elif tm == "qko":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        elif tm == "qvo":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        elif tm == "kvo":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        elif tm == "qkvo":
            tms = [
                "clip.vision_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.vision_model.encoder.layers.{i}.self_attn.out_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.q_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.k_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.v_proj",
                "clip.text_model.encoder.layers.{i}.self_attn.out_proj",
            ]
        cfg.model.lora.TARGET_MODULES = tms
        logger.info(f"Reset TARGET_MODULES to {params['TARGET_MODULES']}")

    return cfg