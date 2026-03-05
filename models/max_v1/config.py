from transformers import PretrainedConfig


class RegHeadConfig(PretrainedConfig):
    model_type = "reg_head"

    def __init__(self, hidden_size, pred_len=8, reg_head_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.reg_head_bias = reg_head_bias


class MaxConfig(PretrainedConfig):
    model_type = "max"

    def __init__(self, qwen_model_dir=None, pred_len=8, use_cache=False, **kwargs):
        super().__init__(**kwargs)
        self.qwen_model_dir = qwen_model_dir
        self.pred_len = pred_len
        self.use_cache = use_cache
        # -----------------------------------------------------------------------------
        # PID controller
        # -----------------------------------------------------------------------------
        # Controller
        self.turn_kp = 1.25
        self.turn_ki = 0.75
        self.turn_kd = 0.3
        self.turn_n = 20  # buffer size

        self.speed_kp = 1.75
        self.speed_ki = 1.0
        self.speed_kd = 2.0
        self.speed_n = 20  # buffer size

        # Numbers for the lateral PID controller
        self.lateral_k_p = 3.118357247806046
        self.lateral_k_d = 1.3782508892109167
        self.lateral_k_i = 0.6406067986034124
        self.lateral_speed_scale = 0.9755321901954155
        self.lateral_speed_offset = 1.9152884533402488
        self.lateral_default_lookahead = 24
        self.lateral_speed_threshold = 23.150102938235136
        self.lateral_n = 6
