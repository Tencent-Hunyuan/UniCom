from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import Optional


logger = logging.get_logger(__name__)


class UniComConfig(PretrainedConfig):
    r"""
    Configuration class for UniCom model. Uses SigLIP2 compressed features as the
    image generation latent space (no VAE, no MoE).

    Architecture: Transfusion (Qwen 2.5 7B Instruct backbone) + SigLIP2 + AttentionCompressionNet.
    """

    model_type = "UniCom"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            # --- Qwen 2.5 7B Instruct defaults ---
            vocab_size: int = 153768,
            hidden_size: int = 3584,
            intermediate_size: int = 18944,
            num_hidden_layers: int = 28,
            num_attention_heads: int = 28,
            num_key_value_heads: Optional[int] = 4,
            attention_head_dim: Optional[int] = None,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            eod_token_id=3,
            im_start_id=4,
            im_end_id=5,
            text_start_id=6,
            text_end_id=7,
            image_token_id=8,
            video_start_id=9,
            video_end_id=10,
            im_newline_id=11,
            mask_init_id=12,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=1000000.0,
            rope_scaling=None,
            attention_bias=True,
            attention_o_proj_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            use_qk_norm=False,
            use_rotary_pos_emb=True,
            use_cla=False,
            cla_share_factor=1,
            norm_type="hf_rms",
            use_mla=False,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            v_head_dim=128,
            qk_nope_head_dim=128,
            add_classification_head=False,
            class_num=0,
            pool_type="last",
            pad_id=-1,
            # --- SigLIP / image generation ---
            siglip_downsample_factor=(16, 16),
            max_image_tokens=1024,
            image_base_size=1024,
            rope_type="default",
            cond_token_attn_type="full",
            cond_image_type="vit",
            siglip_latent_dim=64,
            siglip_original_dim=1152,
            compression_checkpoint_path=None,
            # --- ViT (SigLIP2-so400m-patch16-naflex) ---
            vit_type="siglip2-so400m-patch16-naflex",
            vit=None,
            vit_processor=None,
            vit_aligner=None,
            # --- model flags ---
            cfg_distilled=False,
            use_meanflow=False,
            use_compile=False,
            model_version="UniCom-Instruct",
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.model_version = model_version
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        else:
            self.attention_head_dim = self.hidden_size // num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_o_proj_bias = attention_o_proj_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.use_cla = use_cla
        self.cla_share_factor = cla_share_factor
        self.norm_type = norm_type
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.add_classification_head = add_classification_head
        self.class_num = class_num
        self.pool_type = pool_type
        self.pad_id = pad_id

        if self.class_num is not None:
            self.dense_list = [self.hidden_size, self.class_num]

        # SigLIP / image generation
        self.siglip_downsample_factor = siglip_downsample_factor
        self.max_image_tokens = max_image_tokens
        self.siglip_latent_dim = siglip_latent_dim
        self.siglip_original_dim = siglip_original_dim
        self.compression_checkpoint_path = compression_checkpoint_path
        self.image_base_size = image_base_size
        self.rope_type = rope_type
        self.cond_token_attn_type = cond_token_attn_type
        self.cond_image_type = cond_image_type

        # ViT
        self.vit_type = vit_type
        self.vit = vit
        self.vit_processor = vit_processor
        self.vit_aligner = vit_aligner

        # token ids
        self.eod_token_id = eod_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.text_start_id = text_start_id
        self.text_end_id = text_end_id
        self.image_token_id = image_token_id
        self.video_start_id = video_start_id
        self.video_end_id = video_end_id
        self.im_newline_id = im_newline_id
        self.mask_init_id = mask_init_id

        # model flags
        self.cfg_distilled = cfg_distilled
        self.use_meanflow = use_meanflow
        self.use_compile = use_compile

        kwargs.setdefault("auto_map", {
            "AutoConfig": "configuration_unicom.UniComConfig",
            "AutoModel": "modeling_unicom.UniComModel",
            "AutoModelForCausalLM": "modeling_unicom.UniComForCausalMM",
        })
        kwargs.setdefault("architectures", ["UniComForCausalMM"])

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["UniComConfig"]
