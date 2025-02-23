from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from .aliases import PathOrStr
from .exceptions import OLMoConfigurationError
from .util import StrEnum
from .config import (
    BaseConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataConfig,
    EvaluatorConfig,
    TokenizerConfig,
    WandbConfig,
    SpeedMonitorConfig,
    CompilerConfig,
    DistributedStrategy,
    FSDPConfig,
    DDPConfig,
    SingleGPUConfig,
    ActivationCheckpointingStrategy,
    FSDPPrecision,
    ShardedCheckpointerType,
    PaddingDirection,
    InstanceFilterConfig,
    CustomDatasetConfig,
)


__all__ = [
    "KASTrainConfig",
    "KASEvaluatorConfig",
    "KASEvaluatorType",
    "KASDataConfig"
]

class KASEvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"

@dataclass
class KASDataConfig(BaseConfig):
    paths: Optional[List[str]] = None
    memmap_dtype: str = "uint16"
    datasets: Optional[Dict[str, List[str]]] = None
    label_mask_paths: Optional[List[str]] = None
    pad_direction: PaddingDirection = PaddingDirection.right
    generate_attention_mask: bool = False
    generate_doc_lengths: bool = False
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0
    seed: Optional[int] = None
    instance_filter: Optional[InstanceFilterConfig] = None
    custom_dataset: Optional[CustomDatasetConfig] = None

    @property
    def effective_memmap_dtype(self):
        try:
            # getattr will check this is part of numpy module, while np.dtype will check
            # if this is a valid numpy dtype.
            np.dtype(dtype := getattr(np, self.memmap_dtype))
        except (AttributeError, TypeError) as e:
            raise TypeError(f"Value {self.memmap_dtype} is not a valid numpy type") from e
        return dtype

@dataclass
class KASEvaluatorConfig(BaseConfig):
    label: str
    type: KASEvaluatorType = KASEvaluatorType.lm
    data: KASDataConfig = field(default_factory=KASDataConfig)
    device_eval_batch_size: Optional[int] = None
    subset_num_batches: Optional[int] = None

@dataclass
class KASTrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    """
    The name of the run.
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    epoch: Optional[int] = None
    """
    Increment this when starting a new epoch.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """
    OLMo Model configuration.
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """
    Training data configuration.
    """

    restore_dataloader: bool = True
    """
    When restarting, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    fast_forward_batches: Optional[int] = None
    """
    When restarting, use this to fast-forward the dataloader beyond the last checkpoint.
    This can be useful when restarting due to a loss spike in order to skip the data that
    corresponded to the spike.
    """

    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    """
    Evaluation configurations.
    """

    eval_interval: int = 1000
    """
    How often (in terms of batches) to run evaluations.
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    """
    Tokenizer configuration.
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    remote_save_folder: Optional[str] = None
    """
    A folder in a cloud bucket to upload saved checkpoints to.
    """

    canceled_check_interval: int = 50
    """
    How often (in batches) to check if the run has been canceled or reached its time limit.
    """

    save_interval: Optional[int] = 1000
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_interval_unsharded: Optional[int] = None
    """
    How often (if at all) to save unsharded training state checkpoint.
    For large models it can be costly to save these, so it usually makes sense to save
    these less often than regular (sharded) training checkpoints.
    """

    save_interval_ephemeral: Optional[int] = None
    """
    How often (if at all) to save ephemeral sharded checkpoints. These checkpoints are the same
    as those saved every `save_interval` except that at most only the most recent one of these is kept.
    This is useful when you want to checkpoint often for restarts in case of failures, but don't
    want to keep the majority of these checkpoints.

    For example, suppose you want to keep your checkpoints at every 1000 steps, but you also want to save
    a temporary checkpoint every 100 steps in case your job fails. In that case you would
    set `save_interval=1000` and `save_interval_ephemeral=100`.
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many sharded checkpoints to keep.
    """

    save_num_unsharded_checkpoints_to_keep: int = -1
    """
    How many unsharded checkpoints to keep.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite any conflicting checkpoint files.
    """

    force_save_unsharded: bool = False
    """
    Save an unsharded checkpoint before training (even during a dry run).
    Use this option with `--load-path={PATH}` and `--dry_run` to convert a sharded
    checkpoint into an unsharded checkpoint.
    """

    no_pre_train_checkpoint: bool = False
    """
    Skip saving pre-train checkpoint.
    """

    load_path: Optional[str] = None
    """
    The path to a training checkpoint to restore/resume from. If not set, then training begins from scratch.

    Note that you can make use of the "path.last_checkpoint" Omegaconfig YAML resolver here, which takes
    a local or remote directory and resolves to the latest checkpoint (sharded or unsharded) in that directory.
    For example,

    ```bash
    --load_path='${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/v1_5-mix-run-001}'
    ```

    If `try_load_latest_save` is set and saved checkpoints exist, then `load_path` will be overriden
    by the latest saved checkpoint.
    """

    load_path_sharded_checkpointer: Optional[ShardedCheckpointerType] = None
    """
    The sharded checkpointer type to use to load the initial checkpoint from ``load_path``.
    """

    try_load_latest_save: bool = False
    """
    If set, then training will be resumed from the latest checkpoint in the local save folder, falling
    back to the latest checkpoint in the remote save folder if none exists. If there are no checkpoints
    in the local and remote save folders, then checkpoint loading will fall back to `load_path`.
    """

    reset_optimizer_state: bool = False
    """
    When this is set, we restore the model from a checkpoint (if given), but we leave the optimizer uninitialized.
    We also set a new learning rate schedule that does a new warmup, such that it intercepts the original learning
    curve (according to the current learning rate schedule settings), and continues from there.
    """

    reset_trainer_state: bool = False
    """
    When this is set we don't restore the trainer state from a checkpoint.
    """

    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.torch_legacy
    """
    The name of the sharded checkpointer to use to save (sharded) checkpoints throughout training.
    """

    new_style_checkpoints: Optional[bool] = None
    """
    Deprecated. Use ``sharded_checkpointer`` instead.
    """

    max_duration: Union[int, str] = 10000
    """
    How long to train for.

    If specified without a unit (the default), the units are assumed to be steps.
    You can also specify this in terms of tokens, for example: `max_duration="2e12T"` means train until
    2 trillion tokens.
    """

    global_train_batch_size: int = 512
    """
    The effective global batch size.
    """

    device_train_batch_size: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``global_train_batch_size // world_size``.
    """

    device_train_microbatch_size: int = 16
    """
    The number of instances passed to the model in a single forward-backward pass. You should set
    this as large as you can based on available GPU memory.
    """

    device_eval_batch_size: int = 16
    """
    The number of evaluation instances passed to the model in a single forward pass on each device.
    """

    eval_subset_num_batches: int = -1
    """
    The number of batches to use for downstream evaluation from each dataset.
    """

    eval_on_load: bool = False
    """
    When resuming from a checkpoint, run the evaluation loop right away.
    """

    device_train_grad_accum: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``device_train_batch_size // device_train_microbatch_size``.
    """

    max_grad_norm: Optional[float] = None
    """
    Clip gradient norms to this value if set.
    """

    max_grad_norm_ratio: Optional[float] = None
    """
    If set, gradient norms will be clipped to `max_grad_norm_ratio * exp_avg(norm(grad))`.
    This takes priority over `max_grad_norm` when set.
    """

    precision: Optional[str] = None
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    gen1_gc_interval: Optional[int] = 1
    """
    How often (in steps) to run generation 1 garbage collection.
    Set to ``None`` to use automatic garbage collection (i.e. we don't mess with it).
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    distributed_strategy: Optional[DistributedStrategy] = DistributedStrategy.fsdp
    """
    Distributed strategy for OLMo model (eg. single GPU, DDP, FSDP).
    """

    fsdp: Optional[FSDPConfig] = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    ddp: Optional[DDPConfig] = None
    """
    DDP settings.
    """

    single: SingleGPUConfig = field(default_factory=lambda: SingleGPUConfig(device="auto"))
    """
    Single device settings for GPU/CPU/MPS. Defaults to auto-detect the best device.
    """

    softmax_auxiliary_loss: bool = False
    """
    If ``True``, we add the auxiliary loss function from PaLM that encourages the softmax
    normalizing term to be close to 0.
    """

    auxiliary_loss_multiplier: Optional[float] = 1e-4
    """
    Used with `softmax_auxiliary_loss`. PaLM uses 1e-4, Chameleon uses 1e-5.
    """

    time_limit: Optional[float] = None
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.
    """

    extra_steps_after_cancel: int = 10
    """
    Under certain conditions when a run is canceled we train for a few extra steps after saving
    the final checkpoint so that when the run is restarted from the latest checkpoint we have some
    overlap in metrics.
    """

    early_stopping_factor: Optional[float] = None

    save_data_indices: bool = True
    """
    Save training data indices from each batch for each worker.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = False
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[int] = None
    """
    Stop at a specific step.
    """

    stop_after: Optional[int] = None
    """
    Stop after a specific number of steps.
    """

    activation_checkpointing: Optional[ActivationCheckpointingStrategy] = None
    """
    The activation checkpointing strategy to use.
    """

    fused_loss: Optional[bool] = None
    """
    Whether to use the fused CE loss function from `flash-attn`.
    """

    hf_datasets_cache_dir: Optional[str] = None
    """
    Deprecated, HF datasets are now stored in `olmo_data.hf_datasets`.

    Path to cache directory of HF datasets saved with `datasets.save_to_disk`.
    """

    module_outputs_save_steps: Optional[List[int]] = None
    """
    Outputs of model submodules are saved during the provided steps. Submodule outputs
    can be compared using `scripts/compare_module_outputs.py`.
    """

    debug: bool = False
    """
    If ``True``, run in non-distributed set up (this is for debug purposes in VSCode).
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @property
    def fsdp_precision(self) -> Optional[MixedPrecision]:
        if self.fsdp is not None:
            if self.fsdp.precision is None:
                return None
            elif self.fsdp.precision == FSDPPrecision.pure:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=self.autocast_precision,
                    buffer_dtype=self.autocast_precision,
                )
            elif self.fsdp.precision == FSDPPrecision.mixed:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=torch.float32,
                    buffer_dtype=self.autocast_precision,
                )
            else:
                raise NotImplementedError(f"{self.fsdp.precision}")
        else:
            raise ValueError("self.fsdp is None!")

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "activation_checkpointing"):
                if new_config.activation_checkpointing is False:
                    new_config.activation_checkpointing = None
                if new_config.activation_checkpointing is True:
                    new_config.activation_checkpointing = ActivationCheckpointingStrategy.whole_layer

            if hasattr(new_config, "optimizer"):
                new_config.optimizer = OptimizerConfig.update_legacy_settings(new_config.optimizer)

        return new_config
