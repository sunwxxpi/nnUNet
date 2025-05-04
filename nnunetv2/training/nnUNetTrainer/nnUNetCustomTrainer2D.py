import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.focal_tversky_loss import TverskyLoss, FocalTverskyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP

class nnUNetCustomTrainer2D(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Learning Rate
        self.initial_lr = 1e-2
        
    # def _do_i_compile(self):
        # return False
        
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            
            from torchinfo import summary
            torchinfo_summary = str(summary(self.network, 
                input_size=(16, 1, 512, 512),
                col_width=20, 
                depth=10, 
                row_settings=["depth", "var_names"], 
                col_names=["input_size", "kernel_size", "output_size", "params_percent"]))
            
            output_file = "model_summary_2d.txt"
            with open(output_file, "w") as file:
                file.write(torchinfo_summary)
            
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
        
    '''def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        
        # CE Loss
        ce_loss = RobustCrossEntropyLoss(
            weight=None, ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        # Tversky Loss
        tversky_loss = TverskyLoss(
            alpha=0.6, beta=0.4
        )
        
        """ # Focal Tversky Loss
        focal_tversky_loss = FocalTverskyLoss(
            alpha=0.6, beta=0.4, gamma=4/3
        ) """
        
        if self._do_i_compile():
            tversky_loss = torch.compile(tversky_loss)
            
        def combined_loss(output, target):
            return ce_loss(output, target) + tversky_loss(output, target)
        
        # Deep Supervision 적용
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            combined_loss = DeepSupervisionWrapper(combined_loss, weights)

        return combined_loss'''