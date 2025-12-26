from mmcv.runner import HOOKS, Hook
@HOOKS.register_module()
class UnfreezeViTHook(Hook):
    def __init__(self, unfreeze_epoch):
        self.unfreeze_epoch = unfreeze_epoch
    def before_train_epoch(self, runner):
        if runner.epoch == self.unfreeze_epoch:
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            backbone = model.backbone
            backbone.embeddings.train()
            for param in backbone.embeddings.parameters():
                param.requires_grad = True
            backbone.layers.train()
            for param in backbone.layers.parameters():
                param.requires_grad  = True
            # runner.optimizer.add_param_group({'params': model.backbone.embeddings.parameters()})
            # runner.optimizer.add_param_group({'params': model.backbone.layers.parameters()})