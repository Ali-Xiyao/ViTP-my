from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class UnfreezeViTHook(Hook):
    def __init__(self, unfreeze_epoch=1,start=0.0, end=1.0,by_epoch=True):
        super().__init__()
        self.start = start
        self.end = end
        self.by_epoch = by_epoch
        self.param_groups = []  # 存储参数组索引和初始学习率
        self.layers_groups = []
        self.unfreeze_epoch=unfreeze_epoch

    def before_run(self, runner):
        
        optimizer = runner.optimizer
        model = runner.model
        param_to_name = {param: name for name, param in model.named_parameters()}
        
        # 识别embeddings和layers的参数组
        for group_idx, param_group in enumerate(optimizer.param_groups):
            
            if not param_group['params']:
                continue
            print('param_group[lr]:',param_group['lr'])
            for param_idx,sample_param in enumerate(param_group['params']): 
                param_name = param_to_name.get(sample_param, "")
                # print("param_name:"+param_name)
                if param_name.startswith('module.backbone.embeddings'):
                    self.param_groups.append([group_idx,param_idx])
                elif param_name.startswith('module.backbone.layers'):
                    self.param_groups.append([group_idx,param_idx])
        print('all_freeze_param: ',self.param_groups)
    def _get_lr_scale(self , progress):
        return self.start + (self.end - self.start) * progress
    
    def _freeze(self , runner):
        current_epoch = runner.epoch
        max_epochs = runner.max_epochs
        
        for group_idx,param_idx in self.param_groups:
            optimizer.param_groups[group_idx]['lr'][param_idx] = 0.

        

    def _adjust_lr(self, runner):
        current_epoch = runner.epoch
        print('current_epoch: ',current_epoch)
        max_epochs = runner.max_epochs
        progress = (current_epoch - self.unfreeze_epoch + 1) / (max_epochs - self.unfreeze_epoch)

        # 计算学习率调整系数
        factor = self._get_lr_scale(progress)#/self._get_lr_scale(progress_before)

        optimizer = runner.optimizer

        # 调整embeddings参数组
        for group_idx in self.param_groups:
            optimizer.param_groups[group_idx]['lr'] = optimizer.param_groups[group_idx]['lr'] * factor

    def before_train_epoch(self, runner):
        if self.by_epoch:
            if runner.epoch<self.unfreeze_epoch:
                self._freeze(runner)
            else:
                self._adjust_lr(runner)
        