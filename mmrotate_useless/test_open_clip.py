import open_clip
from safetensors import safe_open
import torch
from mmrotate.models.backbones import CLIPViT
# pretrained = 'pretrained/mae_pretrain_vit_large.pth'
# checkpoint = torch.load(pretrained, map_location='cpu')
# checkpoint = checkpoint['model']
# print(checkpoint.keys())
# print(open_clip.list_pretrained())
model=CLIPViT(model_name='ViT-L-14',pretrained='pretrained/RS5M_ViT-L-14.pt',pretrained_type= 'CLIP')
print(model.model.visual.positional_embedding)
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='/nfs/liyuxuan/zhangyicheng/mmrotate/pretrained/clip_vit_l_14_laion2B.bin') 

# pretrained = '/nfs/liyuxuan/zhangyicheng/mmrotate/pretrained/clip_vit_large_patch14.safetensors'
# with safe_open(pretrained, framework="pt") as f:
#     print(f.keys())
