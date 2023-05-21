import os
from run_imagenet import run_func
from utils.imagenet_ood_loader import get_loader_in, get_loader_out
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
from models import resnet_in
save_name = f"cache_r_35"

batch_size = 256 #batch size

torch.cuda.empty_cache()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
    model = resnet_in.resnet(r = 0.35, depth = 101, pretrained =  False, num_classes = 100)
    checkpoint = torch.load('./checkpoints/in_100/r_35/best.pth')
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    print("Model loaded")
    model.cuda()
    model.eval() 
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model


loader_in_dict = get_loader_in(batch_size, config_type="eval")
trainloaderIn, testloaderIn = loader_in_dict.train_loader, loader_in_dict.val_loader

num_classes = 100
model = get_model()


featdim = 2048

FORCE_RUN = False
ID_RUN = True
OOD_RUN = True
print(f"Num classes : {num_classes}")
if ID_RUN:
    for split, in_loader in [('val', testloaderIn), ('train', trainloaderIn)]:

        cache_dir = f"./cache/{save_name}/IN-100_{split}_in"
        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), featdim))
            score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset),))

            model.eval()
            for batch_idx, (inputs, targets) in enumerate(in_loader):
               
                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))
                out, score = model.get_feat(inputs)
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)}")
        else:
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), featdim))
            score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset),))

if OOD_RUN:
    out_datasets = ['inat', 'sun50', 'places50', 'dtd']
    for ood_dataset in out_datasets:
        out_loader= get_loader_out(batch_size, ood_dataset)
        cache_dir = f"./cache/{save_name}/{ood_dataset}vsIN_100_out"
        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), featdim))
            ood_score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), num_classes))
            model.eval()
            for batch_idx, (inputs, _) in enumerate(out_loader):
               
                inputs = inputs.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))
                out, score = model.get_feat(inputs)
                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(out_loader)}")


        else:
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), featdim))
            ood_score_log = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), num_classes))


print("Running Inference")
run_func(save_name, out_datasets)