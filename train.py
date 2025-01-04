import dataclasses
import numpy as np
import pandas as pd
import torch



import einops

import torch
import torch.nn as nn

from architecture import ArchitectureConfig, DiT

from diffusion import Diffuzz


from tqdm import tqdm
import os
import transformers
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available
import wandb
import itertools

from collections import defaultdict
from typing import List, Dict, Any

from data import load_data, transform_data, ProteinStructureDataset


@dataclasses.dataclass
class DataLoaderConfig:
    seq_len:int = 256
    batch_size:int = 64
    num_workers:int = 1
    seed:int = 42

data_cfg = DataLoaderConfig()

@dataclasses.dataclass
class TrainingConfig:
    max_iterations:int = 20000
    train_log_freq:int = 100
    eval_freq:int = 1000
    max_val_iter:int = 10
    checkpoint_dir:str = "./exps3"
    checkpoint_freq:int = 2000

    lr:float = 1e-4
    warmup_percent:float = 0.05
    weight_decay:float =0.0
    grad_clip:float =1.0
    grad_accum_steps:int = 1
    compile:bool =True
    seed:int =42


## simple class to easily keep track of variables (without wandb)
class Logger():

    def __init__(self) -> None:
        self.logs = defaultdict(list)
        self.global_step = 0

    def log(self,params:Dict[str,Any]):

        for k,v in params.items():
            self.logs[k].append(v)

        self.logs["global_step"].append(self.global_step)
        self.global_step += 1



def get_checkpoint(train_cfg, arch_config):

    run_name= f"{DiT.model_name(arch_config)}-lr={train_cfg.lr}"
    checkpoint_path = os.path.join(train_cfg.checkpoint_dir, run_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    ## searching for checkpoints
    print(f"Searching checkpoints in {checkpoint_path}")
    checkpoints = [file for file in os.listdir(checkpoint_path) if '.pt' in file]
    if checkpoints:
        if "model.pt" in checkpoints:
            checkpoint_name= "model.pt"
            print("Previously trained model, training for more iterations")
        else:
            checkpoint_name = checkpoints[-1]


        checkpoint_path = os.path.join(checkpoint_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        print(f"Restarting from checkpoint {checkpoint_path}")
    else:
        checkpoint = None

    return checkpoint

def save_checkpoint(train_cfg, arch_config,data_cfg, checkpoint_name, model, optimizer, scheduler, scaler, it, run_id):

    run_name= f"{DiT.model_name(arch_config)}-lr={train_cfg.lr}"
    save_path = os.path.join(train_cfg.checkpoint_dir, run_name , checkpoint_name)
    print(f"checkpointing {save_path} ")

    torch.save({
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_last_step': scheduler.last_epoch,
        'iter': it,
        'grad_scaler_state_dict': scaler.state_dict(),
        "arch_config": arch_config,
        "train_config": train_cfg,
        "data_config": data_cfg,
        'wandb_run_id': run_id,
    }, save_path)



def get_model(train_cfg, arch_config, device):

    model = DiT(arch_config)

    checkpoint = get_checkpoint(train_cfg, arch_config)


    model_state_dict = {k.replace("_orig_mod.", ""):v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(model_state_dict)

    model.to(device)

    if train_cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+


    return model


## DATALOADER
def get_data_loader(data_cfg, df, split="train", shuffle=True):
    dataset = ProteinStructureDataset(df[df.split == split],
                                    max_seq_length=data_cfg.seq_len)

    g = torch.Generator()
    g.manual_seed(data_cfg.seed)

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=data_cfg.batch_size, num_workers=data_cfg.num_workers, generator=g, shuffle=shuffle)
    
    return data_loader




def eval(model, diffuzz, val_iter, device, _float16_dtype, max_val_iter=10):


    pbar = tqdm(range(max_val_iter))


    loss_cum=0
    max_l1_error_cum=0

    for it in pbar:
        batch = next(val_iter)
        batch = transform_data(batch, data_cfg)
        attn_masks = batch["attn_masks"].to(device)
        batch_size=attn_masks.shape[0]
        atom_mask= batch["atom_mask"].to(device)
        batch_size = atom_mask.shape[0]
        

        with torch.no_grad():
    
            ## forward diffusion
            input = einops.rearrange(batch["atom_positions"], "batch seq_len num_atoms coord -> batch seq_len (num_atoms coord)").to(device)
            t = (1 - torch.rand(batch_size, device=device)).mul(1.08).add(0.001).clamp(0.001, 1.0) 
            noised_input, noise = diffuzz.diffuse(input, t)


            ### noise prediciton
            with torch.cuda.amp.autocast(dtype=_float16_dtype):
                pred_noise = model(noised_input, t, attn_masks)

                ## rearrange and masking unknown atoms
                pred_noise = einops.rearrange(pred_noise, "batch seq_len (num_atoms coord) -> batch seq_len num_atoms coord", num_atoms=4, coord=3)
                noise = einops.rearrange(noise, "batch seq_len (num_atoms coord) -> batch seq_len num_atoms coord", num_atoms=4, coord=3)


                loss = nn.functional.mse_loss(pred_noise, noise, reduction='none') ## not reducing over all dimensions, so that we can reweight the loss depending on the timestep
                weight = einops.rearrange(diffuzz.p2_weight(t), "batch_size -> batch_size 1 1 1")
                loss_adjusted = loss*weight
                loss_adjusted = loss_adjusted[atom_mask > 0] ## masking unknown atoms
                loss_adjusted = loss_adjusted.mean()

                loss_cum += loss_adjusted.item()
            
                max_l1_error = nn.functional.l1_loss(pred_noise, noise, reduction='none')[atom_mask > 0].mean()
                max_l1_error_cum += max_l1_error.item()


        ## logging
        logs={
            'val/loss_adjusted': loss_cum/ max_val_iter,
            "val/max_l1_error": max_l1_error_cum / max_val_iter
        }


    return logs




def train(arch_config, data_cfg, train_cfg, df):

    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)


    _float16_dtype = torch.float16 if not is_torch_bf16_gpu_available() else torch.bfloat16
    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    model = DiT(arch_config)

    print(f"Number of parameters: {model.get_num_params()/1e6:.2f}M")

    device= "cuda"
    model = model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.95))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=train_cfg.lr, total_steps=train_cfg.max_iterations, 
                                                                pct_start=train_cfg.warmup_percent, anneal_strategy="cos", 
                                                                cycle_momentum=False, div_factor=1e2, final_div_factor= 3)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda= lambda epoch: 1)
    scaler = torch.cuda.amp.GradScaler()
    diffuzz = Diffuzz(device=device)

    checkpoint = get_checkpoint(train_cfg, arch_config)


    ## loading checkpoint
    start_iter = 1
    if checkpoint is not None:
        # checkpoints from compiled model have _orig_mod keyword
        model_state_dict = {k.replace("_orig_mod.", ""):v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.last_epoch = checkpoint['scheduler_last_step']
        if 'grad_scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

        start_iter = checkpoint['scheduler_last_step'] * train_cfg.grad_accum_steps + 1


    wandb_run_name = f""

    wandb_run_name= f"{DiT.model_name(arch_config)}-lr={train_cfg.lr}-batch_size={data_cfg.batch_size}-seq_len={arch_config.seq_len}"
    run_id = checkpoint['wandb_run_id'] if checkpoint is not None else wandb.util.generate_id()
    wandb.init(project="protein-diffusion", name=wandb_run_name, entity="entropyy", id=run_id, resume="allow", group="nightly")


    train_loader = get_data_loader(data_cfg, df, split="train", shuffle=True)

    train_iter = iter(train_loader)

    len_train = len(train_loader)


    val_loader = get_data_loader(data_cfg, df, split="validation", shuffle=False)

    val_iter = iter(val_loader)

    val_iter = itertools.cycle(val_iter)


    pbar = tqdm(range(start_iter, train_cfg.max_iterations + 1))
    logger = Logger()


    if train_cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    for it in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            

        batch = transform_data(batch, data_cfg)
        attn_masks = batch["attn_masks"].to(device)
        batch_size=attn_masks.shape[0]
        atom_mask= batch["atom_mask"].to(device)


        ## forward diffusion
        with torch.no_grad():
            input = einops.rearrange(batch["atom_positions"], "batch seq_len num_atoms coord -> batch seq_len (num_atoms coord)").to(device)
            t = (1 - torch.rand(batch_size, device=device)).mul(1.08).add(0.001).clamp(0.001, 1.0) 
            noised_input, noise = diffuzz.diffuse(input, t)

        
        ### noise prediciton
        with torch.cuda.amp.autocast(dtype=_float16_dtype):
            pred_noise = model(noised_input, t, attn_masks)

            ## rearrange and masking unknown atoms
            pred_noise = einops.rearrange(pred_noise, "batch seq_len (num_atoms coord) -> batch seq_len num_atoms coord", num_atoms=4, coord=3)
            noise = einops.rearrange(noise, "batch seq_len (num_atoms coord) -> batch seq_len num_atoms coord", num_atoms=4, coord=3)


            loss = nn.functional.mse_loss(pred_noise, noise, reduction='none') ## not reducing over all dimensions, so that we can reweight the loss depending on the timestep
            weight = einops.rearrange(diffuzz.p2_weight(t), "batch_size -> batch_size 1 1 1")
            loss_adjusted = loss*weight
            loss_adjusted = loss_adjusted[atom_mask > 0] ## masking unknown atoms
            loss_adjusted = loss_adjusted.mean() / train_cfg.grad_accum_steps


        ## max_l1_error_metric
        with torch.no_grad():
            max_l1_error = nn.functional.l1_loss(pred_noise, noise, reduction='none')[atom_mask > 0].mean()



        ## backward prop
        scaler.scale(loss_adjusted).backward()
        if it % train_cfg.grad_accum_steps == 0 or it == train_cfg.max_iterations:

            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()


        ## checking if we have done a full epoch
        if it % (len_train-1) == 0:
            train_iter = iter(train_loader)


        ## logging
            
        if it % train_cfg.train_log_freq == 0:

            logs={
                "train/epoch": it // len_train,
                'train/loss_adjusted': loss_adjusted.item(),
                "train/max_l1_error": max_l1_error.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/total_steps': scheduler.last_epoch,
            }

            pbar.set_postfix(logs)

            logger.log(logs)

            wandb.log(logs)

        ## checkpoitning

        if it % train_cfg.checkpoint_freq == 0:
            
            save_checkpoint(train_cfg, arch_config,data_cfg, f"model_{it}.pt", model, optimizer, scheduler, scaler, it, run_id)


        if it % train_cfg.eval_freq == 0:
            model.eval()
            print("Evaluating....")
            with torch.no_grad():
                logs= eval(model,diffuzz, val_iter, device, _float16_dtype, max_val_iter=train_cfg.max_val_iter)
            logs["train/total_steps"] = scheduler.last_epoch
            model.train()
            wandb.log(logs)



    ## final checkpointing
    save_checkpoint(train_cfg, arch_config,data_cfg, f"model.pt", model, optimizer, scheduler, scaler, it, run_id)

    wandb.finish()

    return logger





if __name__ == "__main__":


    df = load_data()

    import time

    train_cfg = TrainingConfig()

    for n_heads, d_model in zip([6,8, 4],[384, 512, 256]):

        for n_layers in [8]:

            if n_layers == 8 and n_heads == 6 and d_model == 384:
                print("continue...")
                continue

            arch_config = ArchitectureConfig(
                n_layers=n_layers,
                n_heads=n_heads,
                d_model=d_model
            )   

            logger = train(arch_config, data_cfg, train_cfg, df)
            time.sleep(10)
