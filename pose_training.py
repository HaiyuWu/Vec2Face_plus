#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pose_training.py – Vec2Face pose‑condition fine‑tuning with optional LoRA

import os, sys, time, argparse
from datetime import datetime, timedelta
from collections import defaultdict
import math
import torch, torch.nn as nn, torch.optim as optim, torch.distributed as dist
import torchvision.transforms as T
import numpy as np, imageio
from tqdm import tqdm
from lpips.lpips import LPIPS
from pytorch_msssim import SSIM
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import pixel_generator.vec2face.model_vec2face as model_vec2face
from pixel_generator.vec2face.pose_condition import PoseCondModel
from pixel_generator.vec2face.lora import LoRALinear
from dataloader.training_loader_landmarks import LMDBDataLoader
from models import iresnet

# --------------------------------------------------------------------------- #
# Force NCCL to use localhost / loopback so single‑node jobs don’t hang
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")


# --------------------------------------------------------------------------- #


class AverageMeter:
    def __init__(self): self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, v, n=1):
        self.val = v;
        self.sum += v * n;
        self.count += n;
        self.avg = self.sum / self.count


# --------------------------------------------------------------------------- #
def get_args_parser():
    p = argparse.ArgumentParser("Vec2Face verify / fine‑tune script")

    # ---------- basic ----------
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--model', default='vec2face_vit_base_patch16')
    p.add_argument('--epochs', default=100, type=int)

    # ---------- pre‑trained representation flags ----------
    p.add_argument('--use_rep', action='store_false')
    p.add_argument('--use_class_label', action='store_true')
    p.add_argument('--rep_dim', default=512, type=int)
    p.add_argument('--rep_drop_prob', default=0., type=float)

    # ---------- masking ----------
    p.add_argument('--mask_ratio_min', type=float, default=0.6,
                        help='Minimum mask ratio')
    p.add_argument('--mask_ratio_max', type=float, default=0.8,
                        help='Maximum mask ratio')
    p.add_argument('--mask_ratio_mu', type=float, default=0.7,
                        help='Mask ratio distribution peak')
    p.add_argument('--mask_ratio_std', type=float, default=0.1,
                        help='Mask ratio distribution std')

    # ---------- file paths ----------
    p.add_argument('--model_weights', default='')
    p.add_argument('--start_end', default=None)  # (kept)
    p.add_argument('--train_source', required=True)
    p.add_argument('--mask', default=None)  # (kept)
    p.add_argument('--output_dir', default='./output_dir')
    p.add_argument('--log_dir', default='./output_dir')

    # ---------- device / distribution ----------
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--resume', default='')
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--workers', default=4, type=int)
    p.add_argument('--pin_memory', action='store_false')
    p.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    p.add_argument('--use_amp', action='store_false')
    p.add_argument('--amp_dtype', type=lambda x: getattr(torch, x), default='float16')
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--local-rank', dest='local_rank', type=int, default=-1)  # torchrun compat
    p.add_argument('--dist_on_itp', action='store_true')
    p.add_argument('--dist_url', default='env://')

    # ---------- LoRA ----------
    p.add_argument('--use_lora', action='store_true')
    p.add_argument('--lora_r', type=int, default=4)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.)

    # ---------- optimiser & LR schedule ----------
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--min_lr', default=1e-6, type=float, help='LR at the end of cosine schedule')
    p.add_argument('--warmup_epochs', default=5, type=int, help='Linear warm-up length')
    p.add_argument('--cosine_lr', action='store_true', help='Enable cosine annealing schedule')
    p.add_argument('--weight_decay', default=0.05, type=float)

    # ---------- misc ----------
    p.add_argument('--save_freq', default=10, type=int)
    p.add_argument('--print_freq', default=20, type=int)

    args = p.parse_args()
    if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    return args


def is_main(args):  # helper for “rank‑0 or single process”
    return (not args.distributed) or args.rank == 0


# --------------------------------------------------------------------------- #
def setup_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        args.distributed = False
        args.rank = 0
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()


def create_fr_model(path='./weights/arcface-r100-glint360k.pth',
                    depth='100', use_amp=True):
    m = iresnet(depth)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    if use_amp: m.half()
    m.eval()
    return m


def freeze_except_lora(model):
    for p in model.parameters():  # freeze everything
        p.requires_grad_(False)
    for m in model.modules():  # un‑freeze LoRA A/B
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)


# --------------------------------------------------------------------------- #
def main():
    args = get_args_parser()
    torch.manual_seed(args.seed)

    # ---- distributed init ----
    if args.local_rank != -1:
        setup_distributed(args)
    else:
        args.distributed = False
        args.rank = 0
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- output dir ----
    if is_main(args):
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- instantiate models ----
    pose_model = PoseCondModel().to(device)
    fr_model = create_fr_model(use_amp=args.use_amp).to(device)

    vec2face = model_vec2face.__dict__[args.model](
        mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
        mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
        use_rep=args.use_rep, rep_dim=args.rep_dim, rep_drop_prob=args.rep_drop_prob,
        use_class_label=args.use_class_label,
        use_lora=args.use_lora, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_drop=args.lora_dropout
    ).to(device)

    # optional warm‑start
    if args.model_weights:
        if is_main(args): print(f'Loading weights from {args.model_weights}')
        st = torch.load(args.model_weights, map_location=device)
        vec2face.load_state_dict(st['model_vec2face'], strict=False)
        if 'pose_model' in st:
            pose_model.load_state_dict(st['pose_model'], strict=False)

    # freeze heavy backbone, enable LoRA
    freeze_except_lora(vec2face)
    vec2face.eval()

    # ---- DDP wrapper for pose_model (BEFORE optimizer) ----
    if args.distributed:
        pose_model = torch.nn.parallel.DistributedDataParallel(
            pose_model, device_ids=[args.gpu], find_unused_parameters=False
        )

    # ---- build optimiser *after* DDP wrap ----
    trainable = list(pose_model.parameters()) + [
        p for p in vec2face.parameters() if p.requires_grad
    ]
    optimizer = optim.AdamW(trainable, lr=args.lr,
                            weight_decay=args.weight_decay)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # ---- resume ----
    start_epoch = args.start_epoch
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        vec2face.load_state_dict(ckpt['model_vec2face'], strict=False)
        if isinstance(pose_model, nn.parallel.DistributedDataParallel):
            pose_model.module.load_state_dict(ckpt['pose_model'], strict=False)
        else:
            pose_model.load_state_dict(ckpt['pose_model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if is_main(args): print(f'Resumed from {args.resume} (epoch {start_epoch})')

    # ---- dataset & dataloader ----
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    dataset = LMDBDataLoader(args, transform=tfm)
    loader = dataset.get_loader()

    # ---- LR scheduler (optional cosine) ----
    if args.cosine_lr:
        steps_per_epoch = len(loader)
        warmup_iters = args.warmup_epochs * steps_per_epoch
        total_iters = args.epochs * steps_per_epoch

        if warmup_iters == 0:  # no warm-up requested
            scheduler = CosineAnnealingLR(
                optimizer, T_max=total_iters, eta_min=args.min_lr
            )
        else:
            warmup_sched = LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0,
                total_iters=warmup_iters
            )
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=total_iters - warmup_iters,
                eta_min=args.min_lr
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_iters]
            )
    else:
        scheduler = None

    # ---- losses ----
    perceptual = LPIPS().to(device)
    ssim_fn = SSIM(data_range=1, size_average=True, channel=3).to(device)

    if is_main(args):
        print('==> Begin fine‑tuning PoseCondModel + LoRA adapters')

    # ----------------------------------------------------------------------- #
    for epoch in range(start_epoch, args.epochs):
        if args.distributed and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

        meters = defaultdict(AverageMeter)
        t0 = time.time()
        len_loader = len(loader)

        pose_model.train()
        vec2face.eval()

        for step, (im, feat, _, landmark, cur_base) in enumerate(loader):
            im, feat = im.to(device, non_blocking=True), feat.to(device, non_blocking=True)
            landmark = landmark.to(device, non_blocking=True)
            cur_base = cur_base.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp,
                                         dtype=args.amp_dtype):
                pfeat = pose_model(landmark)
                # fused = (gamma * pfeat + (1 - gamma) * cur_base) + beta
                _, _, gen_img, _, _ = vec2face(feat, pfeat)
                # _, _, gen_img, _, _ = vec2face(feat)

                rec = (gen_img - im).pow(2).mean()
                fid = (1 - torch.cosine_similarity(feat, fr_model(gen_img))).mean()
                ploss = perceptual(gen_img, im).mean() * 2
                with torch.cuda.amp.autocast(enabled=False):
                    ssim_l = 1 - ssim_fn((gen_img + 1) / 2, (im + 1) / 2)

                total = rec + 0.2 * ploss + ssim_l + fid

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            # update meters
            meters['rec'].update(rec.item())
            meters['fid'].update(fid.item())
            meters['ploss'].update(ploss.item())
            meters['ssim'].update(ssim_l.item())
            meters['total'].update(total.item())

            if is_main(args) and step % args.print_freq == 0:
                elapsed = time.time() - t0
                eta_s = ((len_loader - step - 1) + (args.epochs - epoch - 1) * len_loader) \
                        * elapsed / max(1, step + 1)
                eta = str(timedelta(seconds=int(eta_s)))
                mem = torch.cuda.max_memory_allocated() / 1024 ** 2
                cur = datetime.now().strftime("%H:%M:%S")

                print(f"[{cur}] E:{epoch} {step}/{len_loader} "
                      f"eta:{eta} lr:{optimizer.param_groups[0]['lr']:.2e} "
                      f"tot:{meters['total'].val:.3f} "
                      f"rec:{meters['rec'].val:.3f} fid:{meters['fid'].val:.3f} "
                      f"p:{meters['ploss'].val:.3f} ssim:{meters['ssim'].val:.3f} "
                      f"mem:{mem:.0f}MB",
                      flush=True)

        # ---- save outputs & checkpoint every epoch ----
        if is_main(args):
            ep_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(ep_dir, exist_ok=True)
            vis = ((gen_img[:32].permute(0, 2, 3, 1).cpu() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
            raw = ((im[:32].permute(0, 2, 3, 1).cpu() + 1) / 2 * 255).byte().numpy()
            for i in range(vis.shape[0]):
                imageio.imwrite(f"{ep_dir}/gt_{i:03}.jpg", raw[i])
                imageio.imwrite(f"{ep_dir}/gen_{i:03}.jpg", vis[i])

            if (epoch % args.save_freq == 0) or (epoch + 1 == args.epochs):
                ckpt = dict(
                    epoch=epoch,
                    model_vec2face=vec2face.state_dict(),
                    pose_model=(pose_model.module if isinstance(pose_model,
                                                                nn.parallel.DistributedDataParallel) else pose_model).state_dict(),
                    optimizer=optimizer.state_dict(),
                    args=args,
                )
                torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_{epoch}.pth"))
                torch.save(ckpt, os.path.join(args.output_dir, "checkpoint_latest.pth"))
                print(f"[save] checkpoint_{epoch}.pth")

    if is_main(args):
        print("Training complete.")


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()
