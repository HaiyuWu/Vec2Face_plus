import torch
import argparse
import pixel_generator.vec2face.model_vec2face as model_vec2face
from pixel_generator.vec2face.pose_condition import PoseCondModel
import imageio
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from glob import glob
import os
from models import iresnet
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T


class ImageDataset(Dataset):
    """Simple image dataset that either returns tensors or (tensor, path)."""

    def __init__(self, image_paths, use_path: bool = True):
        self.image_paths = image_paths
        self.use_path = use_path
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if not self.use_path:  # already a tensor
            return self.image_paths[idx].half()

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img).half(), img_path


# ---------------------------------------------------------------------------
#                               Utility helpers
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("Vec2Face – multi‑GPU inference", add_help=False)
    parser.add_argument("--batch_size", default=64, type=int, help="Effective batch size per forward pass (will be split across GPUs if >1)")

    # Model parameters
    parser.add_argument("--model", default="vec2face_vit_base_patch16", type=str, metavar="MODEL", help="Vec2Face backbone")
    parser.add_argument("--image_file", required=True, type=str, help="Folder/file with reference images")
    parser.add_argument("--input_size", default=112, type=int, help="Input resolution expected by FR model")

    # Conditioning options
    parser.add_argument("--use_rep", action="store_false", help="Use identity representation embedding as condition")
    parser.add_argument("--use_class_label", action="store_true", help="Use integer class labels instead of embeddings")
    parser.add_argument("--rep_dim", default=512, type=int)

    # Pixel Generation options
    parser.add_argument("--rep_drop_prob", default=0.0, type=float)
    parser.add_argument("--feat_batch_size", default=2000, type=int, help="Batch size when extracting reference features")
    parser.add_argument("--example", default=50, type=int, help="Samples to generate per reference")
    parser.add_argument("--name", default="run", type=str, help="Sub‑directory name used when saving images")
    parser.add_argument("--pose_file", required=True, type=str, help="the .txt file that contains the landmark images")

    # Masking hyper‑params (kept identical to original script)
    parser.add_argument("--mask_ratio_min", type=float, default=0.8)
    parser.add_argument("--mask_ratio_max", type=float, default=0.9)
    parser.add_argument("--mask_ratio_mu", type=float, default=0.85)
    parser.add_argument("--mask_ratio_std", type=float, default=0.05)

    # Checkpoint
    parser.add_argument("--model_weights", required=True, type=str, help="Path to checkpoint with Vec2Face + Pose modules")

    # LoRA (optional)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument('--start_end', default=None,
                        help='slicing dataset generation')
    parser.add_argument('--start_idx', default=0,
                        help='start_idx')

    return parser


def get_device():
    """Return primary device and number of visible CUDA GPUs."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[Info] Found {n} CUDA device(s)")
        return torch.device("cuda"), n
    print("[Warning] CUDA not available – running on CPU only")
    return torch.device("cpu"), 0


def maybe_parallel(model: torch.nn.Module, device: torch.device, ngpu: int):
    """Wrap model in DataParallel if more than one GPU is available."""
    model = model.to(device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def save_images(images: np.ndarray, id_num, root: str, name: str):
    """Write images to disk grouped by identity."""
    global _save_counter, _prev_id
    save_root = os.path.join(root, name)
    for i, image in enumerate(images):
        save_folder = os.path.join(save_root, id_num[i])
        os.makedirs(save_folder, exist_ok=True)
        if _prev_id != id_num[i]:
            _prev_id = id_num[i]
            _save_counter = 0
        imageio.imwrite(os.path.join(save_folder, f"{_save_counter:03d}.jpg"), image)
        _save_counter += 1


# initialise globals for save_images()
_save_counter = 0
_prev_id= ""


# ---------------------------------------------------------------------------
#                       Data loading / preprocessing helpers
# ---------------------------------------------------------------------------

def load_paths(file_path: str):
    """List all image files from a directory or a text file."""
    if os.path.isdir(file_path):
        return sorted(glob(os.path.join(file_path, "*")))
    if os.path.isfile(file_path):
        return np.genfromtxt(file_path, dtype=str)
    raise FileNotFoundError("Provided --image_file is neither a directory nor a file")


def processing_images(ref_images, feature_model=None, pose: bool = False, use_path: bool = True, batch_size: int = 256, device: torch.device = "cuda"):
    """Extract face features or return tensors for pose images."""
    dataset = ImageDataset(ref_images, use_path)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    if pose:  # For pose reference we only need the tensor batch (no paths)
        all_tensors = []
        for data in loader:
            all_tensors.append(data[0])  # (tensor, path)
        return torch.cat(all_tensors, dim=0)

    # otherwise: run FR model to obtain embeddings
    features, im_ids = [], []
    with torch.no_grad():
        for batch, paths in tqdm(loader, desc="Extracting FR features"):
            batch = batch.to(device, non_blocking=True)
            feats = feature_model(batch)
            features.append(feats.cpu())
            im_ids.extend([os.path.basename(os.path.dirname(p)) for p in paths])
    return torch.cat(features, dim=0), im_ids


# ---------------------------------------------------------------------------
#                           Face recognition backbone
# ---------------------------------------------------------------------------

def _create_fr_model(model_path: str, depth: str = "100"):
    model = iresnet(depth)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.half().eval()  # inference‑only
    return model


# ---------------------------------------------------------------------------
#                                   Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    parser = get_args_parser()
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Environment set‑up
    # ---------------------------------------------------------------------
    device, ngpu = get_device()
    torch.backends.cudnn.benchmark = True  # speedup for fixed input size

    # ---------------------------------------------------------------------
    # Build models
    # ---------------------------------------------------------------------
    print("[Info] Loading Vec2Face backbone …")
    vec2face = model_vec2face.__dict__[args.model](
        mask_ratio_mu=args.mask_ratio_mu,
        mask_ratio_std=args.mask_ratio_std,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        use_rep=args.use_rep,
        rep_dim=args.rep_dim,
        rep_drop_prob=args.rep_drop_prob,
        use_class_label=args.use_class_label,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_drop=args.lora_dropout,
    )

    pose_model = PoseCondModel()
    fr_model = _create_fr_model("./weights/arcface-r100-glint360k.pth")

    # Move to device (+ replicate if multiple GPUs)
    print("[Info] Loading weights from", args.model_weights)
    ckpt = torch.load(args.model_weights, map_location=device)

    vec2face.load_state_dict(ckpt["model_vec2face"])
    # vec2face.load_state_dict(strong_ckpt["model_vec2face"], strict=False)
    pose_model.load_state_dict(ckpt["pose_model"])

    vec2face = maybe_parallel(vec2face, device, ngpu)
    pose_model = maybe_parallel(pose_model, device, ngpu)
    fr_model = maybe_parallel(fr_model, device, ngpu)

    vec2face.eval()
    pose_model.eval()
    fr_model.eval()

    # ---------------------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------------------
    ref_paths = load_paths(args.image_file)
    if args.start_end is not None:
        start, end = args.start_end.split(":")
        assert int(end) > int(start)
    else:
        start, end = 0, len(ref_paths)

    ref_paths = ref_paths[int(start):int(end)]

    # Face embeddings of reference identities (batched to avoid OOM)
    reference_ids, im_ids = processing_images(
        ref_paths,
        feature_model=fr_model,
        pose=False,
        use_path=True,
        batch_size=args.feat_batch_size,
        device=device,
    )

    pose_landmark = processing_images(
        load_paths(args.pose_file),
        pose=True,
        use_path=True,
        batch_size=args.batch_size,
        device=device,
    )

    pose_landmark = pose_landmark.to(device, non_blocking=True)

    samples = torch.repeat_interleave(reference_ids, args.example, dim=0).to(device, non_blocking=True)
    im_ids = [im_id for im_id in im_ids for _ in range(args.example)]

    assert args.batch_size % len(pose_landmark) == 0
    pose_landmark = pose_landmark.repeat(args.batch_size // len(pose_landmark), 1, 1, 1)

    print("[Info] Starting generation …")
    torch.cuda.empty_cache()

    with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
        for i in tqdm(range(0, len(samples), args.batch_size)):
            im_features = samples[i : i + args.batch_size]
            pose_feats = pose_model(pose_landmark[: im_features.size(0)]).float()
            _, _, imgs, *_ = vec2face(im_features, pose_feats)
            imgs_np = ((imgs.permute(0, 2, 3, 1).detach().cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            save_images(imgs_np, im_ids[i : i + args.batch_size], "generated_images_ref", args.name)


if __name__ == "__main__":
    main()
