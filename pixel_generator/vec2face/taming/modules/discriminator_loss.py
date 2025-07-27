import torch
import torch.nn as nn
import torch.nn.functional as F
from models import iresnet
from lpips.lpips import LPIPS
from pytorch_msssim import SSIM


def create_fr_model(model_path, depth="100", use_amp=True):
    model = iresnet(depth)
    model.load_state_dict(torch.load(model_path))
    if use_amp:
        model.half()
    return model


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, id_loss="mse",
                 fr_model="./weights/arcface-r100-glint360k.pth",
                 not_use_g_loss_adaptive_weight=False, use_amp=True):
        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.fr_model = create_fr_model(fr_model, use_amp=use_amp).eval()
        if id_loss == "mse":
            self.feature_loss = nn.MSELoss()
        elif id_loss == "cosine":
            self.feature_loss = nn.CosineSimilarity()
        self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        self.not_use_g_loss_adaptive_weight = not_use_g_loss_adaptive_weight

    def forward(self, im_features, gt_indices, logits, gt_img, image, emb_loss,
                epoch, mask=None):
        rec_loss = (image - gt_img) ** 2
        if epoch >= 0:
            gen_feature = self.fr_model(image)
            feature_loss = torch.mean(1 - torch.cosine_similarity(im_features, gen_feature))
        else:
            feature_loss = 0

        p_loss = self.perceptual_loss(image, gt_img) * 2

        with torch.cuda.amp.autocast(enabled=False):
            ssim_loss = 1 - self.ssim_loss((image.float() + 1) / 2, (gt_img + 1) / 2)

        if mask is None:
            token_loss = (logits[:, 1:, :] - gt_indices[:, 1:, :])
            token_loss = torch.mean(token_loss ** 2)
        else:
            token_loss = torch.abs((logits[:, 1:, :] - gt_indices[:, 1:, :])) * mask[:, 1:, None]
            token_loss = token_loss.sum() / mask[:, 1:].sum()

        nll_loss = torch.mean(rec_loss + 0.2 * p_loss) + \
                   ssim_loss + \
                   token_loss + feature_loss + emb_loss
        ae_loss = nll_loss

        return ae_loss, token_loss, rec_loss, ssim_loss, p_loss, feature_loss