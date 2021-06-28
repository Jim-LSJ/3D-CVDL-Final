import torch
from torch import nn

from smoke.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..heads.heads import build_heads


class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)

        in_channel = self.backbone.out_channels+32 if cfg.ADD_DEPTH else self.backbone.out_channels
        self.heads = build_heads(cfg, in_channel)

        if cfg.ADD_DEPTH:
            self.downsample = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )

    def forward(self, images, targets=None, depth=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        # depth = to_image_list(depth) if depth is not None else None
        features = self.backbone(images.tensors)
        # if depth is not None:
        #     depth_feat = self.downsample(depth.tensors)
        #     features = torch.cat((features, depth_feat), dim=1)

        result, detector_losses = self.heads(features, targets, depth)

        if self.training:
            losses = {}
            losses.update(detector_losses)

            return losses

        return result