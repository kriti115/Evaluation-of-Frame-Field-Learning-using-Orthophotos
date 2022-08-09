import torch
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
# from pytorch_memlab import profile, profile_every
from frame_field_learning import tta_utils
#from backbone import get_backbone


def get_out_channels(module):
    if hasattr(module, "out_channels"):
        return module.out_channels
    children = list(module.children())
    i = 1
    out_channels = None
    while out_channels is None and i <= len(children):
        last_child = children[-i]
        out_channels = get_out_channels(last_child)
        i += 1
    # If we get out of the loop but out_channels is None, then the prev child of the parent module will be checked, etc.
    #print('out channels', out_channels) # None, None, None, then 5 times 64
    return out_channels 


class FrameFieldModel(torch.nn.Module):
    def __init__(self, config: dict, backbone, train_transform=None, eval_transform=None):
        """

        :param config:
        :param backbone: A _SimpleSegmentationModel network, its output features will be used to compute seg and framefield.
        :param train_transform: transform applied to the inputs when self.training is True
        :param eval_transform: transform applied to the inputs when self.training is False
        """
        super(FrameFieldModel, self).__init__()
        assert config["compute_seg"] or config["compute_crossfield"], \
            "Model has to compute at least one of those:\n" \
            "\t- segmentation\n" \
            "\t- cross-field"
        assert isinstance(backbone, _SimpleSegmentationModel), \
            "backbone should be an instance of _SimpleSegmentationModel"
        self.config = config
        self.backbone = backbone
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        backbone_out_features = get_out_channels(self.backbone)
        #print('backbone_out_features : ', backbone_out_features ) # 64

        # --- Add other modules if activated in config:
        seg_channels = 0
        if self.config["compute_seg"]:
            seg_channels = self.config["seg_params"]["compute_vertex"]\
                           + self.config["seg_params"]["compute_edge"]\
                           + self.config["seg_params"]["compute_interior"]
            self.seg_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, seg_channels, 1),
                torch.nn.Sigmoid(),)

        if self.config["compute_crossfield"]:
            crossfield_channels = 4
            self.crossfield_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features + seg_channels, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, crossfield_channels, 1),
                torch.nn.Tanh(),
            )

    def inference(self, image):
        outputs = {}

        # --- Extract features for every pixel of the image with a U-Net --- #
        backbone_features = self.backbone(image)["out"]

        if self.config["compute_seg"]:
            # --- Output a segmentation of the image --- #
            seg = self.seg_module(backbone_features)
            seg_to_cat = seg.clone().detach()
            backbone_features = torch.cat([backbone_features, seg_to_cat], dim=1)  # Add seg to image features
            outputs["seg"] = seg

        if self.config["compute_crossfield"]:
            # --- Output a cross-field of the image --- #
            crossfield = 2 * self.crossfield_module(backbone_features)  # Outputs c_0, c_2 values in [-2, 2]
            outputs["crossfield"] = crossfield
        
        #print('output from inference in model.py Line 88 :', outputs) #{'seg': tensor([[[[0.4366, 0.3507, 0.3625,  ..., 0.4431, 0.4449, 0.4102],
                                                                                          # [0.4482, 0.3590, 0.3713,  ..., 0.4379, 0.4434, 0.4543],
                                                                                          # [0.4463, 0.3513, 0.3750,  ..., 0.5071, 0.5018, 0.5062],...
                                                                      #       'crossfield': tensor([[[[ 7.2890e-01, -4.8401e-02, -1.8504e-01,  ...,  8.4678e-02,
                                                                                                  #   2.4399e-02, -2.2948e-02],
                                                                                                  # [ 7.6350e-01,  2.8245e-01,  2.1134e-01,  ...,  2.9804e-01,
                                                                                                  #   2.0644e-01,  9.3794e-02],
        
        return outputs

    # @profile
    def forward(self, xb, tta=False):
        # print("\n### --- PolyRefine.forward(xb) --- ####")
        if self.training:
            if self.train_transform is not None:
                xb = self.train_transform(xb)
        else:
            if self.eval_transform is not None:
                xb = self.eval_transform(xb)
        
        #print('xb from model.py Line 100 :',xb) # batches of content from '.pt' files in processed folder
        
        if not tta:
            final_outputs = self.inference(xb["image"])
        else:
            final_outputs = tta_utils.tta_inference(self, xb, self.config["eval_params"]["seg_threshold"])

            # Save image
            # image_seg_display = plot_utils.get_tensorboard_image_seg_display(image_display, final_outputs["seg"],
            #                                                                  crossfield=final_outputs["crossfield"])
            # image_seg_display = image_seg_display[1].cpu().detach().numpy().transpose(1, 2, 0)
            # skimage.io.imsave(f"out_final.png", image_seg_display)

        return final_outputs, xb
