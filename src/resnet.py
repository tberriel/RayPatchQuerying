# Modified from oficial Pytorch code
from typing import Any, List, Type, Union
import torch
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class ResNetSkip(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs
    ) -> None:
        super().__init__(block, layers, **kwargs)
        del self.avgpool, self.fc

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        return [x_1, x_2, x_3, x_4]



def resnet18(
    pretrained: bool = True,
    progress: bool = True,
    **kwargs: Any,
) -> ResNetSkip:
 
    model = ResNetSkip(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth", progress=progress)
        weights.pop("fc.weight")
        weights.pop("fc.bias") 
        model.load_state_dict(weights)

    return model
