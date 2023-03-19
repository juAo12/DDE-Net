#--------------------------------------------#
#   该部分代码用于看网络参数
#--------------------------------------------#
import torch
from thop import clever_format, profile
# from torchsummary import summary
from torchinfo import summary
from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 20
    # num_classes = 80

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNet_Resnet50(num_classes=num_classes).to(device)
    # model = CenterNet_HourglassNet({'hm': 80, 'wh': 2, 'reg': 2}, pretrained=False).to(device)
    # summary(model, (3, input_shape[0], input_shape[1]))
    summary(model, (1, 3, input_shape[0], input_shape[1]))

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

# if __name__ == '__main__':
#     model = CenterNet_Resnet50()
#     print(model)
#     test_data = torch.rand(5, 3, 512, 512)
#     out = model(test_data)
#     # model_with_loss =
#     a = out[0]
#     b = out[1]
#     c = out[2]
#     print(a.size())
#     print(b.size())
#     print(c.size())