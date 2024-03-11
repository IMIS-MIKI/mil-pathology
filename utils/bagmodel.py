import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import mil_pytorch.mil as mil

class BagModel(nn.Module):
    # Add inputs to select the models

    def __init__(self, prepNN="resnet34", afterNN="default", middleNN="Mean", nb_input_channels=1, nb_classes=2, confidence=False):
        super(BagModel, self).__init__()

        # Preparation layer
        if prepNN == "resnet34":
            list_resnet = list(models.resnet34(pretrained=True).children())[:-1]
            list_resnet[0] = nn.Conv2d(nb_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                             bias=False)
            self.prepNN = nn.Sequential(*list_resnet)
        elif prepNN.startswith("resnet18"):
            # Select pre-trainining
            if prepNN.endswith("pathology"):
                # Network pretrained in histology images from here:
                # https://github.com/ozanciga/self-supervised-histopathology
                path = 'Model Storage/pre_trained_model/tenpercent_resnet18.ckpt'
                list_resnet = self.load_weights(path)
            else:
                list_resnet = models.resnet18(pretrained=True)

            # 1-channel inputs is suggested here: https://datascience.stackexchange.com/questions/65783/pytorch-how-to-use-pytorch-pretrained-for-single-channel-image
            # if this works do it for every model
            # last layer (classification) is discarded?
            list_resnet = list(list_resnet.children())[:-1]
            if nb_input_channels != 3:
                # get weights from first layer (valid for 3 channel inputs)
                w = list_resnet[0].weight
                # replace first layer
                list_resnet[0] = nn.Conv2d(nb_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                # replace weights with mean of 3-channel weights (will this cast to nb_input_cannels>1 automatically?)
                list_resnet[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
            self.prepNN = nn.Sequential(*list_resnet)

        # Middle layer
        middleNNoutputmultiplier = 1
        if middleNN == "Mean":
            self.middleNN = torch.mean

        elif middleNN == "Max":
            self.middleNN = torch.max

        elif middleNN.startswith("MinMax"):

            n = 1
            if "_" in middleNN:
                n = int(middleNN.split("_")[-1])

            def minmaxn(t, *args, **kwargs):
                t_sorted, _ = torch.sort(t, *args, **kwargs)
                return torch.flatten(torch.cat((t_sorted[:n], t_sorted[-n:])))

            middleNNoutputmultiplier = 2 * n
            self.middleNN = minmaxn

        elif middleNN == "SelfAttention":
            self.middleNN = SelfAttention()

        # After layer
        if afterNN == "default":
            self.afterNN = nn.Sequential(
                # number of output nodes is number of classes + confidence node (if True)
                nn.Linear(512*middleNNoutputmultiplier, nb_classes+confidence),
                nn.Sigmoid()
            )

        self.model = mil.BagModel(self.prepNN, self.afterNN, self.middleNN)

    @staticmethod
    def load_weights(path):
        state = torch.load(path, map_location='cpu')
        state_dict = state['state_dict']

        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        def load_model_weights(model, weights):
            model_dict = model.state_dict()
            weights = {k: v for k, v in weights.items() if k in model_dict}
            if weights == {}:
                print('No weight could be loaded..')
            model_dict.update(weights)
            model.load_state_dict(model_dict)

            return model

        model = models.resnet18(pretrained=False)
        model = load_model_weights(model, state_dict)

        return model

class UnBagModel(nn.Module):
    def __init__(self, bagmodel):
        super(UnBagModel, self).__init__()

        # THIS IS WORK IN PROGRESS, NEEDS TO BE SPLIT BY BATCHES ETC.

        self.prepNN = torch.nn.Sequential(*(list(bagmodel.children())[0]))
        self.afterNN = torch.nn.Sequential(*(list(bagmodel.children())[1]))

    def forward(self, x):
        output = self.prepNN(x)
        output = self.afterNN(output[:,:,0,0])

        return output

class ConfidenceCrossEntropyLoss(nn.Module):
    # as in https://arxiv.org/pdf/1802.04865.pdf
    def __init__(self, loss_c_weight = 1):
        super(ConfidenceCrossEntropyLoss, self).__init__()
        self.loss_c_weight = torch.Tensor([loss_c_weight])

    def forward(self, input, target):
        device = target.device
        self.loss_c_weight = self.loss_c_weight.to(device)
        # calculate confidence loss
        loss_c = torch.mean(-torch.log(input[:, -1]))
        # calculate modified predictions
        # p′ = c · p + (1 − c)y
        input = input[:, :-1] * input[:, -1][:, None] + torch.sub(torch.Tensor([1]).to(device), input[:, -1])[:, None] * F.one_hot(target, num_classes=input.size()[1])
        # calculate task loss
        loss_t = F.cross_entropy(input, target)
        #print("loss_t", loss_t.item(), "loss_c", loss_c.item(), "loss_c_total", (loss_c * self.loss_c_weight).item(), "combined", (loss_t + (loss_c * self.loss_c_weight)).item())
        return loss_t + (loss_c * self.loss_c_weight)

# discuss usage of self-attention: source: https://github.com/gmum/Kernel_SA-AbMILP/blob/master/model.py
class SelfAttention(nn.Module):
    def __init__(self):          # Self attention basically models dependencies between instances
        super(SelfAttention, self).__init__()
        inputs = 512
        # query, key and value vectors for Self Attention.
        self.query_conv = nn.Conv1d(in_channels=inputs, out_channels=inputs // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=inputs, out_channels=inputs // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=inputs, out_channels=inputs, kernel_size=1)  # cHECK DIMENSIONS HERE (?)
        self.gamma = nn.Parameter((torch.zeros(1)).cuda())
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)).cuda())

    def forward(self, x):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att
