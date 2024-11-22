from torch.nn import Module

class MSDLoss(Module):
    '''
        Mean of Squared Differences

        MSD = mean( x_i - y_i)
    '''

    def __init__(self):
        super().__init__()


    def forward(self, pred, gt):
        B    = gt.size(0)
        loss = (gt - pred).pow(2).view(B, -1).mean()

        return loss
