import torch

from base_loss import BaseLoss

class CoVLoss(BaseLoss):
    def __init__(self, args):
        super(CoVLoss, self).__init__(args.lm_rdcution)
        
        self.mean_decay = True if args.mean_sort == 'decay' else False
        self.mean_decay_param = args.mean_decay_param
        
        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        
        # Initialize all running statics at 0
        self.running_mean_L = torch.zeros((self.num_lsses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.num_lsses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.num_lsses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None
    
    def forward(self, pred, target):
        unweighted_losses = super(CoVLoss, self).forward(pred, target)
        # 
        pass