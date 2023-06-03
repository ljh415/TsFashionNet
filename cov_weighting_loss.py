import torch

from base_loss import BaseLoss

class CoVLoss(BaseLoss):
    def __init__(self, lm_reduction, shape_only=False, mean_sort='full', mean_decay_param=1.0, class_weight=None, smoothing=0.0):
        super(CoVLoss, self).__init__(lm_reduction, shape_only, class_weight, smoothing)
        
        self.shape = shape_only
        
        self.mean_decay = True if mean_sort == 'decay' else False
        self.mean_decay_param = mean_decay_param
        
        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        
        # Initialize all running statics at 0
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None
    
    def forward(self, preds, targets, mode):
        # [cat_loss, att_loss, vis_loss, lm_loss]
        unweighted_losses = list(super(CoVLoss, self).forward(preds, targets, self.shape))
        
        # if shape:
        #     zero_tensor = [torch.zeros(1).to(self.device) for _ in range(2)]
        #     unweighted_losses = zero_tensor + unweighted_losses
        # if shape 분기문은 새롭게 다 추가한 부분

        if self.shape :
            vis_loss, lm_loss = unweighted_losses
        else :
            cat_loss, att_loss, vis_loss, lm_loss = unweighted_losses
        
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)
        
        ## validation 일땐 그대로 sum해서 return 
        if mode != 'train':
            if self.shape:
                return torch.sum(L), vis_loss, lm_loss
            else:
                return torch.sum(L), cat_loss, att_loss, vis_loss, lm_loss
        
        # increase iter
        self.current_iter += 1
        
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        
        # comput loss ration for current iteration given the current loss L
        l = L / L0
        
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device) / self.num_losses
        # apply the loss weighing method
        else :
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply welford's algorithm to keep running means, variances of L, l.
        # 1. Compute 
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else :
            mean_param = (1. - 1 / (self.current_iter +1))
        
        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # the variance is S / (t-1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)
        
        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L
        
        # get the weighted losses and perform a standard back-pass
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        
        # if self.shape:
        #     self.running_mean_L[:2] = 1e-8
        #     self.running_mean_l[:2] = 1e-8
        #     self.running_S_l[:2] = 1e-8
        #     if self.running_std_l is not None:
        #         self.running_std_l[:2] = 1e-8
        
        if self.shape:
            return loss, vis_loss, lm_loss, self.alphas
        else :
            return loss, cat_loss, att_loss, vis_loss, lm_loss, self.alphas