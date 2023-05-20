
import torch.nn as nn

class SNN(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, r=0.25):
        super(SNN, self).__init__(in_features, out_features, bias)
        self.r = r                #relevance ratio
        self.s = int(self.r * in_features)
        print(f"Starting SNN training. Using subspace dimension = {self.s}")
        
    def forward(self, input):
        vote = input[:, None, :] * self.weight
        if self.bias is not None:
            out = vote.topk(self.s, 2)[0].sum(2) + self.bias
        else:
            out = vote.topk(self.s, 2)[0].sum(2)
        return out