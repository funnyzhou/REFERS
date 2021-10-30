import torch
import torch.nn as nn
import torch.nn.functional as F


import pdb

class LossesOfConVIRT(nn.Module):
    """

    """

    def __init__(self, tau=0.1, lambd=0.75):
        super(LossesOfConVIRT, self).__init__()
        self.tau = tau
        self.lambd = lambd

    def tmp_loss(self, v, u, index):
        """

        """
        assert v.size(0) == u.size(0)
        # cos = torch.nn.CosineSimilarity(dim=0)
        item1 = torch.exp(torch.divide(torch.cosine_similarity(v[index], u[index], dim=0), self.tau))
        # cos2 = torch.nn.CosineSimilarity(dim=1)
        item2 = torch.exp(torch.divide(torch.cosine_similarity(v[index].unsqueeze(0), u, dim=1), self.tau)).sum()
        loss = -torch.log(torch.divide(item1, item2))
        return loss

    def image_text(self, v, u, index):
        """

        """
        assert v.size(0) == u.size(0)
        cos = torch.nn.CosineSimilarity(dim=0)
        item1 = torch.exp(torch.divide(cos(v[index], u[index]), self.tau))
        # for k in range(v.size(0)):
        #     item2 += torch.exp(torch.divide(cos(v[index], u[k]), self.tau))
        cos2 = torch.nn.CosineSimilarity(dim=1)
        item2 = torch.exp(torch.divide(cos2(v[index].unsqueeze(0), u), self.tau)).sum()
        loss = -torch.log(torch.divide(item1, item2))
        return loss

    def text_image(self, v, u, index):
        """

        """
        assert v.size(0) == u.size(0)
        # pdb.set_trace()
        cos = torch.nn.CosineSimilarity(dim=0)
        item1 = torch.exp(torch.divide(cos(v[index], u[index]), self.tau))
        # item2 = 0.0
        # for k in range(v.size(0)):
        #     item2 += torch.exp(torch.divide(cos(v[k], u[index]), self.tau))

        cos2 = torch.nn.CosineSimilarity(dim=1)
        item2 = torch.exp(torch.divide(cos2(v, u[index].unsqueeze(0)), self.tau)).sum()
        loss = -torch.log(torch.divide(item1, item2)).item()
        return loss

    def forward(self, v, u):
        """

        :return:
        """
        # pdb.set_trace()
        # print("v:",v.size())
        # print("u:",u.size())
        assert v.size(0) == u.size(0)
        res = 0.0
        v = v.float()
        u = u.float()
        for i in range(v.size(0)):
            # res += self.lambd * self.tmp_loss(v, u, i) + (1 - self.lambd) * self.tmp_loss(u, v, i)
            res += self.lambd * self.image_text(v, u, i) + (1 - self.lambd) * self.text_image(v, u, i)
        # pdb.set_trace()
        res /= v.size(0)
        return res

def test():
    v = torch.randn(size=(100, 768))
    u = torch.randn(size=(100, 768))
    criterion = LossesOfConVIRT()

    loss = criterion(v, u)
    print("hello")
    print(loss)
    pass


if __name__ == '__main__':
    test()