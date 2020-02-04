import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z, train_test):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)

        if train_test == 'test':
            f = open("C:/Users/DELL/result_vector/result_vector_raw(10, 14).txt", "a")
            f.write(train_test+'\n')
            f.write(str(dist_a)+'\n')

        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
