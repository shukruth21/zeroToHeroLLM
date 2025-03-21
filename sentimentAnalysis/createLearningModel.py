import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer=nn.Embedding(vocabulary_size,16)#creating an embedding table where each word is embedded using 16 size vector 
        self.linear=nn.Linear(16,1)#linear regression layer that solves for 16 parameters to give single output 
        self.sigmoid=nn.Sigmoid()#to give output in the range 0-1
        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embeddings=self.embedding_layer(x)#maps all the words by embedding it suppose x has 2 sentences with max length 5 then table is 2x5x16
        averaged=torch.mean(embeddings,axis=1)#take the average of the words in each sentence  2x16
        projected=self.linear(averaged)#puts the values from embedding table for each sentence into the linear regression equation to get a single value outpue 2x1

        return torch.round(self.sigmoid(projected),decimals=4)#output between 0 to 1
        pass
