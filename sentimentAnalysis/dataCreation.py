import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        combined = positive + negative #combining the sentences
        words=set()#make a set of all the unique words 
        for sentence in combined :
            for word in sentence.split():
                words.add(word)
        
        sorted_list=sorted(list(words))#sort the unique list of words 

        word_to_int={}#declare a dictionary to store the word to int pairs 
        for i, c in enumerate(sorted_list):#this function helps by allowing us to use both the index and the value at the same time 
            word_to_int[c]=i+1
        
        unpadded=[]
        for sentence in combined:
            encoded=[]
            for word in sentence.split():
                encoded.append(word_to_int[word])
            unpadded.append(torch.tensor(encoded))#used to convert array to tensor because pad_sequence works only on tensor 
        
        return rnn.pad_sequence(unpadded, batch_first=True)#creates a tensor in the shape (batch_size,max_length ) and pads the shorter string with 0 
  

