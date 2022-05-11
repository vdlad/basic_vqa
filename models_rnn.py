import torch
from torch._C import set_anomaly_enabled
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from typing import Tuple, Union
import torch.nn.functional as F


class BiRnnEncoder(nn.Module): 
  def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size): 
    super(BiRnnEncoder, self).__init__()
    self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
    self.tanh = nn.Tanh()
    self.forward_lstm = nn.GRU(word_embed_size, hidden_size)
    self.image_encoder = ImgEncoderForRnn(hidden_size, hidden_size) # this is essentially a hidden layer so sizes are hiidden size
    self.backward_lstm = nn.GRU(hidden_size + word_embed_size, embed_size)

  def forward(self, img, qst): 
    qst_vec = self.word2vec(qst)                             # [batch_size, max_qst_length=30, word_embed_size=300]
    qst_vec = self.tanh(qst_vec)
    qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
    output, hidden = self.forward_lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]  
    hidden_img_encoding = self.image_encoder(img, output)     #batch_size x hidden_size
    flipped_encoding = torch.flip(hidden_img_encoding, (0,))
    flipped_vec = torch.flip(qst_vec, (0,))
    concat_inputs = torch.cat((flipped_encoding, flipped_vec), dim=2)
    concat_inputs = self.tanh(concat_inputs)
    output, hidden = self.backward_lstm(concat_inputs)
    return hidden[0,:,:]

class ImgEncoderForRnn(nn.Module): 
    def __init__(self, embed_size, hidden_size):
        super(ImgEncoderForRnn, self).__init__()
        model = models.resnet152(pretrained=True)
        in_features  = model.fc.in_features
        model.fc = nn.Identity() 
        self.model = model 
        self.fc = nn.Linear(in_features + hidden_size, embed_size)
        self.hidden_size = hidden_size 


    def forward(self, image, hidden_state):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        # print(img_feature.shape)
        # print(hidden_state.shape)
        img_feature = img_feature.repeat(hidden_state.shape[0], 1, 1)
        all_features = torch.cat((img_feature, hidden_state), dim=2)
        #print(all_features.shape)
        encoding = self.fc(all_features)                   # [batch_size, embed_size]
        #l2_norm = encoding.norm(p=2, dim=1, keepdim=True).detach()
        #img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return encoding

class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        self.qst_encoder = BiRnnEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):
        combined_feature = self.qst_encoder(img, qst)                     # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature
