import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu
from model.cmatt import CMAtten 

class Bert(nn.Module):
    """ Finetuned *BERT module """

    def __init__(self, tokenizer, lan='RoBERTa'):
        super(Bert, self).__init__()
  
        if lan == 'BERT':
            from transformers import BertTokenizer, BertModel, BertConfig
            config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
            self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        elif lan == 'RoBERTa':
            from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
            config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
            self.bert = RobertaModel.from_pretrained("roberta-base", config=config)
        self.tokenizer = tokenizer
        
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False
        
    def forward(self, tokens):
        attention_mask = (tokens != self.tokenizer.pad_token_id).float()
        outs = self.bert(tokens, attention_mask=attention_mask)
        embds = outs[0]
        return embds, outs[1][-2]


class Sentence_Maxpool(nn.Module):
    """ Utilitary for the answer module """

    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x_in):
        x = self.fc(x_in)
        x = torch.max(x, dim=1)[0]
        if self.relu:
            x = F.relu(x)
        return x


class FFN(nn.Module):
    def __init__(self, word_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        activation = "gelu"
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=word_dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class AModel(nn.Module):
    """
    Answer embedding module
    """

    def __init__(self, tokenizer, lan='RoBERTa', word_dim=768, out_dim=512):
        super(AModel, self).__init__()
        self.bert = Bert(tokenizer, lan=lan)
        self.linear_text = nn.Linear(word_dim, out_dim)

        # self.linear_text = FFN(word_dim, out_dim, out_dim)
        
    def forward(self, answer):
        
        #import ipdb
        #ipdb.set_trace()
        
        if len(answer.shape) == 3:
            #multi-choice
            bs, nans, lans = answer.shape         # 64 ,5 , 38  
            answer = answer.view(bs * nans, lans) # 320, 38
            answer, hd_state = self.bert(answer)  # 320, 38, 768 / 768
            answer = self.linear_text(answer)     # 320, 38, 512     이 layer 없애보고 성능 뽑아서 비교해보기
            answer_g = answer.mean(dim=1)         # 320, 512         모든 단어를 평균 왜지?
            # answer_g = answer[:, 0, :]
            answer_g = answer_g.view(bs, nans, -1) # 64, 5, 512
        else:
            answer, hd_state = self.bert(answer)
            answer = self.linear_text(answer)
            answer_g = answer.mean(dim=1)
            # answer_g = answer[:, 0, :]
        
        return answer_g, answer

