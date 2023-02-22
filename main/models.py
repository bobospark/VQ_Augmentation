import torch
import torch.nn as nn
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments
from transformers import AutoModel
from transformers import DistilBertTokenizer


class VQAugmentation(nn.Module):
    def __init__(self, args):
        super(VQAugmentation, self).__init__()
        self.model_tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
        
        self.model = TrainingArguments(output_dir = args.model_name,
                                        num_train_epochs = )
        
    def forward(self, x):
        