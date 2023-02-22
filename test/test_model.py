import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification
import torch.nn as nn

class VQAugmentation_test(nn.Module):
    def __init__(self, args):
        super(VQAugmentation_test, self).__init__()
        
        if args.dataset == 'sst2':
            self.model = DistilBertForSequenceClassification.from_pretrained(args.model_name,
                                                            num_labels = args.num_labels)
            
            # self.model.to(args.device)
            # self.model.eval()
            # self.model.zero_grad()

    def forward(self, x):
        return self.model#.train()