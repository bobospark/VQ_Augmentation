import torch
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
import datasets
from datasets import disable_caching
from datasets import load_dataset
# disable_caching()
from datasets import list_datasets

class Set_Dataset(Dataset):
    def __init__(self, args, train_eval):
        if args.dataset == 'sst2':
            self.dataset = datasets.load_dataset("glue", "sst2")
        self.train_datset = self.dataset['train']
        self.eval_dataset = self.dataset['validation']
        self.train_eval = train_eval

    def forward(self, args):
        # if train_val == 
        self.data = []
        self.tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
        # with open(path, 'r') as f:
        if self.train_eval == 'train':
            for line in self.train_dataset:
                text, label = line.strip().split('\t')
                self.data.append((text, int(label)))
        elif self.train_eval == 'eval':
            for line in self.eval_dataset:
                text, label = line.strip().split('\t')
                self.data.append((text, int(label)))

        self.dataset_ = self.data
        self.data_loader = DataLoader(self.dataset_, batch_size = args.batch_size, shuffle = True)
    
        return self.data_loader


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        attention_mask = [1] * len(input_ids)
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(label)

