import os
import argparse
import numpy
import torch

from test_data import Set_Dataset
from tqdm import tqdm
from test_model import VQAugmentation_test
from sklearn.metrics import accuracy_score

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Test_training:
    def __init__(self, args):
        self.models = VQAugmentation_test(args).to(device = args.device)
        self.optim = self.configure_optimizers()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt = torch.optim.AdamW(
            lr = lr
        )

        return opt

    def train(self, args):

        train_loader = Set_Dataset(args, 'train')
        eval_loader = Set_Dataset(args, 'eval')
        # eval_dataset, eval_loader = Set_Dataset(args, 'validation')
        # train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

        for epoch in range(args.epochs):
            train_losses = []
            for batch in tqdm(train_loader):
                input_ids, attention_mask, labels = batch
                self.optim.zero_grad()
                outputs = self.model(input_ids, attention_mask = attention_mask, labels = labels)
                loss = outputs.loss
                train_losses.append(loss.item())
                loss.backward()
                self.optim.step()
            print(f'Epoch {epoch + 1}, train loss: {sum(train_losses) / len(train_losses):.4f}')

        self.models.eval()
        eval_losses = []
        eval_labels = []
        eval_preds = []
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                eval_losses.append(loss.item())
                eval_labels.extend(labels.tolist())
                eval_preds.extend(logits.argmax(dim=1).tolist())

        eval_acc = accuracy_score(eval_labels, eval_preds)
        print(f'Eval loss: {sum(eval_losses) / len(eval_losses):.4f}, eval acc: {eval_acc:.4f}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQAugmentation")
    parser.add_argument('--model-name', type = str, default = '"distilbert-base-uncased"', help = 'Base model (default : distilbert)')
    parser.add_argument('--dataset', type = str, default = 'sst2', help = "Choose the dataset")
    # parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    # parser.add_argument('--image-size', type=int, default=512, help='Image height and width (default: 256)') # BERT 최대 length = 512
    # parser.add_argument('--num-codebook-vectors', type=int, default=4096, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--num-labels', type = int, default = 2, help = 'number of classification (sst-2 : 2)')
    # parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    # parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    # parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    # parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    # parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    # parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    # parser.add_argument('--disc-factor', type=float, default=1., help='')
    # parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    # parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    # parser.add_argument('--resume-from', type = str, default = "vqgan_epoch_9.pt", help = 'Resume the training if it is true')

    args = parser.parse_args()
    train_vqAugmentation = Test_training(args)