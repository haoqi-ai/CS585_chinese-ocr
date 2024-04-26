import argparse
import random
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.data

import crnn.AttnCRNN as crnn
import src.dataset as dataset
import src.utils as utils

random.seed(42)
cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--train_list', type=str, help='path to training data_path - label pairs')
parser.add_argument('--val_list', type=str, help='path to validation data_path - label pairs')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=3200, help='batch size')
parser.add_argument('--img_height', type=int, default=32, help='height of the input image')
parser.add_argument('--img_width', type=int, default=280, help='width of the input image')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--num_epochs', type=int, default=2, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--encoder', type=str, default='', help='path to the encoder')
parser.add_argument('--decoder', type=str, default='', help='path to the decoder')
parser.add_argument('--model', default='./model/', help='path to the derived models')
parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--teacher_forcing_prob', type=float, default=0.5, help='degree of dependence on teacher forcing')
parser.add_argument('--max_width', type=int, default=71, help='width of the output feature map from cnn')
args = parser.parse_args()
print(args)


# load alphabet
with open('./data/alphabet.txt') as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)

# define convert bwteen string and label index
converter = utils.Converter(alphabet)

# len(alphabet) + SOS_TOKEN + EOS_TOKEN
num_classes = len(alphabet) + 2


def train(image, text, encoder, decoder, criterion, train_loader, teach_forcing_prob=0.1):
    # optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # loss
    loss_avg = utils.Averager()
    losses = []

    for epoch in range(args.num_epochs):
        train_iter = iter(train_loader)

        for i in range(len(train_loader)):
            cpu_images, cpu_texts = train_iter.next()
            batch_size = cpu_images.size(0)

            for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
                encoder_param.requires_grad = True
                decoder_param.requires_grad = True
            encoder.train()
            decoder.train()

            target_variable = converter.encode(cpu_texts)
            utils.load_data(image, cpu_images)

            # CNN + BiLSTM
            encoder_outputs = encoder(image)
            target_variable = target_variable.cuda()
            # start by passing SOS_TOKEN in
            decoder_input = target_variable[utils.SOS_TOKEN].cuda()
            decoder_hidden = decoder.initHidden(batch_size).cuda()
            
            loss = 0.0
            teach_forcing = True if random.random() > teach_forcing_prob else False
            if teach_forcing:
                for di in range(1, target_variable.shape[0]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])
                    decoder_input = target_variable[di]
            else:
                for di in range(1, target_variable.shape[0]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze()
                    decoder_input = ni
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            losses.append(loss_avg.val())
            loss_avg.add(loss)

            if i % 10 == 0:
                print('[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4}'.format(epoch, args.num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        # save checkpoint
        torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(args.model, epoch))
        torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(args.model, epoch))

    # plt.figure()
    # plt.plot(losses, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss During Training')
    # plt.legend()
    # plt.show()


def evaluate(image, text, encoder, decoder, data_loader, max_eval_iter=100):
    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.Averager()

    for i in range(min(len(data_loader), max_eval_iter)):
        cpu_images, cpu_texts = val_iter.next()
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1

        decoded_words = []
        decoded_label = []
        encoder_outputs = encoder(image)
        target_variable = target_variable.cuda()
        decoder_input = target_variable[0].cuda()
        decoder_hidden = decoder.initHidden(batch_size).cuda()

        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == utils.EOS_TOKEN:
                decoded_label.append(utils.EOS_TOKEN)
                break
            else:
                decoded_words.append(converter.decode(ni))
                decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1

        if i % 10 == 0:
            texts = cpu_texts[0]
            print('pred: {}, gt: {}'.format(''.join(decoded_words), texts))

    accuracy = n_correct / float(n_total)
    print('Test loss: {}, accuray: {}'.format(loss_avg.val(), accuracy))


def main():
    if not os.path.exists(args.model):
        os.makedirs(args.model)

    # create train dataset
    train_dataset = dataset.TextLineDataset(pth=args.train_list, transform=None)
    sampler = dataset.RandomSequentialSampler(train_dataset, args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=int(args.num_workers),
        collate_fn=dataset.Align(img_height=args.img_height, img_width=args.img_width))

    # create test dataset
    test_dataset = dataset.TextLineDataset(text_line_file=args.eval_list, transform=dataset.ResizeNormalize(img_width=args.img_width, img_height=args.img_height))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=int(args.num_workers))

    # create network
    encoder = crnn.Encoder(channel_size=3, hidden_size=args.hidden_size)
    # 'translator'
    decoder = crnn.Decoder(hidden_size=args.hidden_size, output_size=num_classes, dropout_p=0.1, max_lrngth=args.max_width)
    # print(encoder)
    # print(decoder)
    encoder.apply(utils.weights_init)
    decoder.apply(utils.weights_init)
    if args.encoder:
        print(f"pretrained encoder model loaded from {args.encoder}")
        encoder.load_state_dict(torch.load(args.encoder))
    if args.decoder:
        print(f"pretrained encoder model loaded from {args.decoder}")
        decoder.load_state_dict(torch.load(args.decoder))

    # create input tensor
    image = torch.FloatTensor(args.batch_size, 3, args.img_height, args.img_width)
    text = torch.LongTensor(args.batch_size)

    criterion = torch.nn.NLLLoss()

    assert torch.cuda.is_available(), "Device Error: GPU needed!"
    encoder.cuda()
    decoder.cuda()
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

    train(image, text, encoder, decoder, criterion, train_loader, teach_forcing_prob=args.teacher_forcing_prob)
    evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=100)


if __name__ == "__main__":
    main()
