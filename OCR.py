import argparse
from PIL import Image
import torch

import crnn.AttnCRNN as crnn
import src.dataset as dataset
import src.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='', help='path of the input image')
parser.add_argument('--img_height', type=int, default=32, help='height of the input image')
parser.add_argument('--img_width', type=int, default=280, help='width of the input image')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--encoder', type=str, default='./model/encoder.pth', help='path to the encoder')
parser.add_argument('--decoder', type=str, default='./model/decoder.pth', help='path to the decoder')
parser.add_argument('--max_width', type=int, default=71, help='width of the output feature map from cnn')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu')
args = parser.parse_args()


# load alphabet
with open('./data/alphabet.txt', encoding='utf-8') as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)

# define convert bwteen string and label index
converter = utils.Converter(alphabet)

num_classes = len(alphabet) + 2 # len(alphabet) + SOS_TOKEN + EOS_TOKEN

transformer = dataset.ResizeNormalize(img_width=args.img_width, img_height=args.img_height) 


# 'translate' the sentence
def seq2seq(encoder_output, decoder, decoder_input, decoder_hidden, max_length):
    decoded_words = []
    prob = 1.0
    for _ in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
        probs = torch.exp(decoder_output)
        print('probs', probs)
        _,  topi = decoder_output.data.topk(1)
        print('topi', topi)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        print('prob', prob)
        if ni == utils.EOS_TOKEN:
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()
    return words, prob


def model(img):
    device = 'cpu'
    if torch.cuda.is_available() and args.use_gpu:
        device = 'cuda'
    # image = Image.open(args.img_path).convert('RGB')
    image = img.convert('RGB')
    image = transformer(image)  # resize & normalize
    image = image.to(device)
    image = image.view(1, *image.size())
    image = torch.autograd.Variable(image) 

    encoder = crnn.Encoder(3, args.hidden_size).to(device)
    decoder = crnn.Decoder(args.hidden_size, num_classes, dropout_p=0.0, max_length=args.max_width).to(device)  # no dropout during the inference process

    encoder.load_state_dict(torch.load(args.encoder, map_location=device))
    print(f"inference.py: pretrained encoder model loaded from {args.encoder}")
    decoder.load_state_dict(torch.load(args.decoder, map_location=device))
    print(f"inference.py: pretrained dncoder model loaded from {args.decoder}")

    encoder.eval()
    decoder.eval()

    encoder_output = encoder(image)    

    max_length = 20
    decoder_input = torch.zeros(1).long().to(device)
    decoder_hidden = decoder.initHidden(1).to(device)

    words, prob = seq2seq(encoder_output, decoder, decoder_input, decoder_hidden, max_length)
    print(f"prediction: {words} with accmulated probility: {prob}")
    return words
