#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import model
import train
import mydatasets
import re

# --- 1. KIỂM TRA THƯ VIỆN ---
try:
    from underthesea import word_tokenize
except ImportError:
    print("Loi: Chua cai underthesea. Hay chay: pip install underthesea")
    exit()

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=25, help='number of epochs for train [default: 25]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# --- 2. HÀM TOKENIZER ---
def vietnamese_tokenizer(text):
    text = re.sub(r"[^\w\s(),!?]", " ", text)
    return word_tokenize(text, format="text").split()


# --- 3. HÀM LOAD DATASET ---
def load_data_csv(text_field, label_field, **kargs):
    print("Dang doc file CSV...")
    try:
        # path='.' vì file data.csv cùng thư mục
        # Thay đổi path='data' để đọc được file trong thư mục data/
        train_data, dev_data, test_data = mydatasets.MyCSVDataset.splits(
                                            text_field, label_field, 
                                            path='data', 
                                            train='data.csv')
    except AttributeError:
        print("LOI: File mydatasets.py chua co class MyCSVDataset.")
        exit()

    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data)
    print(f"Da tim thay {len(label_field.vocab)} loai nhan: {label_field.vocab.freqs.keys()}")

    # Nạp Vector (Nếu có)
    word2vec_path = 'cc.vi.300.vec' 
    if os.path.exists(word2vec_path):
        print(f"Dang nap vector {word2vec_path}...")
        try:
            import gensim
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
            vocab_size = len(text_field.vocab)
            embed_dim = 300
            embedding_weights = torch.zeros(vocab_size, embed_dim)
            count = 0
            for word, idx in text_field.vocab.stoi.items():
                if word in word2vec_model:
                    embedding_weights[idx] = torch.from_numpy(word2vec_model[word])
                    count += 1
                else:
                    embedding_weights[idx] = torch.randn(embed_dim)
            text_field.vocab.vectors = embedding_weights
            print(f"--> Da nap {count} tu vung.")
        except Exception as e:
            print(f"Loi khi nap vector: {e}")
    else:
        print(f"Khong tim thay '{word2vec_path}', su dung vector ngau nhien.")

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                (train_data, dev_data, test_data), 
                                batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
                                sort_key=lambda x: len(x.text),
                                device=kargs.get('device', 'cpu'),
                                repeat=kargs.get('repeat', False))
    return train_iter, dev_iter, test_iter

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n' + '='*30)
    print(f'MODEL SUMMARY:')
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    print('='*30 + '\n')
    return total_params, trainable_params

# --- 4. CHƯƠNG TRÌNH CHÍNH 
if __name__ == '__main__':
    print("\nLoading data...")
    
    # Khai báo Field (batch_first=True)
    text_field = data.Field(lower=True, tokenize=vietnamese_tokenizer, batch_first=True)
    label_field = data.Field(sequential=False)

    # Load dữ liệu (Cần thiết để xây dựng lại Từ Điển Vocab)
    try:
        train_iter, dev_iter, test_iter = load_data_csv(text_field, label_field, device='cpu', repeat=False)
    except NameError:
        print("LOI: Ham load_data_csv loi.")
        exit()

    args.embed_num = len(text_field.vocab)
    args.class_num = len(label_field.vocab) 
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Khởi tạo Model
    cnn = model.CNN_Text(args)
    if args.cuda: cnn = cnn.cuda()
    count_parameters(cnn)

    # Load Snapshot để dự đoán
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        # map_location='cpu' để tránh lỗi nếu máy không có GPU
        cnn.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage))

    if args.cuda:
        if args.device != -1:
            torch.cuda.set_device(args.device)
        cnn = cnn.cuda()
        
    # --- XỬ LÝ DỰ ĐOÁN ---
    if args.predict is not None:
        # Gọi hàm predict từ train.py (đã được sửa ở bước trước)
        label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
        
        print(f'[Text]  {args.predict}')
        print(f'[Label] {label.upper()}')
        
        
    elif args.test:
        train.eval(test_iter, cnn, args) 
    else:
        # Training
        try:
            train.train(train_iter, dev_iter, cnn, args)
        except KeyboardInterrupt:
            print('Exiting from training early')