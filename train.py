import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import time


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_training_time = time.time()
    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.label
            target.sub_(1)  # index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()) == target).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

        # --- TÍNH THỜI GIAN KẾT THÚC EPOCH ---
        epoch_duration = time.time() - epoch_start_time
        print(f'\nEnd of Epoch {epoch} | Time: {epoch_duration:.2f}s')

    # --- TÍNH TỔNG KẾT THỜI GIAN ---
    total_training_time = time.time() - start_training_time
    avg_time_per_epoch = total_training_time / args.epochs

    print('\n' + '='*40)
    print('TRAINING STATISTICS:')
    print(f'Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)')
    print(f'Average Time per Epoch: {avg_time_per_epoch:.2f} seconds')
    print('='*40 + '\n')


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        target.sub_(1)  # index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, reduction='sum')

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()) == target).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_field, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    if cuda_flag:
        x = x.cuda()
    print(x)
    with torch.no_grad():
        output = model(x)
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.item()+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
