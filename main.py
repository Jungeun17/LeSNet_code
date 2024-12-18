import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import os.path as osp
import logging

from transformers import get_cosine_schedule_with_warmup
from args import get_args
from LeSNet.model.LeSNet import LeSNet
from loss import LogSoftmax
from util import compute_a2v, load_model_by_key, save_to
from dataloader.cvqa_loader import get_videoqa_loaders
from train.train_covgt import train, eval



def main(args):
    if not (os.path.isdir(args.save_dir)):
        os.mkdir(os.path.join(args.save_dir))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "w+")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    logging.info(args)


    if args.lan == 'BERT':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.lan == 'RoBERTa':
        from transformers import RobertaTokenizerFast,RobertaTokenizer
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    
    a2id, id2a, a2v = None, None, None
    if not args.mc:
        a2id, id2a, a2v = compute_a2v(
            vocab_path=args.vocab_path,
            bert_tokenizer=tokenizer,
            amax_words=args.amax_words,
        )
        logging.info(f"Length of Answer Vocabulary: {len(a2id)}")

    # Model
    model = LeSNet(
        tokenizer = tokenizer,
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.max_feats,
        Q=args.qmax_words,
        vocab_size = tokenizer.vocab_size,
        baseline=args.baseline,
        bnum=args.bnum,
        lan=args.lan
    )
    model.cuda()
    logging.info("Using {} GPUs".format(torch.cuda.device_count()))

    # Load pretrain path
    model = nn.DataParallel(model)
    
    if args.pretrain_path != "":
        #model.load_state_dict(torch.load(args.pretrain_path))
        model.load_state_dict(load_model_by_key(model, args.pretrain_path))
        logging.info(f"Loaded checkpoint {args.pretrain_path}")
    logging.info(
        f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_videoqa_loaders(args, args.features_path, a2id, tokenizer, test_mode = args.test)

    if args.test:
        logging.info("number of test instances: {}".format(len(test_loader.dataset)))
    else:
        logging.info("number of train instances: {}".format(len(train_loader.dataset)))
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # criterion = MultipleChoiceLoss()
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(
        params_for_optimization, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion.cuda()

    # Training
    if not args.test:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0, len(train_loader) * args.epochs
        )
        logging.info(
            f"Set cosine schedule with {len(train_loader) * args.epochs} iterations"
        )
        if args.pretrain_path != "":
            val_acc, results = eval(model, val_loader, a2v, args, test=False, tokenizer=tokenizer)  
            save_path = osp.join(args.save_dir, 'val-res0.json')
            save_to (save_path, results)
        best_val_acc = 0 if args.pretrain_path == "" else val_acc

        best_test_acc = 0 if args.pretrain_path == "" else test_acc
        best_epoch = 0
        for epoch in range(args.epochs):
            train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, tokenizer)
            val_acc, results = eval(model, val_loader, a2v, args, test=False, tokenizer=tokenizer)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
                save_path = osp.join(args.save_dir, 'val-res.json')
                save_to (save_path, results)

            test_acc, test_results = eval(model, test_loader, a2v, args, test=True, tokenizer=tokenizer)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
                save_path = osp.join(args.save_dir, 'test-res.json')
                save_to (save_path, test_results)

            if args.dataset == 'webvid': 
                ep_file = os.path.join(args.save_dir, f"e{epoch}.pth")
                torch.save(model.state_dict(), ep_file)
                logging.info('Save to '+ep_file)
        logging.info(f"Best val model at epoch {best_epoch + 1}")
    else:   
    # Evaluate on test set
        test_acc, results = eval(model, test_loader, a2v, args, test=True, tokenizer=tokenizer)
        save_path = osp.join(args.save_dir, 'test-res.json')
        save_to(save_path, results)


if __name__ == "__main__":
    # set random seeds
    args = get_args()
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
        
    main(args)
