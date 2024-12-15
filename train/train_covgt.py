import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens, tokenize
import os.path as osp
import json
#from fvcore.nn import FlopCountAnalysis

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import itertools
import ipdb
import random as rd

from tqdm import tqdm

def eval(model, data_loader, a2v, args, test=False, tokenizer="RoBERTa"):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = {}
        for i, batch in enumerate(data_loader):
            answer_id, answer, video_o, video_f, question, question_id, seg_feats, seg_num = (
                batch["answer_id"],
                batch["answer"].cuda(),
                batch["video_o"].cuda(),
                batch["video_f"].cuda(),
                batch["question"].cuda(),
                batch['question_id'],
                batch['seg_feats'].cuda(),
                batch['seg_num']
            )
           
            video_len = batch["video_len"]
            seq_len = batch["seq_len"]
           
            question_mask = (question!=tokenizer.pad_token_id).float() #RobBERETa
            answer_mask = (answer!=tokenizer.pad_token_id).float() #RobBERETa

            video_mask = get_mask(video_len, video_o.size(1)).cuda()
            count += answer_id.size(0)
            video = (video_o, video_f)
            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    seq_len = seq_len
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
            else:
                #############Model FLOPs##########
                # inputs = (video, question, None, answer.cuda(), seq_len, video_mask, answer_mask)
                # flops = FlopCountAnalysis(model, inputs)
                # print('Model FLOPs:', flops.total()/1000000) #use batch_size 1
                # break
                ###################################
                fusion_proj, answer_proj = model(
                    video,
                    question,
                    text_mask=answer_mask,
                    video_mask=video_mask,
                    answer=answer,
                    seq_len = seq_len,
                    seg_feats = seg_feats,
                    seg_num = seg_num
                )
                # predicts = fusion_proj.squeeze() 
                
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze() # (64, 5, 512) (64, 512, 1)
                
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}

    step = "val" if not test else "test"
    
    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break

    return metrics["acc"] / count, results


def calculate_bleu_score(reference, candidate):
    """
    reference: 참조 문장
    candidate: 비교할 후보 문장
    """
    reference_tokenized = word_tokenize(reference.lower())
    candidate_tokenized = word_tokenize(candidate.lower())
    
    score = sentence_bleu([reference_tokenized], candidate_tokenized, weights=(1, 0, 0, 0))

    return score


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss, running_cl_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter()
    )
  
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        answer_id, answer, video_o, video_f, question, seg_feats, seg_num, qsn_id, qsn_token_ids, qsn_seq_len = (
            batch["answer_id"],         # (64)    
            batch["answer"],            # (64,5,38) ans_token_ids
            batch["video_o"].cuda(),    # (64,8,4,5,2053)
            batch["video_f"].cuda(),    # (64,8,4,2048)
            batch["question"].cuda(),   # (64,1) [0] ...
            batch['seg_feats'].cuda(),  # (64,5,38,2048)
            batch['seg_num'],           # (64,5)
            batch['qsn_id'],            # (64)
            batch['qsn_token_ids'],     # (64,5,30)
            batch['qsn_seq_len']        # (64,5)
        )
        
        video_len = batch["video_len"]
        
        question_txt = batch["question_txt"]          # (64,) 'Who is the one filming the video'
        answer_candidate = batch["answer_candidate"]  # (5, 64) answer candidate
        answer_txt = batch["answer_txt"]              # (5, 64) question + answer candidate
        answer_only = batch["answer_only"]            # (64 )


        final_answer = []
        new_answer_id_list = []
        for id, answer_1 in enumerate(answer_only):  # 64개의 answer_only에 대해 반복
            scores = []
            new_answer = []

            for idx, candidates in enumerate(answer_candidate):  # 각 answer_only마다 5개의 후보를 가진 list
                for j, candidate in enumerate(candidates):
                    if candidate != answer_1 :
                        score = calculate_bleu_score(answer_1, candidate)
                        scores.append((score, j, idx, answer_1, candidate))  # 점수와 (answer_only의 인덱스, candidate의 인덱스) 저장
            
            top_indices = sorted(scores, key=lambda x: x[0], reverse=True)[:4]

            answer_score = (1000, id, answer_id[id].item(), answer_1, answer_1)

            top_indices.append(answer_score)

            rd.shuffle(top_indices)
            
            #ipdb.set_trace()
            
            for idx, score in enumerate(top_indices) :
                new_answer.append(question_txt[id]+f' {tokenizer.sep_token} '+ score[4])

                if score[0] == 1000:
                    new_answer_id_list.append(idx)

            ans_token_ids, answer_tokens = tokenize(
                    new_answer,
                    tokenizer,
                    add_special_tokens=True,
                    max_length=args.amax_words,
                    dynamic_padding=False,
                    truncation=True
                )
          
            final_answer.append(ans_token_ids.tolist())

        final_answer = torch.tensor(final_answer, dtype=torch.long)
        new_answer_id_list = torch.tensor(new_answer_id_list, dtype=torch.long)
        
        question_mask = (question != tokenizer.pad_token_id).float().cuda() #RobBERETa / tokenizer.pad_token_id : 1
        answer_mask = (answer!=tokenizer.pad_token_id).float().cuda() #RobBERETa

        final_answer_mask = (final_answer!=tokenizer.pad_token_id).float().cuda() #RobBERETa ##addn
        video_mask = (
            get_mask(video_len, video_o.size(1)).cuda() if args.max_feats > 0 else None
        )
       
        
        video = (video_o, video_f)
        N = answer_id.size(0)  #batch
        seq_len = batch["seq_len"]
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video,
                question,
                text_mask=question_mask,
                video_mask=video_mask,
                seq_len = seq_len
            )
        else:
            fusion_proj, answer_proj = model(
                video,
                question, 
                text_mask=answer_mask,
                video_mask=video_mask,
                answer=answer.cuda(),  
                seq_len = seq_len,
                seg_feats = seg_feats,
                seg_num = seg_num
            )
                    
            fusion_proj = fusion_proj.unsqueeze(2) # 64,512 -> 64,512,1 / answer_proj : 65,5,512
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze() #64,5

        
        vqa_loss = criterion(predicts, answer_id.cuda())
        predicted = torch.max(predicts, dim=1).indices.cpu() 
        running_acc.update((predicted == answer_id).sum().item() / N, N)
        
        ### add ###
        vt_proj, txt_proj = model(
            video,
            question,
            text_mask= final_answer_mask,
            video_mask=video_mask,
            answer=final_answer.cuda(),
            seq_len = seq_len,
            seg_feats = seg_feats,
            seg_num = seg_num
        )
        vt_proj = vt_proj.unsqueeze(2)
        cl_predicts = torch.bmm(txt_proj, vt_proj).squeeze()
        cl_loss = criterion(cl_predicts, new_answer_id_list.cuda())
            # cl_predicted = torch.max(cl_predicts, dim=1).indices.cpu()
            # running_acc.update((predicted == answer_id).sum().item() / N, N)

        if args.mlm_prob:
            max_seq_len = args.qmax_words
            if args.mc > 0:
                tmp_id = [aid+(args.mc*i) for i, aid in enumerate(answer_id)]
                inputs = answer.view(N*args.mc, -1)[tmp_id,:]
                # question_mask = (inputs>0).float()
                question_mask = (inputs!=1).float()
                max_seq_len = args.amax_words
            else:
                inputs = batch["question"]
            
            inputs, labels = mask_tokens(inputs, tokenizer, mlm_probability=args.mlm_prob)
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                max_seq_len=max_seq_len,
                mode="mlm",
            )
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        ####
        loss = vqa_loss + 0.3*cl_loss

        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()
        
        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)

        running_cl_loss.update(cl_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Vqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, MLM loss: {running_mlm_loss.avg:.4f}, Cl Loss: {running_cl_loss.avg:.4f}"
                )
            else : #args.cl_loss:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Progress: {float(i + 1) / len(train_loader):.4f}, Vqa loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}, Cl Loss: {running_cl_loss.avg:.4f}"
                )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()
            running_cl_loss.reset()
