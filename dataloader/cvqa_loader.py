
import sys
sys.path.insert(0, '../')
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
from util import tokenize, transform_bb, load_file, pkload, group, get_qsn_type
from tools.object_align import align
import os.path as osp
import h5py
import random as rd
import numpy as np

import pickle as pkl
import ipdb

class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        features,
        qmax_words=20,
        amax_words=5,
        tokenizer=None,
        a2id=None,
        max_feats=20,
        mc=0,
        bnum=10,
        cl_loss=0
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features: dictionary mapping video_id to torch tensor of features
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param tokenizer: tokenizer
        :param a2id: answer to index mapping
        :param max_feats: maximum frames to sample from a video
        """
        
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.dset = self.csv_path.split('/')[-2]
        
        self.video_feature_path = features
        self.bbox_num = bnum
        self.use_frame = True
        self.use_mot =  False
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.a2id = a2id
        self.tokenizer = tokenizer
        self.max_feats = max_feats
        self.mc = mc
        self.lvq = cl_loss 
        self.mode = osp.basename(csv_path).split('.')[0] #train, val or test
        
        # self.all_answers = set(self.data['answer'])
        
        if self.mode not in ['val', 'test']:
            self.all_answers = set(self.data['answer'])      #정답들만, 후보들은 X / self.data['answer'] : 34132 -> set(self.data['answer']) : 19913
            self.all_questions = set(self.data['question'])  #self.data['question'] : 34132 -> set(self.data['question']) : 31173
            self.ans_group, self.qsn_group = group(self.data, gt=False)   # question type에 따라서 question과 해당 question에 대한 answer 분류 

        if self.dset == 'star':
            self.vid_clips = load_file(osp.dirname(csv_path)+f'/clips_{self.mode}.json')
        
        if self.dset == 'causalvid':
            #import ipdb
            #ipdb.set_trace()
            data_dir = osp.dirname(csv_path)
            self.map_dir = load_file(osp.join(data_dir, 'map_dir_caul.json'))
            ### add ###
            
            self.vids = pkload(osp.join(data_dir, f'split/{self.mode}.pkl'))
            
            vf_info = pkload(osp.join(data_dir, 'idx2vid.pkl'))
            self.vf_info = dict()
            for idx, vid in enumerate(vf_info):
                if vid in self.vids:
                    self.vf_info[vid] = idx
                    
            ### appearance feature
            app_file = osp.join(self.video_feature_path, 'appearance_feat.h5')
            print('Load {}...'.format(app_file))
            self.frame_feats = {}
            with h5py.File(app_file, 'r') as fp:
                feats = fp['resnet_features']  # feats[0].shape = (8 ,16, 2048) / (26900,8,16,2048)
                for vid, idx in self.vf_info.items():
                    self.frame_feats[vid] = feats[idx][...]
            
            ### motion feature
            mot_file = osp.join(self.video_feature_path, 'motion_feat.h5')
            self.mot_feats = {}
            with h5py.File(mot_file, 'r') as fp:
                feats = fp['resnet_features']   # feats  = (26900,8,2048)
                for vid, idx in self.vf_info.items():
                    self.mot_feats[vid] = feats[idx][...]
                    
            # self.txt_obj = {}
            # with h5py.File(osp.join(data_dir, 'ROI_text.h5'), 'r') as f:
            #     keys = [item for item in vids if item in f.keys()]
            #     for key in keys:
            #         tmp = {}
            #         labels = f[key].keys()
            #         for label in labels:
            #             new_label = label.replace('_', '')
            #             tmp[new_label] = f[key][label][...]
            #         self.txt_obj[key] = tmp
 
        
    def __len__(self):
        return len(self.data)
    
    def get_video_feature(self, video_name, width=320, height=240):
        """
        :param video_name
        :param width
        :param height
        :return:
        """
        cnum = 8
        cids = list(range(cnum))
        pick_ids = cids
        
        if self.use_frame:
            temp_feat = self.frame_feats[video_name][pick_ids] #[:, pick_id,:] #.reshape(clip_num*fnum, -1)[pick_ids,:] #[:,pick_ids,:]
            app_feat = torch.from_numpy(temp_feat).type(torch.float32)
            
            temp_mot = self.mot_feats[video_name][pick_ids]
            mot_feat = torch.from_numpy(temp_mot).type(torch.float32)
        else:
            app_feat = torch.tensor([0])
  
        return mot_feat, app_feat
        
    def __getitem__(self, index):
        
        #import ipdb
        #ipdb.set_trace()
        
        cur_sample = self.data.loc[index]
        vid_id = cur_sample["video_id"]
        vid_id = str(vid_id)
        qid =  str(cur_sample['qid'])
      
        width, height = cur_sample['width'], cur_sample['height']
      
        video_o, video_f = self.get_video_feature(vid_id, width, height)
        
        vid_duration = video_o.shape[0]

        # video_o, video_f = torch.tensor([0]), torch.tensor([0])
        # vid_duration = 0
        
        question_txt = cur_sample['question']
        
        #print(len(question_txt))
        if self.mc == 0:
            #open-ended QA
            question_embd = torch.tensor(
                self.bert_tokenizer.encode(
                    question_txt,
                    add_special_tokens=True,
                    padding="longest",
                    max_length=self.qmax_words,
                    truncation=True,
                ),
                dtype=torch.long
            )
            seq_len = torch.tensor([len(question_embd)], dtype=torch.long)
        else:
            question_embd = torch.tensor([0], dtype=torch.long)
        
        qtype, ans_token_ids, answer_len = 0, 0, 0
        max_seg_num = self.amax_words
        seg_feats = torch.zeros(self.mc, max_seg_num, 2048)
        seg_num = torch.LongTensor(self.mc)

        qsn_id , qsn_token_ids, qsn_seq_len = 0, 0, 0
        qtype = 'null' if 'type' not in cur_sample  else cur_sample['type'] 
        if self.lvq and self.mode not in ['val','test']:  # train 
            
            qtype = get_qsn_type(question_txt, qtype) 
            neg_num = 5
            if qtype not in self.qsn_group or len(self.qsn_group[qtype]) < neg_num-1:
                valid_qsncans = self.all_questions #self.qsn_group[self.mtype]
            else:
                valid_qsncans = self.qsn_group[qtype]

            cand_qsn = valid_qsncans - set(question_txt)
            qchoices = rd.sample(list(cand_qsn), neg_num-1) # 자기 자신 question 제외하고 4개 무작위 추출
            qchoices.append(question_txt)
            rd.shuffle(qchoices)
            qsn_id = qchoices.index(question_txt) # neg 4개 + 자기 자신 question 1 중에서 진짜 질문의 index 
            qsn_token_ids, qsn_tokens = tokenize(
                    qchoices,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.qmax_words,
                    dynamic_padding=False,
                    truncation=True
                )
            qsn_seq_len = torch.tensor([len(qsn) for qsn in qsn_token_ids], dtype=torch.long) # 총 질문 5개에 대한 length
        
        question_id = vid_id +'_'+str(cur_sample["qid"])
        if self.mc:
            if self.dset == 'causalvid':
                qtype = str(cur_sample["type"])   
                question_id = vid_id +'_'+qtype      
            
            if self.dset=='webvid': # delete
                ans = cur_sample["answer"]
                cand_answers = self.all_answers
                choices = rd.sample(cand_answers, self.mc-1)
                choices.append(ans)
                rd.shuffle(choices)
                answer_id = choices.index(ans)
                answer_txts = choices

            else:
                ans = cur_sample['answer']
                choices = [str(cur_sample["a" + str(i)]) for i in range(self.mc)] # a0,1,2,3,4
                answer_id = choices.index(ans) if ans in choices else -1          # 정답 인덱스 
            
                answer_txts = [question_txt+f' {self.tokenizer.sep_token} '+opt for opt in choices]
                
                # print(answer_txts)
                # if self.dset == 'causalvid':
                    # if vid_id in self.txt_obj:
                    #     labels = set(self.txt_obj[vid_id])
                    #     for ai, qa in enumerate(answer_txts):
                    #         labs = list(labels.intersection(set(qa.split())))
                    #         cnt = 0
                    #         for i, lab in enumerate(labs):
                    #             seg_feats[ai][i] = torch.from_numpy(self.txt_obj[vid_id][lab])
                    #             cnt += 1
                    #         seg_num[ai] = cnt
        
            try:
                ans_token_ids, answer_tokens = tokenize(
                    answer_txts,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.amax_words,
                    dynamic_padding=False,
                    truncation=True
                )

                # if self.dset == 'causalvid' and vid_id in self.txt_obj:
                #     for mcid, opt_tks in enumerate(answer_tks):
                #         for idx, tk in enumerate(opt_tks):
                #             if idx > 1 and tk.isdigit():
                #                 label = str(opt_tks[idx-1][1:])+tk #label should be like 'person1'
                #                 if label in self.txt_obj[vid_id]:
                #                     seg_feats[mcid][idx] = torch.from_numpy(self.txt_obj[vid_id][label])
            except:
                print('Fail to tokenize: '+answer_txts)
            seq_len = torch.tensor([len(ans) for ans in ans_token_ids], dtype=torch.long)
        else:
            answer_txts = cur_sample["answer"]
            answer_id = self.a2id.get(answer_txts, -1)  
        #choices = (choices)   
        return {
            "video_id": vid_id,
            "video_o": video_o,
            "video_f": video_f,
            "video_len": vid_duration,
            "question": question_embd,
            "question_txt": question_txt,
            "type": qtype,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "answer": ans_token_ids,
            "answer_candidate" : choices,
            "answer_only" : ans,
            "seq_len": seq_len,
            "question_id": question_id,
            "seg_feats": seg_feats,
            "seg_num": seg_num,
            "qsn_id": qsn_id,
            "qsn_token_ids": qsn_token_ids,
            "qsn_seq_len": qsn_seq_len
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, features, a2id, tokenizer, test_mode):
    
    if test_mode:
        test_dataset = VideoQADataset(
            csv_path=args.test_csv_path, ##
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            bnum =args.bnum,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        train_loader, val_loader = None, None
    else:
        train_dataset = VideoQADataset(
        csv_path=args.train_csv_path,
        features=features,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        tokenizer=tokenizer,
        a2id=a2id,
        max_feats=args.max_feats,
        mc=args.mc,
        bnum =args.bnum,
        cl_loss=args.cl_loss
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=True,
            drop_last=True,
            collate_fn=videoqa_collate_fn,
        )
        if args.dataset.split('/')[0] in ['tgifqa','tgifqa2']:
            args.val_csv_path = args.test_csv_path
        val_dataset = VideoQADataset(
            csv_path=args.val_csv_path, ##
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            bnum =args.bnum,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn,
        )
        test_dataset = VideoQADataset(
            csv_path=args.test_csv_path, ##
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            tokenizer=tokenizer,
            a2id=a2id,
            max_feats=args.max_feats,
            mc=args.mc,
            bnum =args.bnum,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        #test_loader = None

    return (train_loader, val_loader, test_loader)
