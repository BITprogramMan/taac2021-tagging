import os
import json
import pandas as pd
import numpy as np
import shutil
import codecs

def main():
    
    with open('step1/results1/tagging_1.json') as f:
        part1=json.load(f)
    with open('step1/results2/tagging_2.json') as f:
        part2=json.load(f)
    with open('step1/results3/tagging_3.json') as f:
        part3=json.load(f)
    with open('step1/results4/tagging_4.json') as f:
        part4=json.load(f)
    with open('step1/results5/tagging_5.json') as f:
        part5=json.load(f)
    all_label=[]
    with open('../dataset/label_id.txt','r') as f:
        for line in f.readlines():
            label,_ = line.split('\t')
            all_label.append(label)
    all_label_set = set(all_label)
    for file in part1.keys():
        label_set = set(part1[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part1[file]['result'][0]['labels'] +=  res_label
        part1[file]['result'][0]['scores'] += ['0']*32
        assert len(res_label)==32
        assert len(part1[file]['result'][0]['labels'])==82
    for file in part2.keys():
        label_set = set(part2[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part2[file]['result'][0]['labels'] +=  res_label
        part2[file]['result'][0]['scores'] += ['0']*32
        assert len(res_label)==32
        assert len(part2[file]['result'][0]['labels'])==82
    for file in part3.keys():
        label_set = set(part3[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part3[file]['result'][0]['labels'] +=  res_label
        part3[file]['result'][0]['scores'] += ['0']*32
        assert len(res_label)==32
        assert len(part3[file]['result'][0]['labels'])==82
    for file in part4.keys():
        label_set = set(part4[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part4[file]['result'][0]['labels'] +=  res_label
        part4[file]['result'][0]['scores'] += ['0']*32
        assert len(res_label)==32
        assert len(part4[file]['result'][0]['labels'])==82
    for file in part5.keys():
        label_set = set(part5[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part5[file]['result'][0]['labels'] +=  res_label
        part5[file]['result'][0]['scores'] += ['0']*32
        assert len(res_label)==32
        assert len(part5[file]['result'][0]['labels'])==82
    w1 = 0.2
    w2 = 0.2
    w3 = 0.2
    w4 = 0.2
    w5 = 0.2
    av_result = {}
    for file in part1.keys():
        av_result[file] = {}
        av_result[file]['result'] = []
        av_result[file]['result'].append({})
        av_result[file]['result'][0]['labels'] = []
        av_result[file]['result'][0]['scores'] = []
        for index, label in enumerate(part1[file]['result'][0]['labels']):
            av_result[file]['result'][0]['labels'].append(label)
            index1 = part1[file]['result'][0]['labels'].index(label)
            index2 = part2[file]['result'][0]['labels'].index(label)
            index3 = part3[file]['result'][0]['labels'].index(label)
            index4 = part4[file]['result'][0]['labels'].index(label)
            index5 = part5[file]['result'][0]['labels'].index(label)
            score =  (w1*eval(part1[file]['result'][0]['scores'][index1]) + 
                     w2*eval(part2[file]['result'][0]['scores'][index2]) + 
                     w3*eval(part3[file]['result'][0]['scores'][index3]) +
                     w4*eval(part4[file]['result'][0]['scores'][index4]) +
                     w5*eval(part5[file]['result'][0]['scores'][index5]) )
            av_result[file]['result'][0]['scores'].append("%.4f" % score)
    for file in av_result.keys():
        score = av_result[file]['result'][0]['scores']
        label = av_result[file]['result'][0]['labels']
        sorted_score = sorted(enumerate(score), key=lambda x: x[1],reverse=True)
        idx = [i[0] for i in sorted_score]
        top20_index=idx[:20]
        av_result[file]['result'][0]['scores'] =[score[i] for i in top20_index]
        av_result[file]['result'][0]['labels'] =[label[i] for i in top20_index]
    with open('step1/results1/train_av_result.json', 'w', encoding="utf-8") as f:
        json.dump(av_result, f, ensure_ascii=False, indent=4)

    train_label=pd.read_csv('../dataset/tagging/GroundTruth/tagging_info.txt',header=None,names=['video','label'],sep='\t')

    predict_df=pd.read_json('step1/results1/train_av_result.json', orient='index')
    predict_df['label']=predict_df.apply(lambda row:row['result'][0]['labels'],axis=1)
    predict_df=predict_df.reset_index()
    predict_df.rename(columns={'index':'video'},inplace=True)
    predict_df['label_score']=predict_df.apply(lambda row:row['result'][0]['scores'],axis=1)
    predict_df.drop('result',axis=1,inplace=True)
    predict_df['soft_label'] = predict_df.apply(lambda row:[0.]*82,axis=1)

    class Preprocess_label_sparse_to_dense:

        def __init__(self,
                     index_dict,
                     sep_token=',',
                     is_training=False):
            self.index_to_tag,self.tag_to_index = extract_dict(index_dict)
            self.sep_token = sep_token
            self.is_training = is_training
            self.max_index = 0
            for index in self.index_to_tag:
                self.max_index = max(index, self.max_index)
            self.seq_size = self.max_index + 1
            self.label_num = self.seq_size

        def __call__(self, index_str):
            dense_array = np.zeros(self.seq_size)
            label_lst = index_str.split(self.sep_token)
            for label in label_lst:
                if label in self.tag_to_index:
                    index = self.tag_to_index[label]
                    dense_array[index] = 1.0
            return dense_array.astype('float32')
    def extract_dict(dict_file):
        index_to_tag = {}
        tag_to_index = {}
        for i, line in enumerate(codecs.open(dict_file, 'r', encoding='utf-8')):
            line = line.strip()
            if '\t' in line:
                index, tag = line.split('\t')[:2]
            elif ' ' in line:
                index, tag = i, line.rsplit(' ', 1)[0]
            else:
                index, tag = i, line

            try:
                index = int(index)
            except:
                index, tag = int(tag), index

            index_to_tag[index] = tag
            tag_to_index[tag] = index
        return index_to_tag, tag_to_index
    preprocess_instance=Preprocess_label_sparse_to_dense('../dataset/label_id.txt')
    tag_to_index = preprocess_instance.tag_to_index
    predict_df['label_index']=predict_df.apply(lambda row:[tag_to_index[tag] for tag in row['label']],axis=1)
    for i in range(predict_df.shape[0]):
        temp_list = np.array(predict_df.iloc[i]['soft_label'],dtype=np.float)
        temp_list[predict_df.iloc[i]['label_index']] = predict_df.iloc[i]['label_score']
        update_row=dict(predict_df.iloc[i])
        update_row['soft_label']= temp_list.tolist()
        predict_df.iloc[i] = pd.Series(update_row)
    predict_dfV1 = predict_df[['video','soft_label']]

    predict_dfV1.to_csv('./train_soft_label.csv',index=None)

if __name__=='__main__':
    
    main()
    