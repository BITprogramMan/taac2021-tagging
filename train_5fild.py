import copy
import random

def train_5fild():
    with open('../dataset/tagging/GroundTruth/datafile/train.txt') as f:
        train =f.readlines()
    with open('../dataset/tagging/GroundTruth/datafile/val.txt') as f:
        val =f.readlines()  
    train.extend(val)
    train_file_list=copy.deepcopy(train)
    part1_file_list=[]
    part2_file_list=[]
    part3_file_list=[]
    part4_file_list=[]
    part5_file_list=[]
    train_index = random.sample(range(5000), 5000)
    part1_index = train_index[:1000]
    part2_index = train_index[1000:2000]
    part3_index = train_index[2000:3000]
    part4_index = train_index[3000:4000]
    part5_index = train_index[4000:5000]
    for index in part1_index:
        part1_file_list.extend(train_file_list[index*6:index*6+6])
    for index in part2_index:
        part2_file_list.extend(train_file_list[index*6:index*6+6])
    for index in part3_index:
        part3_file_list.extend(train_file_list[index*6:index*6+6])
    for index in part4_index:
        part4_file_list.extend(train_file_list[index*6:index*6+6])
    for index in part5_index:
        part5_file_list.extend(train_file_list[index*6:index*6+6])
    train1= part1_file_list+part2_file_list+part3_file_list+part4_file_list
    val1 = part5_file_list
    train2 = part1_file_list+part2_file_list+part3_file_list+part5_file_list
    val2 = part4_file_list
    train3 = part1_file_list+part2_file_list+part5_file_list+part4_file_list
    val3 = part3_file_list
    train4 = part1_file_list+part5_file_list+part3_file_list+part4_file_list
    val4 = part2_file_list
    train5 = part5_file_list+part2_file_list+part3_file_list+part4_file_list
    val5 = part1_file_list
    with open('../dataset/tagging/GroundTruth/datafile/train1.txt','w',encoding='utf-8') as f:
        f.writelines(train1)
    with open('../dataset/tagging/GroundTruth/datafile/train2.txt','w',encoding='utf-8') as f:
        f.writelines(train2)
    with open('../dataset/tagging/GroundTruth/datafile/train3.txt','w',encoding='utf-8') as f:
        f.writelines(train3)
    with open('../dataset/tagging/GroundTruth/datafile/train4.txt','w',encoding='utf-8') as f:
        f.writelines(train4)
    with open('../dataset/tagging/GroundTruth/datafile/train5.txt','w',encoding='utf-8') as f:
        f.writelines(train5)
    with open('../dataset/tagging/GroundTruth/datafile/val1.txt','w',encoding='utf-8') as f:
        f.writelines(val1)
    with open('../dataset/tagging/GroundTruth/datafile/val2.txt','w',encoding='utf-8') as f:
        f.writelines(val2)
    with open('../dataset/tagging/GroundTruth/datafile/val3.txt','w',encoding='utf-8') as f:
        f.writelines(val3)
    with open('../dataset/tagging/GroundTruth/datafile/val4.txt','w',encoding='utf-8') as f:
        f.writelines(val4)
    with open('../dataset/tagging/GroundTruth/datafile/val5.txt','w',encoding='utf-8') as f:
        f.writelines(val5)
        
if __name__=='__main__':
    train_5fild()