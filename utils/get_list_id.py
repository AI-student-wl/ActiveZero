import os
import glob
import random

def make_file(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def write_txt(data, save_path):
    with open(save_path, 'w') as f:
        for i in range(len(data)):
            f.write(data[i])
            f.write('\n')
        f.close()


id = "first"  # first, second
root = "/home/wanglei"
file_name = glob.glob("{}/data/{}/part*".format(root, id))
name_list = []
for i in range(len(file_name)):
    tt = file_name[i].split("/")[-1]
    name_list.append(tt)
random.shuffle(name_list)



num = [int(len(name_list)*0.5), len(name_list)]
for i in range(len(num)):
    train_index = []
    val_index = []
    test_index = []
    list_index_path = "{}/data/{}/list_index_{}".format(root, id, num[i])
    make_file(list_index_path)
    for j in range(num[i]):
        if j <= int(num[i]*0.80):
            train_index.append(name_list[j])
        else:
            if j % 2 == 0:
                val_index.append(name_list[j])
            else:
                test_index.append(name_list[j])
    write_txt(train_index, "{}/data/{}/list_index_{}/train.txt".format(root, id, num[i]))
    write_txt(val_index, "{}/data/{}/list_index_{}/val.txt".format(root, id, num[i]))
    write_txt(test_index, "{}/data/{}/list_index_{}/test.txt".format(root, id, num[i]))



