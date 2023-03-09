import os
import glob


def make_file(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def read_txt(root):
    data = []
    with open(root, 'r') as f:
        for file in f.readlines():
            file = file.strip()
            data.append(file)
    return data


def write_txt(data, save_path):
    with open(save_path, 'w') as f:
        for i in range(len(data)):
            f.write(data[i])
            f.write('\n')
        f.close()


def make_data_index(load_root, file_index, save_path):
    data_list = []
    for file_name in file_index:
        data_path = os.path.join(load_root, file_name, "color", "*.png")
        data = glob.glob(data_path)
        for i, path in enumerate(data):
            data_list.append(path)
    write_txt(data_list, save_path)


if __name__ == "__main__":
    num = [14, 28]
    id = "first" # first second
    root = "/home/wanglei"
    for i in num:
        list_id = "list_index_{}".format(i)
        save_id = "list_{}".format(i)
        make_file(r"{}/data/{}/{}".format(root, id, save_id))
        load_root = r"{}/data/{}".format(root, id)
        train_save_path = os.path.join(load_root, save_id, "train.txt")
        val_save_path = os.path.join(load_root, save_id, "val.txt")
        test_save_path = os.path.join(load_root, save_id, "test.txt")

        train_file_index = os.path.join(load_root, list_id, "train.txt")
        val_file_index = os.path.join(load_root, list_id, "val.txt")
        test_file_index = os.path.join(load_root, list_id, "test.txt")

        train_data_path = read_txt(train_file_index)
        val_data_path = read_txt(val_file_index)
        test_data_path = read_txt(test_file_index)

        make_data_index(load_root, train_data_path, train_save_path)
        make_data_index(load_root, val_data_path, val_save_path)
        make_data_index(load_root, test_data_path, test_save_path)




