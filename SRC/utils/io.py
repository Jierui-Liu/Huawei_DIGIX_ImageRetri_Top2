import pickle

#文件结构：
#dict["fname"]=list([....])
#dict["data"]=np.array([])


def dict_to_file(file_path,dict_data):
    f = open(file_path, "wb")
    pickle.dump(dict_data, f)
    f.close()


def file_to_dict(file_path):
    f = open(file_path, "rb")
    dict_data = pickle.load(f)
    f.close()
    return dict_data



def file_to_fname_data(file_path):
    f = open(file_path, "rb")
    dict_data = pickle.load(f)
    f.close()
    return dict_data["fname"],dict_data["data"]



if __name__ == '__main__':
    pass



