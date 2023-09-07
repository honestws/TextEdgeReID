import json

def dataparse(CFG):
    project_path = CFG.proj_path
    if CFG.dataset == 'CUHK-PEDES':
        data_path = project_path + 'data/CUHK-PEDES/CUHK-PEDES/'
        with open(data_path + 'reid_raw.json') as file:
            json_data = json.load(file)
        train_list, val_list, test_list = [], [], []
        for item in json_data:
            item['file_path'] = data_path + 'imgs/' + item['file_path']
            if item['split'] == 'train':
                train_list.append(item)
            elif item['split'] == 'val':
                val_list.append(item)
            elif item['split'] == 'test':
                test_list.append(item)
        return train_list, val_list, test_list

    elif CFG.dataset == 'ICFG-PDES':
        data_path = project_path + 'data/ICFG-PDES/ICFG-PEDES/'
        with open(data_path + 'ICFG-PEDES.json') as file:
            json_data = json.load(file)
        train_list = []
        test_list = []
        for item in json_data:
            item['file_path'] = data_path + 'imgs/' + item['file_path']
            if item['split'] == 'train':
                train_list.append(item)
            elif item['split'] == 'test':
                test_list.append(item)
        return train_list, {}, test_list

    elif CFG.dataset == 'RSTPReid':
        data_path = project_path + 'data/RSTPReid/'
        with open(data_path + 'data_captions.json') as file:
            json_data = json.load(file)
        train_list, val_list, test_list = [], [], []
        for item in json_data:
            item['img_path'] = data_path + 'imgs/' + item['img_path']
            if item['split'] == 'train':
                train_list.append(item)
            elif item['split'] == 'val':
                val_list.append(item)
            elif item['split'] == 'test':
                test_list.append(item)
        return train_list, val_list, test_list

    else:
        NameError('Dataset not found!')
