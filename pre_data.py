'''
api of layers
'''
import os
import numpy as np
from glob import glob
from PIL import Image
from dataset import PATH
'''
{'num_cases_per_batch': 10000, 
'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_vis': 3072}

'''
cifar10_dict = {
'num_cases_per_batch': 10000,
'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
'num_vis': 3072
}


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def get_datalist(data_dir, data_pattern):
    data_dir = PATH + data_dir
    image_list = glob(os.path.join(data_dir, data_pattern))
    return (image_list)


def get_cifar10(data_dir, name, w, h):
    pic_list = []
    data_dir = PATH + data_dir
    list = cifar10_dict['label_names']
    id = None
    for i in range(len(list)):
        if name == list[i]:
            id = i
    for i in range(1, 6):
        data = unpickle(data_dir + '/data_batch_' + str(i))
        for index in range(len(data['labels'])):
            if data['labels'][index] == id:
                pic_list.append(data['data'][index])
        pass
    pic_list = np.array(pic_list)
    pic_list = np.reshape(pic_list, newshape=[-1, 32, 32, 3], order='F')
    pic_list = np.swapaxes(pic_list, 1, 2)
    pic_list = np.array(pic_list, dtype=np.uint8)
    new_pic_list = []
    for i in range(len(pic_list)):
        im = Image.fromarray(pic_list[i])
        im = im.resize(size=(h, w))
        new_pic_list.append(np.array(im, dtype=np.float32) / 127.5 - 1)
    return np.array(new_pic_list)


def show_pic(data):
    data = np.squeeze(data)
    im = Image.fromarray(data)
    im.show()
    im.save(fp=PATH + '/img/' + str())


def get_image(image_list, batch_size, img_h, img_w):
    image_batch = []
    for img in image_list:
        data = Image.open(img)
        data = data.resize((img_h, img_w))
        data = np.array(data)
        data = data.astype('float32') / 127.5 - 1
        image_batch.append(data)
    return image_batch


def restruct_image(x, batch_size):
    image_batch = []
    for k in range(batch_size):
        data = x[k, :, :, :]
        data = (data + 1) * 127.5
        # data = np.clip(data,0,255).astype(np.uint8)
        image_batch.append(data)
    return (image_batch)


if __name__ == '__main__':
    a = get_cifar10(data_dir='/cifar/', name='dog', w=96, h=96)

