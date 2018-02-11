from Util import log
from Util import array_to_multi_hot
import random
import numpy as np

class DataUtil:
    def __init__(self, reindex_path=r'C:\Users\v-sixwu\Downloads\eca_blogCatalog3.txt.labeled.reindex',max_line = -1,test_rate=0.3, sample_mode=True):
        log('loading Edge-Centic dataset form : %s' % (reindex_path))
        vertex_set = set()
        labels_set = set()
        vertex_list = list()
        label_list = list()
        vertex_label = dict()
        tmp = list()
        with open(reindex_path,'r+',encoding='utf-8') as fin:
            lines = fin.readlines()
            if max_line > 0:
                lines = lines[0:max_line]
            log('total lines: %d' % len(lines))
            for line in lines:
                items = line.strip('\n').split('\t')
                vertex1 = int(items[0])
                vertex2 = int(items[1])
                weight = float(items[2])
                labels1 = [int(x) for x in items[3].split(' ')]
                labels2 = [int(x) for x in items[4].split(' ')]
                overlap_ratio = float(items[5])
                vertex_set.add(vertex1)
                vertex_set.add(vertex2)
                labels_set = labels_set | set(labels1)
                labels_set = labels_set | set(labels2)
                tmp.append((vertex1,labels1,vertex2,labels2))
                vertex_label[vertex1] = labels1
                vertex_label[vertex2] = labels2
        log('the dataset has been loaded!')
        log('total account of vertex: %d' % len(vertex_set))
        log('total labels of vertex: %d' % len(labels_set))
        log('transforming the dataset')
        n = len(vertex_set)
        self.adj_matrix = np.eye(n)
        num_class = len(labels_set) + 1
        for vertex1,labels1,vertex2,labels2 in tmp:
            self.adj_matrix[vertex1][vertex2] = 1
            self.adj_matrix[vertex2][vertex1] = 1
            vertex_list.append(vertex1)
            label_list.append(array_to_multi_hot(labels1, num_class))
            vertex_list.append(vertex2)
            label_list.append(array_to_multi_hot(labels2, num_class))

        self.num_class = num_class
        self.num_vertex = len(vertex_set)
        self.x = np.array(vertex_list)
        self.y = np.array(label_list)
        self.ids = range(0, len(vertex_list))
        self.test_num = int(len(self.ids) * test_rate)
        self.train_num = len(self.ids) - self.test_num
        if sample_mode:
            print('Sample Mode')
            self.ids = random.sample(self.ids,len(self.ids))
        self.train_ids = self.ids[0: self.train_num]
        self.test_ids = self.ids[self.train_num:]

        self.infer_step = 0

        log('transforming the done!')
        log('train size : %d,  test size: %d' % (self.train_num, self.test_num))


    def next_batch(self,batch_size, mode='train'):

        if mode == 'train':
            batch_ids = np.array(random.sample(self.train_ids, batch_size), dtype=np.int32)
        elif mode == 'test':
            batch_ids = np.array(random.sample(self.test_ids, batch_size), dtype=np.int32)
        x = np.array(self.adj_matrix[self.x[batch_ids],:])
        y = np.array(self.y[batch_ids])
        return x,y

    def next_infer_batch(self,batch_size):
        if self.infer_step < len(self.x):
            batch_ids = np.array(self.train_ids[self.infer_step,self.infer_step+batch_size],dtype=np.int32)
            x = np.array(self.adj_matrix[self.x[batch_ids], :])
            self.infer_step += batch_size
            return x
        else:
            return None


if __name__ =='__main__':
    test = DataUtil(max_line=100)
    x, y = test.next_batch(2)
    print(x)
    print(np.shape(x))
