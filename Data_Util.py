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
        edge_set = set()
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
                edge_set.add((vertex1,vertex2))
                edge_set.add((vertex2,vertex1))
                weight = float(items[2])
                labels1 = [int(x)-1 for x in items[3].split(' ')]
                labels2 = [int(x)-1 for x in items[4].split(' ')]
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
        num_class = len(labels_set)
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
        self.edge_set = edge_set

        self.iedge_set = set()

        log('transforming the done!')
        log('train size : %d,  test size: %d' % (self.train_num, self.test_num))


    def generate_negative_set(self,num=100000):
        self.iedge_set = set()

        print('Sampling negatives')
        while(len(self.iedge_set) < num):
            x = 0
            y = 0
            while x==y or (x,y) in self.edge_set:
                x = np.random.randint(0, self.num_vertex)
                y = np.random.randint(0, self.num_vertex)
            self.iedge_set.add((x,y))
        print('Done')

    def next_batch(self,batch_size, mode='train'):

        h = []
        t = []
        ih = []
        it = []
        if mode == 'train':
            batch_ids = np.array(random.sample(self.train_ids, batch_size), dtype=np.int32)
            correct_relations = random.sample(self.edge_set, batch_size)
            incorrect_relations = random.sample(self.iedge_set, batch_size)
            h = [x[0] for x in correct_relations]
            t = [x[1] for x in correct_relations]
            ih = [x[0] for x in incorrect_relations]
            it = [x[1] for x in incorrect_relations]
            h = self.adj_matrix[h]
            t = self.adj_matrix[t]
            ih = self.adj_matrix[ih]
            it = self.adj_matrix[it]
        elif mode == 'test':
            batch_ids = np.array(random.sample(self.test_ids, batch_size), dtype=np.int32)
        x = np.array(self.adj_matrix[self.x[batch_ids],:])
        y = np.array(self.y[batch_ids])
        return x, y,h,t,ih,it

    def next_infer_batch(self,batch_size):
        if self.infer_step < len(self.x):
            batch_ids = np.array(self.train_ids[self.infer_step,self.infer_step+batch_size],dtype=np.int32)
            x = np.array(self.adj_matrix[self.x[batch_ids], :])
            self.infer_step += batch_size
            return x
        else:
            return None


if __name__ =='__main__':
    test = DataUtil(max_line=100000)
    test.generate_negative_set(10000)
    for i in range(0,5000):
        x, y,h,t,ih,it = test.next_batch(128)
        print(x)
        print(np.shape(x))
