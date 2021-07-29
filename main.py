# -*- coding: utf-8 -*-
"""
Created on Fri May 04 22:22:49 2021

@author: tejas1234
"""
from scipy import misc
import os
from skimage.transform import resize
import imageio
import torch
import torch.nn as nn
import math
import numpy as np
import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau



dataset = []
examples = []

data_root = "./data/"
alphabets = os.listdir(data_root + "images_background")
for alphabet in alphabets:
    characters = os.listdir(os.path.join(data_root, "images_background", alphabet))
    for character in characters:
        files = os.listdir(os.path.join(data_root, "images_background", alphabet, character))
        examples = []
        for img_file in files:
            img = resize(
                imageio.imread(os.path.join(data_root, "images_background", alphabet, character, img_file)), [28, 28])
            examples.append(img)
        dataset.append(examples)


data_root = "./data/"
alphabets = os.listdir(data_root + "images_evaluation")
for alphabet in alphabets:
    characters = os.listdir(os.path.join(data_root, "images_evaluation", alphabet))
    for character in characters:
        files = os.listdir(os.path.join(data_root, "images_evaluation", alphabet, character))
        examples = []
        for img_file in files:
            img = resize(
                imageio.imread(os.path.join(data_root, "images_evaluation", alphabet, character, img_file)), [28, 28])
            examples.append(img)
        dataset.append(examples)

np.save(data_root + "dataset.npy", np.asarray(dataset))

class OmniglotNShotDataset():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2017, shuffle=True, use_cache=True):
        
        np.random.seed(seed)
        self.x = np.load('./data/dataset.npy')
        self.x = np.reshape(self.x, newshape=(self.x.shape[0], self.x.shape[1], 28, 28, 1))
        if shuffle:
            np.random.shuffle(self.x)
        self.x_train, self.x_val, self.x_test = self.x[:1200], self.x[1200:1411], self.x[1411:]
        self.x_train = self.processes_batch(self.x_train, np.mean(self.x_train), np.std(self.x_train))
        self.x_test = self.processes_batch(self.x_test, np.mean(self.x_test), np.std(self.x_test))
        self.x_val = self.processes_batch(self.x_val, np.mean(self.x_val), np.std(self.x_val))
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.use_cache = use_cache
        if self.use_cache:
            self.cached_datatset = {"train": self.load_data_cache(self.x_train),
                                    "val": self.load_data_cache(self.x_val),
                                    "test": self.load_data_cache(self.x_test)}

    def processes_batch(self, x_batch, mean, std):
        
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack):
       
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), np.float32)

        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
        target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]), np.float32)
        target_y = np.zeros((self.batch_size, 1), np.int32)

        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)
            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

            x_temp = data_pack[choose_classes]
            x_temp = x_temp[:, choose_samples]
            y_temp = np.arange(self.classes_per_set)
            support_set_x[i] = x_temp[:, :-1]
            support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
            target_x[i] = x_temp[choose_label, -1]
            target_y[i] = y_temp[choose_label]

        return support_set_x, support_set_y, target_x, target_y

    def _rotate_data(self, image, k):
        
        return np.rot90(image, k)

    def _rotate_batch(self, batch_images, k):
        
        batch_size = batch_images.shape[0]
        for i in np.arange(batch_size):
            batch_images[i] = self._rotate_data(batch_images[i], k)
        return batch_images

    def _get_batch(self, dataset_name, augment=False):
        
        if self.use_cache:
            support_set_x, support_set_y, target_x, target_y = self._get_batch_from_cache(dataset_name)
        else:
            support_set_x, support_set_y, target_x, target_y = self._sample_new_batch(self.datatset[dataset_name])
        if augment:
            k = np.random.randint(0, 4, size=(self.batch_size, self.classes_per_set))
            a_support_set_x = []
            a_target_x = []
            for b in range(self.batch_size):
                temp_class_set = []
                for c in range(self.classes_per_set):
                    temp_class_set_x = self._rotate_batch(support_set_x[b, c], k=k[b, c])
                    if target_y[b] == support_set_y[b, c, 0]:
                        temp_target_x = self._rotate_data(target_x[b], k=k[b, c])
                    temp_class_set.append(temp_class_set_x)
                a_support_set_x.append(temp_class_set)
                a_target_x.append(temp_target_x)
            support_set_x = np.array(a_support_set_x)
            target_x = np.array(a_target_x)
        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                               support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, target_x, target_y

    def get_train_batch(self, augment=False):
        return self._get_batch("train", augment)

    def get_val_batch(self, augment=False):
        return self._get_batch("val", augment)

    def get_test_batch(self, augment=False):
        return self._get_batch("test", augment)

    def load_data_cache(self, data_pack, argument=True):
        cached_dataset = []
        classes_idx = np.arange(data_pack.shape[0])
        samples_idx = np.arange(data_pack.shape[1])
        for _ in range(1000):
            support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                      data_pack.shape[3], data_pack.shape[4]), np.float32)

            support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
            target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                                np.float32)
            target_y = np.zeros((self.batch_size, 1), np.int32)
            for i in range(self.batch_size):
                choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
                choose_label = np.random.choice(self.classes_per_set, size=1)
                choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

                x_temp = data_pack[choose_classes]
                x_temp = x_temp[:, choose_samples]
                y_temp = np.arange(self.classes_per_set)
                support_set_x[i] = x_temp[:, :-1]
                support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
                target_x[i] = x_temp[choose_label, -1]
                target_y[i] = y_temp[choose_label]
            cached_dataset.append([support_set_x, support_set_y, target_x, target_y])
        return cached_dataset

    def _get_batch_from_cache(self, dataset_name):
        
        if self.indexes[dataset_name] >= len(self.cached_datatset[dataset_name]):
            self.indexes[dataset_name] = 0
            self.cached_datatset[dataset_name] = self.load_data_cache(self.datatset[dataset_name])
        next_batch = self.cached_datatset[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target

def convLayer(in_channels, out_channels, keep_prob=0.0):
    
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq

class Classifier(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28):
        super(Classifier, self).__init__()
        
        self.layer1 = convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = convLayer(layer_size, layer_size, keep_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

    def forward(self, image_input):
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)
        return x

class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds
    
class DistanceNetwork(nn.Module):
    
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()
    
class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim,use_cuda):
        super(BidirectionalLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)

    def init_hidden(self,use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda(),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False))

    def repackage_hidden(self,h):
        
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        
        self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output

class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, batch_size=32, num_channels=1, learning_rate=1e-3, fce=False, num_classes_per_set=20, \
                 num_samples_per_class=1, image_size=28, use_cuda=True):
        
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.fce = fce
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.g = Classifier(layer_size=64, num_channels=num_channels, keep_prob=keep_prob, image_size=image_size)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        if self.fce:
            self.lstm = BidirectionalLSTM(layer_size=[32], batch_size=self.batch_size, vector_dim=self.g.outSize,use_cuda=use_cuda)

    def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
        
       
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, i, :, :])
            encoded_images.append(gen_encode)

        
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode)
        output = torch.stack(encoded_images)

        
        if self.fce:
            outputs = self.lstm(output)

        
        similarites = self.dn(support_set=output[:-1], input_image=output[-1])

        
        preds = self.classify(similarites, support_set_y=support_set_y_one_hot)

        
        values, indices = preds.max(1)
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())

        return accuracy, crossentropy_loss

class OmniglotBuilder:
    def __init__(self, data):
        """
        Initializes the experiment
        :param data:
        """
        self.data = data

    def build_experiment(self, batch_size, num_channels, lr, image_size, classes_per_set, samples_per_class, keep_prob,
                         fce, optim, weight_decay, use_cuda):
        
        self.classes_per_set = classes_per_set
        self.sample_per_class = samples_per_class
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.optim = optim
        self.wd = weight_decay
        self.isCuadAvailable = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.matchNet = MatchingNetwork(keep_prob, batch_size, num_channels, self.lr, fce, classes_per_set,
                                        samples_per_class, image_size, self.isCuadAvailable & self.use_cuda)
        self.total_iter = 0
        if self.isCuadAvailable & self.use_cuda:
            cudnn.benchmark = True  # set True to speedup
            torch.cuda.manual_seed_all(2017)
            self.matchNet.cuda()
        self.total_train_iter = 0
        self.optimizer = self._create_optimizer(self.matchNet, self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',verbose=True)

    def run_training_epoch(self, total_train_batches):
        
        total_c_loss = 0.0
        total_accuracy = 0.0
        

        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch(True)
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.classes_per_set).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                
                size = x_support_set.size()
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                if self.isCuadAvailable & self.use_cuda:
                    acc, c_loss = self.matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                y_target.cuda())
                else:
                    acc, c_loss = self.matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

                
                self.optimizer.zero_grad()
                c_loss.backward()
                self.optimizer.step()

                

                iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                

            total_c_loss = total_c_loss / total_train_batches
            total_accuracy = total_accuracy / total_train_batches
            return total_c_loss, total_accuracy

    def _create_optimizer(self, model, lr):
        
        if self.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=self.wd)
        else:
            raise Exception("Not a valid optimizer offered: {0}".format(self.optim))
        return optimizer

    def _adjust_learning_rate(self, optimizer):
        

    def run_val_epoch(self, total_val_batches):
        
        total_c_loss = 0.0
        total_accuracy = 0.0

        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_val_batch(False)
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                 
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.classes_per_set).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                
                size = x_support_set.size()
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                if self.isCuadAvailable & self.use_cuda:
                    acc, c_loss = self.matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                y_target.cuda())
                else:
                    acc, c_loss = self.matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

                

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_val_batches
            total_accuracy = total_accuracy / total_val_batches
            self.scheduler.step(total_c_loss)
            return total_c_loss, total_accuracy

    def run_test_epoch(self, total_test_batches):
        
        total_c_loss = 0.0
        total_accuracy = 0.0

        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch(False)
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                # convert to one hot encoding
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.classes_per_set).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                # reshape channels and change order
                size = x_support_set.size()
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                if self.isCuadAvailable & self.use_cuda:
                    acc, c_loss = self.matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                y_target.cuda())
                else:
                    acc, c_loss = self.matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

                # TODO: update learning rate?

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_test_batches
            total_accuracy = total_accuracy / total_test_batches
            return total_c_loss, total_accuracy
        
batch_size = 20
fce = True
classes_per_set = 5
samples_per_class = 1
channels = 1
# Training setup
total_epochs = 100
total_train_batches = 1000
total_val_batches = 250
total_test_batches = 500
best_val_acc = 0.0

data = OmniglotNShotDataset(batch_size=batch_size, classes_per_set=classes_per_set,
                            samples_per_class=samples_per_class, seed=2017, shuffle=True, use_cache=False)
obj_oneShotBuilder = OmniglotBuilder(data)
obj_oneShotBuilder.build_experiment(batch_size=batch_size, num_channels=1, lr=1e-3, image_size=28, classes_per_set=20,
                                    samples_per_class=1, keep_prob=0.0, fce=True, optim="adam", weight_decay=0,
                                    use_cuda=True)

with tqdm.tqdm(total=total_train_batches) as pbar_e:
    for e in range(total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches)
        print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_val_epoch(total_val_batches)
        print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
        if total_val_accuracy>best_val_acc:
            best_val_acc = total_val_accuracy
            total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_test_epoch(total_test_batches)
            print("Epoch {}: test_loss:{} test_accuracy:{}".format(e, total_test_c_loss, total_test_accuracy))
        pbar_e.update(1)
