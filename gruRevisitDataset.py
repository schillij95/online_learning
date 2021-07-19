'''
Created on 30.06.2021

@author: Julian
'''
import tensorflow as tf
import numpy as np
import copy

import loader
import keras_VAE_ENSEMBLE_GRU

'''this needs testing!'''
class AnalizeTrainingDataGRU():
    def __init__(self, model, max_size = 10, batch_size = 16, gru_batch_size = 24, total = 1, nr = 0, parrent = None):
        self.model = model
        self.revisit_examples = [None] * max_size
        self.revisit_ids = np.zeros((max_size)) - 1
        self.revisit_values = np.zeros((max_size)) - 1
        self.revisit_updated = np.zeros((max_size))
        self.revisit_trained_nr = np.zeros((max_size))
        self.max_size = max_size
        self.position = 0
        self.average = 0
        self.batch_size = batch_size
        self.gru_batch_size = gru_batch_size
        self.added = 0
        self.checked = 0
        self.parrent = parrent
        self.child = None
        self.nr = nr
        self.total = total
        if (self.nr + 1) < self.total:
            self.child = AnalizeTrainingDataGRU(model=self.model, max_size=self.max_size, batch_size=self.batch_size, gru_batch_size=self.gru_batch_size, total = self.total, nr = self.nr + 1, parrent = self)

    
    #call this function with a list of experiment runs to add the run to the revisit dataset if the runs meet the requirements
    def analize_examples(self, runs_raw, examples_values, example_ids):
        max_id = np.max(example_ids)
        mean_list = runs_raw[0]
        sampeled_list = runs_raw[1]
        actions_list = runs_raw[2]
        dones_list = runs_raw[3]
        for i in range(len(example_ids)):
            #do not look at runs that have only trained once in the normal dataset
            if example_ids[i] > max_id - 6:
                continue
            self.analize_example([mean_list[i], sampeled_list[i], actions_list[i], dones_list[i]], examples_values[i], example_ids[i])
        
    #call this function with an experiment run to add the run to the revisit dataset if the run meets requirements
    def analize_example(self, example, example_value, example_id):
        #get errors of examples
        #check example if we want to add it to revisit
        self.checked += 1
        if self.check_example(example_value, example_id):
            self.add_example(example, example_value, example_id)
            self.added += 1
            
    def verbose_added(self):
        print("Added elements to revisit dataset: ", self.added, self.checked)
        self.added = 0
        self.checked = 0
            
    def check_example(self, example_value, example_id):
        return example_value > self.average and not (example_id in self.revisit_ids)
    
    def add_example(self, example, example_value, example_id):
        #search spot with example that has low loss to put new example inside
        start = self.position
        while self.revisit_values[self.position % self.max_size] > self.average and self.position < start + self.max_size:
            self.position += 1
        self.position = self.position % self.max_size
        
        self.revisit_examples[self.position] = copy.deepcopy(example)
        self.revisit_values[self.position] = example_value
        self.revisit_ids[self.position] = example_id
        self.position = (self.position + 1) % self.max_size
        self.update_average()
        #debug
        #print(self.revisit_values)
        
    def add_run(self, example, example_id, example_value = float("inf"), trained_nr = 0):
        min_value = float("inf")
        min_index = -1
        for i in range(self.max_size):
            if ((self.revisit_trained_nr[i] > 2) or (self.revisit_ids[i] < 0)) and (self.revisit_values[i] < min_value):
                min_value = self.revisit_values[i]
                min_index = i
        if min_index == -1:
            print("please expand revisit dataset size! it is too small")
        

        self.evict_example(min_index)
        
        self.revisit_examples[min_index] = copy.deepcopy(example)
        self.revisit_ids[min_index] = example_id
        self.revisit_values[min_index] = example_value
        self.revisit_trained_nr[min_index] = trained_nr
        return
    
    #todo moving runs back up
    def evict_example(self, i):
        # no example to evict at this position
        if self.revisit_examples[i] is None:
            print("free space in dataset found ", i)
            return
        # example has lowest priority in entire dataset list
        if self.child is None:
            print("removing run with loss ", self.revisit_values[i])
            return
        # evict to child revisit dataset
        print("shuffle run to deeper dataset")
        self.child.add_run(self.revisit_examples[i], self.revisit_ids[i], self.revisit_values[i], self.revisit_trained_nr[i])
        return
        
    def update_average(self):
        self.average = 1.0 * sum(self.revisit_values) / self.max_size
        
    #trains model on revisit data and updates their weights
    def train_revisit(self):
        print("saving revisit dataset")
        self.save()
        print("training revisit")
        
        
        dataset_vae_wm, done_array = self.create_datasets()
        if dataset_vae_wm is None:
            print("nothing to train, revisit dataset is empty")
            return
             
        # warm up landmark grus before taking landmarks in fit/train_step
        for i in range(2):
            u = 0
            for x in dataset_vae_wm:
                #print(u)
                u += 1
                for ensemble in range(self.model.ensemble_nr):
                    _, _, _ = self.model.gru_landmark_ensemble[ensemble](x)
                    _, _, _ = self.model.gru_ensemble[ensemble](x)
                    
        #dataset_code_mean = dataset_code_mean.prefetch(20)
        run_loss = []
        run_ids  = []
        epochs = 10
        self.model.is_revisit = True
        history = self.model.fit(dataset_vae_wm, epochs=epochs, callbacks=[keras_VAE_ENSEMBLE_GRU.UpdateWeightsCallback(run_loss, run_ids, done_array, self.gru_batch_size * self.batch_size, epochs)], shuffle=False, use_multiprocessing=True, verbose=False)
        self.model.is_revisit = False
        print(run_loss)
        #print(run_ids)
        print(history.history)
        
        self.distribute_loss(run_loss, run_ids)
        print("training revisit done, ", self.average)
        
    # create tensorflow dataset from saved revisit runs
    def create_datasets(self):
        # collect all data in revisit dataset and its children
        runs, ids, weights = self.collect()
        
        #extract mean, sampled, actions and dones from revisit example runs data
        mean_list = []
        sampeled_list = []
        actions_list = []
        dones_list = []
        for i, x in enumerate(runs):
            if not (x is None):
                mean_list.append(x[0])
                sampeled_list.append(x[1])
                actions_list.append(x[2])
                dones_list.append(x[3])
                
        # convert data to numpy
        if len(mean_list) < 1:
            return None, None, None, None
        
        mean_np = np.concatenate(mean_list)
        sample_np = np.concatenate(sampeled_list)
        actions_np = np.concatenate(actions_list)
        done_array = np.concatenate(dones_list)
        weights_np = np.concatenate(weights)
        
        
        
        # convert to tensorflow datasets
        dataset_code_mean = tf.data.Dataset.from_tensor_slices(mean_np)
        dataset_code_sampled = tf.data.Dataset.from_tensor_slices(sample_np)
        dataset_actions = tf.data.Dataset.from_tensor_slices(actions_np)
        dataset_weights = tf.data.Dataset.from_tensor_slices(weights_np)
        
        dataset_vae_wm = loader.load_dataset_img_code(dataset_code_mean, dataset_code_sampled, dataset_actions, dataset_weights, self.batch_size)
        dataset_vae_wm = dataset_vae_wm.batch(self.gru_batch_size, drop_remainder=True).prefetch(10).cache()
        ''''i = 0
        for x in dataset_vae_wm:
            i += 1
            print(i, x)
        print(mean_np.shape, sample_np.shape, actions_np.shape, done_array.shape, weights_np.shape)
        print(dataset_vae_wm)'''
        return dataset_vae_wm, done_array
    
    # collect all data including offspring data
    def collect(self):
        runs = []
        ids = []
        weights = []
        if not (self.child is None):
            runs, ids, weights = self.child.collect()
        
        max_pos = 0
        while(max_pos < len(self.revisit_examples) and not self.revisit_examples[max_pos] is None):
            max_pos += 1

        runs.extend(self.revisit_examples[:max_pos])
        if len(self.revisit_ids[:max_pos]) > 0:
            ids.extend(self.revisit_ids[:max_pos])
        
        #append right nr of examples?
        
        for i in range(max_pos):
            weights_list = []
            for _ in range(len(self.revisit_examples[i][0])):
                weights_list.append((self.total - self.nr) / (1.0 * self.total))
            weights.append(np.asarray(weights_list))
        
        return runs, ids, weights
    
    def distribute_loss(self, run_loss, run_ids):
        # recursive
        if not (self.child is None):
            self.child.distribute_loss(run_loss, run_ids)
            
        # set update tracked to 0
        self.revisit_updated = np.zeros((self.max_size))
        
        #update and move to distribute_loss()
        for i, run_id in enumerate(run_ids):
            positions = np.where(self.revisit_ids == run_id)[0]
            if len(positions) < 1: #run_id not in this part of the dataset
                #print("error in self.revisit_ids vs run_ids")
                continue
            
            
            
            position = [0]
            if self.revisit_values[position] < 0:
                self.revisit_values[position] = run_loss[i]
                self.revisit_updated[position] = 1
            else:
                self.revisit_values[position] = (2.0 * self.revisit_values[position] + run_loss[i]) / 3.0
                self.revisit_updated[position] = 1
            
        for i in range(len(self.revisit_updated)):
            # count how many times each run got trained
            self.revisit_trained_nr[i] += 1
            # housekeeping for the trained runs that havent gotten their loss updated
            if self.revisit_updated[i] < 1:
                self.revisit_values[i] = 0.8 * self.revisit_values[i]
        
        
        self.update_average()
    
    def load(self):
        # recursive load
        if not (self.child is None):
            self.child.load()
            
        #load this dataset
        for i in range(self.max_size):
            for u in range(4):
                try:
                    file_location_example = "revisit_dataset/dataset" + str(self.nr) + "example" + str(i) + "part" + str(u)
                    with open(file_location_example, 'rb') as f_example:
                        self.revisit_examples[i][u] = np.load(f_example)
                except:
                    print("no example data at position ", i, " ", u, "for dataset nr ", self.nr)
        try:
            file_location_trained_nr = "revisit_dataset/dataset" + str(self.nr) + "trained_nrs"
            with open(file_location_trained_nr, 'rb') as f_trained_nr:
                self.revisit_trained_nr = np.load(f_trained_nr)
            file_location_value = "revisit_dataset/dataset" + str(self.nr) + "values"
            with open(file_location_value, 'rb') as f_value:
                self.revisit_values = np.load(f_value)
            file_location_id = "revisit_dataset/dataset" + str(self.nr) + "ids"
            with open(file_location_id, 'rb') as f_id:
                self.revisit_ids = np.load(f_id)
        except:
            print("no data at position ", i, "for dataset nr ", self.nr)
        return 
    
    def save(self):
        # recursive save
        if not (self.child is None):
            self.child.save()
        
        for i, example in enumerate(self.revisit_examples):
            #save
            if not (example is None):
                for u in range(len(self.revisit_examples[i])):
                    file_location_example = "revisit_dataset/dataset" + str(self.nr) + "example" + str(i) + "part" + str(u)
                    with open(file_location_example, 'wb') as f_example:
                        np.save(f_example, self.revisit_examples[i][u])
        file_location_trained_nr = "revisit_dataset/dataset" + str(self.nr) + "trained_nrs"
        with open(file_location_trained_nr, 'wb') as f_trained_nr:
            np.save(f_trained_nr, self.revisit_trained_nr)
        file_location_value = "revisit_dataset/dataset" + str(self.nr) + "values"
        with open(file_location_value, 'wb') as f_value:
            np.save(f_value, self.revisit_values)
        file_location_id = "revisit_dataset/dataset" + str(self.nr) + "ids"
        with open(file_location_id, 'wb') as f_id:
            np.save(f_id, self.revisit_ids)
                
        return
        
    def get_max_id(self):
        max_child = -1
        if not (self.child is None):
            max_child = self.child.get_max_id()
        self_max = np.max(self.revisit_ids)
        return int(max([self_max, max_child]))
    