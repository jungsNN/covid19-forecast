class creator:
    def __init__(self):
        self.batchload = batchload
        self.idx_list = []
        self.idx_history = {}
        self.current_batch = current_batch
        self.batch_size = 0
        

    def batchload(self, dataset, batch_size=1, train=True, shuffle=True, step=1):
        self.batch_size += batch_size
        
        input_data, output_data = (dataset[0].view(dataset[0].shape[1],
                                                   dataset[0].shape[2]),
                                   dataset[1].view(dataset[1].shape[1],
                                                   dataset[1].shape[2]))
        zip_data = input_data, output_data
        idx_list = list(range(len(dataset[0])))
        random_idx = np.random.shuffle(idx_list)

        if batch_size == 1:
            if shuffle == True:
                return input_data[random_idx], output_data[random_idx]

            else:
                return input_data, output_data

        elif batch_size > 1:

            if shuffle == True:

                if train == True:
                    i = 1
                    if len(set(idx_list)) < len(dataset):
                        batch = np.random.choice(random_idx, size=batch_size)
                        self.idx_history[i] = batch
                        
                        
                        i += 1
                    
                    if self.idx_history[i] != []:
                        self.current_batch = batch
                        return zip(input_data[batch], output_data[batch])
            
                    
            else:
                self.current_batch = idx_list[batch_size-batch_size:batch_size]#input_data[:batch_size], output_data[:batch_size]
                #nput_data, output_data = input_data[batch_size:], output_data[batch_size:]
                
                batch_size+=batch_size
            

                
