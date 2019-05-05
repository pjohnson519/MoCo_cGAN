
import numpy as np

def data_generator(input_imgs, target_imgs, batch_size=1):
    num_images=target_imgs.shape[0]
    for batch_num in range(0, num_images, batch_size):
        i = batch_num
        i_end = i + batch_size
        x_batch=np.array(input_imgs[i:i_end,:,:,:,:)
        y_batch=np.array(target_imgs[i:i_end,:,:,:,:])
                                                
        #print(x_batch.shape) 
                                                                    
        yield x_batch, y_batch
                                                                
