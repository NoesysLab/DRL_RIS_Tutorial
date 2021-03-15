import numpy as np
import itertools
from datetime import datetime



class BinaryEnumerator:


    def __init__(self, batch_size, num_digits):
        self.B            = np.array([[0],[1]], dtype=np.byte)
        self.k            = 1
        self.batch_size   = batch_size
        self.num_digits   = num_digits
        self.max_number   = np.power(2, num_digits)-1
        self.curr_padding = self.num_digits - self.k

        self._curr_num   = 0


    def _expand_array(self):
        self.B = np.block([[np.zeros((2**self.k,1), dtype=np.byte), self.B],
                           [np.ones((2**self.k,1),  dtype=np.byte), self.B]])
        self.k += 1
        self.curr_padding = self.num_digits - self.k


    def __iter__(self):
        self._curr_num = 0
        return self


    def __next__(self):
        if self._curr_num > self.max_number:
            raise StopIteration


        this_batch_size = min(self.batch_size, self.max_number-self._curr_num+1)

        while self._curr_num + this_batch_size > np.power(2, self.k):
            self._expand_array()

        batch = self.B[self._curr_num:self._curr_num+this_batch_size,:]
        self._curr_num += this_batch_size


        if self.curr_padding>0:
            batch = np.hstack([np.zeros((batch.shape[0], self.curr_padding), dtype=np.byte),
                               batch])

        return batch





if __name__ == '__main__':

    K           = 26
    batch_size  = 2048
    num_batches = int(2**K)//batch_size



    start = datetime.now()
    it = itertools.product([0,1], repeat=K)
    for _ in range(num_batches):
        B = np.array([next(it) for _ in range(batch_size)])
    end = datetime.now()
    print("Itertools: {}".format(end-start))



    start = datetime.now()
    be = BinaryEnumerator(batch_size, K)
    for _ in range(num_batches):
        B = next(be)
    end = datetime.now()
    print("Itertools: {}".format(end-start))
