import random

class GeneDataset:
    def __init__(self, A, B, unaligned=False):

        self.numbersA = A
        self.numbersB = B
        self.unaligned = unaligned
    def __getitem__(self, index):

        if self.unaligned:
            itemA = self.numbersA[random.randint(0,len(self.numbersA)-1)]
            itemB = self.numbersB[index % len(self.numbersB)]

        else:
            itemA = self.numbersA[index % len(self.numbersA)]


        return {'A': itemA, 'B': itemB}

    def __len__(self):
        return max(len(self.numbersA), len(self.numbersB))

