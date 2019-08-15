import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    #f = open('/Users/ziqiaom/Desktop/in_aminer/aminer.cac.w1000.l100.txt', encoding="ISO-8859-1")
    #f = open('/Users/ziqiaom/Desktop/in_aminer/aminer.cac.w1000.l100.txt', encoding="utf-8")
    #f1 = open("/Users/ziqiaom/Desktop/in_aminer/aminer.txt", "w+")
    #f = f.readlines()[3:]
    #count = 0
    #for line in f:
        #count += 1
        #if count < 4:
            #print(line)
    #f.close()
    #file = open("/Users/ziqiaom/Desktop/label 2/googlescholar.8area.venue.label.txt")
    #a = np.random.shuffle([1,2,3,4])
    #print(a)
    #neg_v = numpy.random.choice(self.sample_table, size=(len(pos_word_pair), count)).tolist()
    for i in tqdm(range(100000000)):
        if i % 1000000 == 0:
            print("lalala")



