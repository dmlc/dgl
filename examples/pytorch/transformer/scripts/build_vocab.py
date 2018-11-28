import sys

N = 10000
word_dict = {}
for filename in sys.argv[1:-1]:
    with open(filename, 'r') as f:
        for sentence in f:
            for token in sentence.strip().split():
                if token not in word_dict:
                    word_dict[token] = 0
                else:
                    word_dict[token] += 1

#sorted_words = sorted(word_dict.items(), key=lambda pair:pair[1], reverse=True)

with open(sys.argv[-1], 'w') as f:
    for k, v in word_dict.items(): #sorted_words[:10000]:
        if v > 2:
            f.write(k + '\n')
            
