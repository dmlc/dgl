import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import tqdm
import logging
import sys
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
log_file = '{}-{}.log'.format("dgl_test", "aminer")
logger.addHandler(logging.FileHandler("log/" + log_file))

if __name__ == "__main__":
    venue_count = 133
    author_count = 246678
    experiment_times = 1
    percents = np.arange(0.1, 1, 0.1)
    file = open("./out/myout0.txt")
    file_1 = open("data/label/googlescholar.8area.venue.label.txt")
    file_2 = open("data/label/googlescholar.8area.author.label.txt")
    check_venue = {}
    check_author = {}
    for line in file_1:
        venue_label = line.strip().split(" ")
        check_venue[venue_label[0]] = int(venue_label[1])
    for line in file_2:
        author_label = line.strip().split(" ")
        check_author[author_label[0]] = int(author_label[1])
    venue_embed_dict = {}
    author_embed_dict = {}
    # collect embeddings separately in dictionary form
    file.readline()
    print("read line by line")
    for line in file:
        embed = line.strip().split(' ')
        if embed[0] in check_venue:
            venue_embed_dict[embed[0]] = []
            for i in range(1, len(embed), 1):
                venue_embed_dict[embed[0]].append(float(embed[i]))
        if embed[0] in check_author:
            author_embed_dict[embed[0]] = []
            for j in range(1, len(embed), 1):
                author_embed_dict[embed[0]].append(float(embed[j]))
    #get venue embeddings
    print("reading finished")
    for percent in percents:
        venues = list(venue_embed_dict.keys())
        authors = list(author_embed_dict.keys())
        macro_average_venue = 0
        micro_average_venue = 0
        macro_average_author = 0
        micro_average_author = 0
        for time in range(experiment_times):
            print("one more time")
            np.random.shuffle(venues)
            np.random.shuffle(authors)
            venue_embedding = np.zeros((len(venues),128))
            author_embedding = np.zeros((len(authors),128))
            print("collecting venue embeddings")
            index=0
            for venue in tqdm.tqdm(venues):
                temp = np.array(venue_embed_dict[venue])
                venue_embedding[index,:]=temp
                index+=1
            print("collecting author embeddings")
            index=0
            for author in tqdm.tqdm(authors):
                # print("one more author " + str(count))
                temp_1 = np.array(author_embed_dict[author])
                author_embedding[index, :] = temp_1
                index += 1
            # split data into training and testing
            print("splitting")
            venue_split = int(venue_count * percent)
            venue_training = venue_embedding[:venue_split,:]
            venue_testing = venue_embedding[venue_split:,:]
            author_split = int(author_count * percent)
            author_training = author_embedding[:author_split,:]
            author_testing = author_embedding[author_split:,:]
            # split label into training and testing
            venue_label = []
            venue_true = []
            author_label = []
            author_true = []
            for i in range(len(venues)):
                if i < venue_split:
                    venue_label.append(check_venue[venues[i]])
                else:
                    venue_true.append(check_venue[venues[i]])
            venue_label = np.array(venue_label)
            venue_true = np.array(venue_true)
            for j in range(len(authors)):
                if j < author_split:
                    author_label.append(check_author[authors[j]])
                else:
                    author_true.append(check_author[authors[j]])
            author_label = np.array(author_label)
            author_true = np.array(author_true)
            file.close()
            print("beging predicting")
            clf_venue = LogisticRegression(random_state=0, solver="lbfgs", multi_class="multinomial").fit(venue_training,venue_label)
            y_pred_venue = clf_venue.predict(venue_testing)
            clf_author = LogisticRegression(random_state=0, solver="lbfgs", multi_class="multinomial").fit(author_training,author_label)
            y_pred_author = clf_author.predict(author_testing)
            macro_average_venue += f1_score(venue_true, y_pred_venue, average="macro")
            micro_average_venue += f1_score(venue_true, y_pred_venue, average="micro")
            macro_average_author += f1_score(author_true, y_pred_author, average="macro")
            micro_average_author += f1_score(author_true, y_pred_author, average="micro")
        print(macro_average_venue/float(experiment_times))
        print(micro_average_venue/float(experiment_times))
        print(macro_average_author / float(experiment_times))
        print(micro_average_author / float(experiment_times))
        logger.info(">>>>>>>>>>"+str(percent)+ ">>>>>>>>>>>>>")
        logger.info("macro_average_venue: " + str(macro_average_venue/float(experiment_times)))
        logger.info("micro_average_venue: " + str(micro_average_venue/float(experiment_times)))
        logger.info("macro_average_author: " +str(macro_average_author / float(experiment_times)))
        logger.info("micro_average_author: " +str(micro_average_author/ float(experiment_times)))