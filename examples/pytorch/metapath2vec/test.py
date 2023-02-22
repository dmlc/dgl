import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


if __name__ == "__main__":
    venue_count = 133
    author_count = 246678
    experiment_times = 1
    percent = 0.05
    file = open(".../output_file_path/...")
    file_1 = open(".../label 2/googlescholar.8area.venue.label.txt")
    file_2 = open(".../label 2/googlescholar.8area.author.label.txt")
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
        embed = line.strip().split(" ")
        if embed[0] in check_venue:
            venue_embed_dict[embed[0]] = []
            for i in range(1, len(embed), 1):
                venue_embed_dict[embed[0]].append(float(embed[i]))
        if embed[0] in check_author:
            author_embed_dict[embed[0]] = []
            for j in range(1, len(embed), 1):
                author_embed_dict[embed[0]].append(float(embed[j]))
    # get venue embeddings
    print("reading finished")
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
        venue_embedding = np.array([])
        author_embedding = np.array([])
        print("collecting venue embeddings")
        for venue in venues:
            temp = np.array(venue_embed_dict[venue])
            if len(venue_embedding) == 0:
                venue_embedding = temp
            else:
                venue_embedding = np.vstack((venue_embedding, temp))
        print("collecting author embeddings")
        count = 0
        for author in authors:
            count += 1
            # print("one more author " + str(count))
            temp_1 = np.array(author_embed_dict[author])
            if len(author_embedding) == 0:
                author_embedding = temp_1
            else:
                author_embedding = np.vstack((author_embedding, temp_1))
        # split data into training and testing
        print("splitting")
        venue_split = int(venue_count * percent)
        venue_training = venue_embedding[:venue_split, :]
        venue_testing = venue_embedding[venue_split:, :]
        author_split = int(author_count * percent)
        author_training = author_embedding[:author_split, :]
        author_testing = author_embedding[author_split:, :]
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
        clf_venue = LogisticRegression(
            random_state=0, solver="lbfgs", multi_class="multinomial"
        ).fit(venue_training, venue_label)
        y_pred_venue = clf_venue.predict(venue_testing)
        clf_author = LogisticRegression(
            random_state=0, solver="lbfgs", multi_class="multinomial"
        ).fit(author_training, author_label)
        y_pred_author = clf_author.predict(author_testing)
        macro_average_venue += f1_score(
            venue_true, y_pred_venue, average="macro"
        )
        micro_average_venue += f1_score(
            venue_true, y_pred_venue, average="micro"
        )
        macro_average_author += f1_score(
            author_true, y_pred_author, average="macro"
        )
        micro_average_author += f1_score(
            author_true, y_pred_author, average="micro"
        )
    print(macro_average_venue / float(experiment_times))
    print(micro_average_venue / float(experiment_times))
    print(macro_average_author / float(experiment_times))
    print(micro_average_author / float(experiment_times))
