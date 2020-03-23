//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  The input is biterm_file. The program will run word2vec on the word net.
//  Multi-threads are supported in this version.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <set>
using namespace std;

#define MAX_STRING 200
#define MAX_LABEL 1000

struct Entry
{
	int id;
	double value;
	friend bool operator < (Entry e1, Entry e2)
	{
		return e1.value > e2.value;
	}
};

int id_size = 0, test_size = 0, label_size = 0;
int lb2id[MAX_LABEL];
int pst2id[MAX_LABEL];
Entry ranked_list[MAX_LABEL];
char candidate_file[MAX_STRING], predict_file[MAX_STRING];
set<int> truth[MAX_LABEL], predict[MAX_LABEL];
vector<int> v_nlabels;

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

void TrainModel()
{
	int len, lb, id, tmp;
	char str[MAX_STRING];
	double prob;

	FILE *fi = fopen(candidate_file, "rb");
	while (fscanf(fi, "%d", &len) == 1)
	{
		v_nlabels.push_back(len);
		for (int k = 0; k != len; k++)
		{
			fscanf(fi, "%d", &lb);
			if (lb2id[lb] == 0) lb2id[lb] = ++id_size;
			id = lb2id[lb];
			truth[id].insert(test_size);
		}
		test_size++;
	}
	fclose(fi);

	fi = fopen(predict_file, "rb");
	fscanf(fi, "%s", str);
	while (1)
	{
		ReadWord(str, fi);
		if (strcmp(str, "</s>") == 0) break;

		lb = atoi(str);
		if (lb2id[lb] == 0) lb2id[lb] = ++id_size;
		id = lb2id[lb];
		pst2id[label_size++] = id;
	}
	for (int k = 0; k != test_size; k++)
	{
		fscanf(fi, "%d", &tmp);
		for (int i = 0; i != label_size; i++)
		{
			fscanf(fi, "%lf", &prob);
			id = pst2id[i];
			ranked_list[i].id = id;
			ranked_list[i].value = prob;
		}
		sort(ranked_list, ranked_list + label_size);
		int n = v_nlabels[k];
		for (int i = 0; i != n; i++)
		{
			id = ranked_list[i].id;
			predict[id].insert(k);
		}
	}
	fclose(fi);

	double macro_f1, micro_f1;
	double tp, fn, fp;
	double stp = 0, sfn = 0, sfp = 0, sf1 = 0;
	double P, R;
	set<int>::iterator i;

	for (int k = 1; k <= id_size; k++)
	{
		tp = 0;
		for (i = truth[k].begin(); i != truth[k].end(); i++) if (predict[k].count(*i) != 0)
			tp++;
		fn = truth[k].size() - tp;
		fp = predict[k].size() - tp;

		stp += tp;
		sfn += fn;
		sfp += fp;

		if (tp + fp == 0) P = 0;
		else P = tp / (tp + fp);
		if (tp + fn == 0) R = 0;
		else R = tp / (tp + fn);

		if (P + R != 0) sf1 += 2 * P * R / (P + R);
	}

	macro_f1 = sf1 / id_size;

	P = stp / (stp + sfp);
	R = stp / (stp + sfn);
	micro_f1 = 2 * P * R / (P + R);

	printf("number of tests: %d\n", test_size);
	printf("number of labels: %d\n", id_size);
	printf("macro-f1: %lf\n", macro_f1);
	printf("micro-f1: %lf\n", micro_f1);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-test <file>\n");
		printf("\t\tUse text data from <file> to test the model\n");
		printf("\t-vector <file>\n");
		printf("\t\tUse vector data from <file>\n");
		printf("\nExamples:\n");
		printf("./evl -train train.txt -test test.txt -vector vec.txt \n\n");
		return 0;
	}
	if ((i = ArgPos((char *)"-predict", argc, argv)) > 0) strcpy(predict_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-candidate", argc, argv)) > 0) strcpy(candidate_file, argv[i + 1]);
	TrainModel();
	return 0;
}