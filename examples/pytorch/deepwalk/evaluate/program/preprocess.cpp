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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	char word[MAX_STRING];
};

char vocab_file[MAX_STRING], vector_file[MAX_STRING], label_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, times;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0;
long long layer1_size;
real *syn0, portion;

std::vector<int> v_set;
std::vector<int> *v_label;

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

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash;
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

void LearnVocabFromTrainFile() {
	printf("Learning vocab...");
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(vocab_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	AddWordToVocab((char *)"</s>");
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		}
		else vocab[i].cn++;
	}
	fclose(fin);
	printf("Done!\n");

	for (int k = 0; k != vocab_size; k++)
		v_set.push_back(k);
}

void ReadData()
{
	char ch, word[MAX_STRING];
	int i, lb;
	float f_num;
	long long T, l;

	printf("Reading vectors...");
	FILE *fi = fopen(vector_file, "rb");
	fscanf(fi, "%lld %lld", &T, &layer1_size);
	syn0 = (real *)malloc(vocab_size*layer1_size*sizeof(real));
	for (long long k = 0; k != T; k++)
	{
		fscanf(fi, "%s", word);
		ch = fgetc(fi);
		i = SearchVocab(word);
		if (i == -1) l = 0;
		else l = i * layer1_size;
		for (int c = 0; c != layer1_size; c++)
		{
			fread(&f_num, sizeof(float), 1, fi);
			syn0[c + l] = (real)f_num;
		}
	}
	fclose(fi);
	printf("Done!\n");

	printf("Reading labels...");
	v_label = new std::vector<int>[vocab_size];
	fi = fopen(label_file, "rb");
	while (fscanf(fi, "%s %d", word, &lb) == 2)
	{
		i = SearchVocab(word);
		if (i == -1) continue;
		v_label[i].push_back(lb);
	}
	fclose(fi);
	printf("Done!\n");

	if (debug_mode>0)
	{
		printf("Vocab size: %lld\n", vocab_size);
		printf("Vector size: %lld\n", layer1_size);
	}
}

void TrainModel() {
	LearnVocabFromTrainFile();
	ReadData();

	char file_name[MAX_STRING];
	FILE *fo;
	for (int T = 0; T != times; T++)
	{
		std::random_shuffle(v_set.begin() + 1, v_set.end());
		int ntrains = (vocab_size - 1) * portion + 1;

		sprintf(file_name, "%strain%d", output_file, T);
		fo = fopen(file_name, "wb");
		for (int k = 1; k <= ntrains; k++)
		{
			int v = v_set[k];
			int len = v_label[v].size();
			long long l = v * layer1_size;
			for (int i = 0; i != len; i++)
			{
				fprintf(fo, "%d", v_label[v][i]);
				for (int c = 0; c != layer1_size; c++)
					fprintf(fo, " %d:%.6f", c + 1, syn0[l + c]);
				fprintf(fo, "\n");
			}
		}
		fclose(fo);

		sprintf(file_name, "%stest%d", output_file, T);
		fo = fopen(file_name, "wb");
		for (int k = ntrains + 1; k != vocab_size; k++)
		{
			int v = v_set[k];
			long long l = v * layer1_size;
			fprintf(fo, "%d", 0);
			for (int c = 0; c != layer1_size; c++)
				fprintf(fo, " %d:%.6f", c + 1, syn0[l + c]);
			fprintf(fo, "\n");
		}
		fclose(fo);

		sprintf(file_name, "%scan%d", output_file, T);
		fo = fopen(file_name, "wb");
		for (int k = ntrains + 1; k != vocab_size; k++)
		{
			int v = v_set[k];
			int len = v_label[v].size();
			fprintf(fo, "%d", len);
			for (int i = 0; i != len; i++)
				fprintf(fo, " %d", v_label[v][i]);
			fprintf(fo, "\n");
		}
		fclose(fo);
	}
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
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-classes <int>\n");
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-samples <int>\n");
		printf("\t\tThe number of training samples will be <int>Million\n");
		printf("\nExamples:\n");
		printf("./btm2vec -train btm.txt -output vec.txt -debug 2 -size 200 -samples 100 -negative 5 -hs 0 -binary 1\n\n");
		return 0;
	}
	output_file[0] = 0;
	if ((i = ArgPos((char *)"-vocab", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-vector", argc, argv)) > 0) strcpy(vector_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-label", argc, argv)) > 0) strcpy(label_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-times", argc, argv)) > 0) times = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-portion", argc, argv)) > 0) portion = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	TrainModel();
	return 0;
}