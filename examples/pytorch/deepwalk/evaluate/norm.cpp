#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#define MAX_STRING 100

typedef float real;                    // Precision of float numbers

char input_file[MAX_STRING], output_file[MAX_STRING];
int binary = 0;
int save_form = 0; // 0 for mine, 1 for line;

void Normalize()
{
    printf("Normalizing Binary: %d\n", binary);

    long long num_vertices, vector_dim, a, b;
    real *vec;
	double len;
    int vid;
	FILE *fi, *fo;

	fi = fopen(input_file, "rb");
	fo = fopen(output_file, "wb");
    if (fi == NULL || fo == NULL) {
        printf("Normalizing Error: embedding file doesn't exit\n");
        exit(1);
    } 

	fscanf(fi, "%lld %lld", &num_vertices, &vector_dim);
	fprintf(fo, "%lld %lld\n", num_vertices, vector_dim);

	vec = (real *)malloc(vector_dim * sizeof(real));

    long long N = num_vertices;
    for (a = 0; a < N; a++)
	{
		fscanf(fi, "\n%d", &vid);
		for (b = 0; b < vector_dim; b++) fscanf(fi, " %f", &vec[b]);
		len = 0;
		for (b = 0; b < vector_dim; b++) len += vec[b] * vec[b];
		len = sqrt(len);
		for (b = 0; b < vector_dim; b++) vec[b] /= len;

		fprintf(fo, "%d ", vid);
		if (binary == 1)
		{
			for (b = 0; b < vector_dim; b++)
				fwrite(&vec[b], sizeof(real), 1, fo);
		}
		else
		{
			for (b = 0; b < vector_dim; b++)
				fprintf(fo, "%lf ", vec[b]);
		}
		fprintf(fo, "\n");
	}
	free(vec);
	fclose(fi);
	fclose(fo);
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
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    Normalize();
	return 0;
}
