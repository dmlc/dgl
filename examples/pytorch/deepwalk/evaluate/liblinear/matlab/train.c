#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../linear.h"

#include "mex.h"
#include "linear_model_matlab.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}
void print_string_matlab(const char *s) {mexPrintf(s);}

void exit_with_help()
{
	mexPrintf(
	"Usage: model = train(training_label_vector, training_instance_matrix, 'liblinear_options', 'col');\n"
	"liblinear_options:\n"
	"-s type : set type of solver (default 1)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
	"	-s 1, 3, 4 and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"col:\n"
	"	if 'col' is setted, training_instance_matrix is parsed in column format, otherwise is in row format\n"
	);
}

// liblinear arguments
struct parameter param;		// set by parse_command_line
struct problem prob;		// set by read_problem
struct model *model_;
struct feature_node *x_space;
int cross_validation_flag;
int col_format_flag;
int nr_fold;
double bias;

double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);
	double retval = 0.0;

	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR || 
	   param.solver_type == L2R_L1LOSS_SVR_DUAL || 
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob.l;i++)
                {
                        double y = prob.y[i];
                        double v = target[i];
                        total_error += (v-y)*(v-y);
                        sumv += v;
                        sumy += y;
                        sumvv += v*v;
                        sumyy += y*y;
                        sumvy += v*y;
                }
                printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
                printf("Cross Validation Squared correlation coefficient = %g\n",
                        ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
                        ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
                        );
		retval = total_error/prob.l;
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
		retval = 100.0*total_correct/prob.l;
	}

	free(target);
	return retval;
}

// nrhs should be 3
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
	int i, argc = 1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];
	void (*print_func)(const char *) = print_string_matlab;	// default printing to matlab display

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation_flag = 0;
	col_format_flag = 0;
	bias = -1;


	if(nrhs <= 1)
		return 1;

	if(nrhs == 4)
	{
		mxGetString(prhs[3], cmd, mxGetN(prhs[3])+1);
		if(strcmp(cmd, "col") == 0)
			col_format_flag = 1;
	}

	// put options in argv[]
	if(nrhs > 2)
	{
		mxGetString(prhs[2], cmd,  mxGetN(prhs[2]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q') // since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'v':
				cross_validation_flag = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					mexPrintf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			default:
				mexPrintf("unknown option\n");
				return 1;
		}
	}

	set_print_string_function(print_func);

	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
			case L2R_LR: 
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL: 
			case L2R_L1LOSS_SVC_DUAL: 
			case MCSVM_CS: 
			case L2R_LR_DUAL: 
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC: 
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
	return 0;
}

static void fake_answer(int nlhs, mxArray *plhs[])
{
	int i;
	for(i=0;i<nlhs;i++)
		plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int read_problem_sparse(const mxArray *label_vec, const mxArray *instance_mat)
{
	int i, j, k, low, high;
	mwIndex *ir, *jc;
	int elements, max_index, num_samples, label_vector_row_num;
	double *samples, *labels;
	mxArray *instance_mat_col; // instance sparse matrix in column format

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	if(col_format_flag)
		instance_mat_col = (mxArray *)instance_mat;
	else
	{
		// transpose instance matrix
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// the number of instance
	prob.l = (int) mxGetN(instance_mat_col);
	label_vector_row_num = (int) mxGetM(label_vec);

	if(label_vector_row_num!=prob.l)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		return -1;
	}
	
	// each column is one instance
	labels = mxGetPr(label_vec);
	samples = mxGetPr(instance_mat_col);
	ir = mxGetIr(instance_mat_col);
	jc = mxGetJc(instance_mat_col);

	num_samples = (int) mxGetNzmax(instance_mat_col);

	elements = num_samples + prob.l*2;
	max_index = (int) mxGetM(instance_mat_col);

	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct feature_node*, prob.l);
	x_space = Malloc(struct feature_node, elements);

	prob.bias=bias;

	j = 0;
	for(i=0;i<prob.l;i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];
		low = (int) jc[i], high = (int) jc[i+1];
		for(k=low;k<high;k++)
		{
			x_space[j].index = (int) ir[k]+1;
			x_space[j].value = samples[k];
			j++;
	 	}
		if(prob.bias>=0)
		{
			x_space[j].index = max_index+1;
			x_space[j].value = prob.bias;
			j++;
		}
		x_space[j++].index = -1;
	}

	if(prob.bias>=0)
		prob.n = max_index+1;
	else
		prob.n = max_index;

	return 0;
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	if(nlhs > 1)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}

	// Transform the input Matrix to libsvm format
	if(nrhs > 1 && nrhs < 5)
	{
		int err=0;

		if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
			mexPrintf("Error: label vector and instance matrix must be double\n");
			fake_answer(nlhs, plhs);
			return;
		}

		if(parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			destroy_param(&param);
			fake_answer(nlhs, plhs);
			return;
		}

		if(mxIsSparse(prhs[1]))
			err = read_problem_sparse(prhs[0], prhs[1]);
		else
		{
			mexPrintf("Training_instance_matrix must be sparse; "
				"use sparse(Training_instance_matrix) first\n");
			destroy_param(&param);
			fake_answer(nlhs, plhs);
			return;
		}

		// train's original code
		error_msg = check_parameter(&prob, &param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				mexPrintf("Error: %s\n", error_msg);
			destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
			fake_answer(nlhs, plhs);
			return;
		}

		if(cross_validation_flag)
		{
			double *ptr;
			plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			ptr = mxGetPr(plhs[0]);
			ptr[0] = do_cross_validation();
		}
		else
		{
			const char *error_msg;

			model_ = train(&prob, &param);
			error_msg = model_to_matlab_structure(plhs, model_);
			if(error_msg)
				mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
			free_and_destroy_model(&model_);
		}
		destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
	}
	else
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}
}
