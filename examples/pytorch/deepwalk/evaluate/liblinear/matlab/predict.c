#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int print_null(const char *s,...) {}
int (*info)(const char *fmt,...);

int col_format_flag;

void read_sparse_instance(const mxArray *prhs, int index, struct feature_node *x, int feature_number, double bias)
{
	int i, j, low, high;
	mwIndex *ir, *jc;
	double *samples;

	ir = mxGetIr(prhs);
	jc = mxGetJc(prhs);
	samples = mxGetPr(prhs);

	// each column is one instance
	j = 0;
	low = (int) jc[index], high = (int) jc[index+1];
	for(i=low; i<high && (int) (ir[i])<feature_number; i++)
	{
		x[j].index = (int) ir[i]+1;
		x[j].value = samples[i];
		j++;
	}
	if(bias>=0)
	{
		x[j].index = feature_number+1;
		x[j].value = bias;
		j++;
	}
	x[j].index = -1;
}

static void fake_answer(int nlhs, mxArray *plhs[])
{
	int i;
	for(i=0;i<nlhs;i++)
		plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void do_predict(int nlhs, mxArray *plhs[], const mxArray *prhs[], struct model *model_, const int predict_probability_flag)
{
	int label_vector_row_num, label_vector_col_num;
	int feature_number, testing_instance_number;
	int instance_index;
	double *ptr_label, *ptr_predict_label;
	double *ptr_prob_estimates, *ptr_dec_values, *ptr;
	struct feature_node *x;
	mxArray *pplhs[1]; // instance sparse matrix in row format
	mxArray *tplhs[3]; // temporary storage for plhs[]

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int nr_class=get_nr_class(model_);
	int nr_w;
	double *prob_estimates=NULL;

	if(nr_class==2 && model_->param.solver_type!=MCSVM_CS)
		nr_w=1;
	else
		nr_w=nr_class;

	// prhs[1] = testing instance matrix
	feature_number = get_nr_feature(model_);
	testing_instance_number = (int) mxGetM(prhs[1]);
	if(col_format_flag)
	{
		feature_number = (int) mxGetM(prhs[1]);
		testing_instance_number = (int) mxGetN(prhs[1]);
	}

	label_vector_row_num = (int) mxGetM(prhs[0]);
	label_vector_col_num = (int) mxGetN(prhs[0]);

	if(label_vector_row_num!=testing_instance_number)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		fake_answer(nlhs, plhs);
		return;
	}
	if(label_vector_col_num!=1)
	{
		mexPrintf("label (1st argument) should be a vector (# of column is 1).\n");
		fake_answer(nlhs, plhs);
		return;
	}

	ptr_label    = mxGetPr(prhs[0]);

	// transpose instance matrix
	if(col_format_flag)
		pplhs[0] = (mxArray *)prhs[1];
	else
	{
		mxArray *pprhs[1];
		pprhs[0] = mxDuplicateArray(prhs[1]);
		if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose testing instance matrix\n");
			fake_answer(nlhs, plhs);
			return;
		}
	}


	prob_estimates = Malloc(double, nr_class);

	tplhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
	if(predict_probability_flag)
		tplhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class, mxREAL);
	else
		tplhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_w, mxREAL);

	ptr_predict_label = mxGetPr(tplhs[0]);
	ptr_prob_estimates = mxGetPr(tplhs[2]);
	ptr_dec_values = mxGetPr(tplhs[2]);
	x = Malloc(struct feature_node, feature_number+2);
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double target_label, predict_label;

		target_label = ptr_label[instance_index];

		// prhs[1] and prhs[1]^T are sparse
		read_sparse_instance(pplhs[0], instance_index, x, feature_number, model_->bias);

		if(predict_probability_flag)
		{
			predict_label = predict_probability(model_, x, prob_estimates);
			ptr_predict_label[instance_index] = predict_label;
			for(i=0;i<nr_class;i++)
				ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
		}
		else
		{
			double *dec_values = Malloc(double, nr_class);
			predict_label = predict_values(model_, x, dec_values);
			ptr_predict_label[instance_index] = predict_label;

			for(i=0;i<nr_w;i++)
				ptr_dec_values[instance_index + i * testing_instance_number] = dec_values[i];
			free(dec_values);
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;

		++total;
	}

	if(model_->param.solver_type==L2R_L2LOSS_SVR ||
	   model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
	   model_->param.solver_type==L2R_L2LOSS_SVR_DUAL)
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d)\n", (double) correct/total*100,correct,total);

	// return accuracy, mean squared error, squared correlation coefficient
	tplhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
	ptr = mxGetPr(tplhs[1]);
	ptr[0] = (double)correct/total*100;
	ptr[1] = error/total;
	ptr[2] = ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
				((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt));

	free(x);
	if(prob_estimates != NULL)
		free(prob_estimates);

	switch(nlhs)
	{
		case 3:
			plhs[2] = tplhs[2];
			plhs[1] = tplhs[1];
		case 1:
		case 0:
			plhs[0] = tplhs[0];
	}
}

void exit_with_help()
{
	mexPrintf(
			"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = predict(testing_label_vector, testing_instance_matrix, model, 'liblinear_options','col')\n"
			"       [predicted_label] = predict(testing_label_vector, testing_instance_matrix, model, 'liblinear_options','col')\n"
			"liblinear_options:\n"
			"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
			"-q quiet mode (no outputs)\n"
			"col: if 'col' is setted testing_instance_matrix is parsed in column format, otherwise is in row format\n"
			"Returns:\n"
			"  predicted_label: prediction output vector.\n"
			"  accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.\n"
			"  prob_estimates: If selected, probability estimate vector.\n"
			);
}

void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	int prob_estimate_flag = 0;
	struct model *model_;
	char cmd[CMD_LEN];
	info = &mexPrintf;
	col_format_flag = 0;

	if(nlhs == 2 || nlhs > 3 || nrhs > 5 || nrhs < 3)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}
	if(nrhs == 5)
	{
		mxGetString(prhs[4], cmd, mxGetN(prhs[4])+1);
		if(strcmp(cmd, "col") == 0)
		{
			col_format_flag = 1;
		}
	}

	if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
		mexPrintf("Error: label vector and instance matrix must be double\n");
		fake_answer(nlhs, plhs);
		return;
	}

	if(mxIsStruct(prhs[2]))
	{
		const char *error_msg;

		// parse options
		if(nrhs>=4)
		{
			int i, argc = 1;
			char *argv[CMD_LEN/2];

			// put options in argv[]
			mxGetString(prhs[3], cmd,  mxGetN(prhs[3]) + 1);
			if((argv[argc] = strtok(cmd, " ")) != NULL)
				while((argv[++argc] = strtok(NULL, " ")) != NULL)
					;

			for(i=1;i<argc;i++)
			{
				if(argv[i][0] != '-') break;
				++i;
				if(i>=argc && argv[i-1][1] != 'q')
				{
					exit_with_help();
					fake_answer(nlhs, plhs);
					return;
				}
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
						break;
					case 'q':
						info = &print_null;
						i--;
						break;
					default:
						mexPrintf("unknown option\n");
						exit_with_help();
						fake_answer(nlhs, plhs);
						return;
				}
			}
		}

		model_ = Malloc(struct model, 1);
		error_msg = matlab_matrix_to_model(model_, prhs[2]);
		if(error_msg)
		{
			mexPrintf("Error: can't read model: %s\n", error_msg);
			free_and_destroy_model(&model_);
			fake_answer(nlhs, plhs);
			return;
		}

		if(prob_estimate_flag)
		{
			if(!check_probability_model(model_))
			{
				mexPrintf("probability output is only supported for logistic regression\n");
				prob_estimate_flag=0;
			}
		}

		if(mxIsSparse(prhs[1]))
			do_predict(nlhs, plhs, prhs, model_, prob_estimate_flag);
		else
		{
			mexPrintf("Testing_instance_matrix must be sparse; "
				"use sparse(Testing_instance_matrix) first\n");
			fake_answer(nlhs, plhs);
		}

		// destroy model_
		free_and_destroy_model(&model_);
	}
	else
	{
		mexPrintf("model file should be a struct array\n");
		fake_answer(nlhs, plhs);
	}

	return;
}
