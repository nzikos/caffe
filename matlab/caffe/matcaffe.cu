//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <sstream>
#include <string>
#include <vector>

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs


// Log and throw a Mex error
inline void mex_error(const std::string &msg){
	LOG(ERROR) << msg;
	mexErrMsgTxt(msg.c_str());
}
using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;
static Phase phase;
static char* param_file;
static int init_key = -2;

/*Five things to be aware of:
	1)Caffe uses row-major order
	2)Matlab uses column-major order
	3)Caffe uses BGR color channel order
	4)Matlab uses RGB color channel order
	5)images need to have the data mean subtracted

	Data coming in from matlab needs to be in the order
	[width, height, channels, images]
	where width is the fastest dimension.

	Here is the rough matlab for putting image data into the correct
	format:
		% convert from uint8 to single
		im = single(im);
		% reshape to a fixed size (e.g., 227x227)
		im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
		% permute from RGB to BGR and subtract the data mean (already in BGR)
		im = im(:,:,[3 2 1]) - data_mean;
		% flip width and height to make width the fastest dimension
		im = permute(im, [2 1 3]);

	If you have multiple images, cat them with cat(4, ...)
	The actual forward function. It takes in a cell array of 4-D arrays as
	input and outputs a cell array.
*/

//--------API CALL: 'upload_input'---------------------------------------------
//PURPOSE:
//	UPLOAD DATA TO THE NETWORK
//ARGUMENTS:
//	CELL OF DATA, ONE CELL PER INPUT (e.g. DATA+LABELS = CELL(2,1));
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void do_upload_input(const mxArray* const bottom){
	const vector<Blob<float>*>& input_blobs = net_->input_blobs();
	if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) != input_blobs.size()) {
		mex_error("Invalid input size");
	}
	for (unsigned int i = 0; i < input_blobs.size(); ++i){
		const mxArray* const elem = mxGetCell(bottom, i);
		if (!mxIsSingle(elem)) {
			mex_error("MatCaffe require single-precision float point data");
		}
		if (mxGetNumberOfElements(elem) != input_blobs[i]->count()) {
			std::string error_msg;
			error_msg += "MatCaffe input size does not match the input size ";
			error_msg += "of the network";
			mex_error(error_msg);
		}

		const float* data_ptr = reinterpret_cast<const float*>(mxGetPr(elem));
		switch (Caffe::mode()) {
			case Caffe::CPU:
				caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_cpu_data());
				break;
			case Caffe::GPU:
				caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_gpu_data());
				break;
			default:
				mex_error("Unknown Caffe mode");
		}  // switch (Caffe::mode())
	}
}

static void upload_input(MEX_ARGS){
	if (nrhs != 1) {
		ostringstream error_msg;
		error_msg << "Expected 1 argument, got " << nrhs;
		mex_error(error_msg.str());
	}
	do_upload_input(prhs[0]);	
}
//--------API CALL 'download_output'-------------------------------------------
//PURPOSE:
//	DOWNLOAD DATA FROM THE OUTPUT BLOBS OF THE NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	CELLS OF DATA, ONE CELL PER OUTPUT WITH THE SAME ORDER AS DECLARED IN 
//	THE PROTOTXT FILE
//-----------------------------------------------------------------------------
static mxArray* do_download_output(){
	const vector<Blob<float>*>& output_blobs = net_->output_blobs();

	mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
	for (unsigned int i = 0; i < output_blobs.size(); ++i) {
		// internally data is stored as (width, height, channels, num) where width is the fastest dimension
		mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),output_blobs[i]->channels(), output_blobs[i]->num()};
		mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
		mxSetCell(mx_out, i, mx_blob);
		float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
		caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),data_ptr);
/*		switch (Caffe::mode()) {
			case Caffe::CPU:
				caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),data_ptr);
				break;
			case Caffe::GPU:
				caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),data_ptr);
				break;
			default:
				mex_error("Unknown Caffe mode");
		}*/
	}
	return mx_out;
}
static void download_output(MEX_ARGS){
	if (nrhs != 0) {
		ostringstream error_msg;
		error_msg << "Expected 0 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}
	plhs[0] = do_download_output();
}

//--------API CALL 'get_blob_data'---------------------------------------------
//PURPOSE:
//	Returns values of a specified blob.
//ARGUMENTS:
//	blob id as integer
//RETURNS:
//	Structure with 4D array of "data" of the specified blob and the "name" 
//	of the blob
//-----------------------------------------------------------------------------
static mxArray* do_get_blob_data(int id){
	const vector<string>& blob_names = net_->blob_names();
	const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();

	// Step 2: prepare output array of structures
	mxArray* mx_out;

	const mwSize dims[2] = {1,1};
	const char* fnames[2] = {"data","name"};
	mx_out = mxCreateStructArray(2, dims, 2, fnames);

	mwSize blob_dims[4] = {blobs[id]->width(), blobs[id]->height(), blobs[id]->channels(), blobs[id]->num()};
	mxArray* mx_blob =  mxCreateNumericArray(4, blob_dims, mxSINGLE_CLASS, mxREAL);

	mxSetField(mx_out, 0, "data", mx_blob);
	mxSetField(mx_out, 0, "name", mxCreateString(blob_names[id].c_str()));

	float* out_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
	switch (Caffe::mode()) {
		case Caffe::CPU:
			caffe_copy(blobs[id]->count(), blobs[id]->cpu_data(),out_data_ptr);
			break;
		case Caffe::GPU:
			caffe_copy(blobs[id]->count(), blobs[id]->gpu_data(),out_data_ptr);
			break;
		default:
			LOG(FATAL) << "Unknown Caffe mode.";
	}  
	return mx_out;
}

static void get_blob_data(MEX_ARGS){
	if (nrhs !=1) {
		ostringstream error_msg;
		error_msg << "Expected 1 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}
	int blob_id = static_cast<int>(mxGetScalar(prhs[0]));
	const vector<string>& blob_names = net_->blob_names();
	if(blob_id<=0 || blob_id >blob_names.size()){
		ostringstream error_msg;
		error_msg << "Invalid blob id " << blob_id << ". Bounds are: [1 , " << blob_names.size()-1 << "]";
		mex_error(error_msg.str());
	}
	blob_id--; //make indexing 0-based	
	plhs[0] = do_get_blob_data(blob_id);
}

//--------API CALL 'get_blob_diff'---------------------------------------------
//PURPOSE:
//	Returns deltas of a specified blob.
//ARGUMENTS:
//	blob id as integer
//RETURNS:
//	Structure with 4D array of "diff" of the specified blob and the "name" 
//	of the blob
//-----------------------------------------------------------------------------
static mxArray* do_get_blob_diff(int id){
	const vector<string>& blob_names = net_->blob_names();
	const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();

	// Step 2: prepare output array of structures
	mxArray* mx_out;

	const mwSize dims[2] = {1,1};
	const char* fnames[2] = {"diff","name"};
	mx_out = mxCreateStructArray(2, dims, 2, fnames);

	mwSize blob_dims[4] = {blobs[id]->width(), blobs[id]->height(), blobs[id]->channels(), blobs[id]->num()};
	mxArray* mx_blob =  mxCreateNumericArray(4, blob_dims, mxSINGLE_CLASS, mxREAL);

	mxSetField(mx_out, 0, "diff", mx_blob);
	mxSetField(mx_out, 0, "name", mxCreateString(blob_names[id].c_str()));

	float* out_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
	switch (Caffe::mode()) {
		case Caffe::CPU:
			caffe_copy(blobs[id]->count(), blobs[id]->cpu_diff(),out_data_ptr);
			break;
		case Caffe::GPU:
			caffe_copy(blobs[id]->count(), blobs[id]->gpu_diff(),out_data_ptr);
			break;
		default:
			LOG(FATAL) << "Unknown Caffe mode.";
	}  
	return mx_out;
}

static void get_blob_diff(MEX_ARGS){
	if (nrhs !=1) {
		ostringstream error_msg;
		error_msg << "Expected 1 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}
	int blob_id = static_cast<int>(mxGetScalar(prhs[0]));
	const vector<string>& blob_names = net_->blob_names();
	if(blob_id<=0 || blob_id >blob_names.size()){
		ostringstream error_msg;
		error_msg << "Invalid blob id " << blob_id << ". Bounds are: [1 , " << blob_names.size()-1 << "]";
		mex_error(error_msg.str());
	}
	blob_id--; //make indexing 0-based	
	plhs[0] = do_get_blob_diff(blob_id);
}

//--------API CALL 'get_blobs_number'------------------------------------------
//PURPOSE:
//	RETURNS THE NUMBER OF BLOBS CREATED FOR THE NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	RETURNS THE NUMBER OF BLOBS CREATED FOR THE NETWORK
//-----------------------------------------------------------------------------
static void get_blobs_number(MEX_ARGS){
	if (nrhs !=0) {
		ostringstream error_msg;
		error_msg << "Expected 0 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}
	const vector<string>& blob_names = net_->blob_names();
	plhs[0] = mxCreateDoubleScalar(blob_names.size());
}

//--------API CALL 'forward'---------------------------------------------------
//PURPOSE:
//	PROPAGATES DATA FORWARD THROUGH THE NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void do_forward(){
	net_->ForwardPrefilled();
}
static void forward(MEX_ARGS) {
	if (nrhs != 0) {
		ostringstream error_msg;
		error_msg << "Expected 0 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}
	do_forward();
}



//--------API CALL 'backward'--------------------------------------------------
//PURPOSE:
//	PROPAGATES DATA diffs BACKWARD THROUGH THE NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void do_backward(){
	net_->Backward();
}
static void backward(MEX_ARGS) {
	if (nrhs != 0) {
		ostringstream error_msg;
		error_msg << "Expected 0 argument, got " << nrhs;
		mex_error(error_msg.str());
	}
	do_backward();
}



//--------API CALL 'forward-backward'------------------------------------------
//PURPOSE:
//	1. PROPAGATES DATA FORWARD
//	2. COMPUTES DIFFS FROM GROUND TRUTH
//	3. THEN PROPAGATES BACKWARD THROUGH THE NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void do_forward_backward(){
	do_forward();
	do_backward();
}

static void forward_backward(MEX_ARGS) {
	if (nrhs != 0) {
		ostringstream error_msg;
		error_msg << "Expected 0 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}
	do_forward_backward();
}



//--------API CALL 'get_params'-----------------------------------------------
//PURPOSE:
//	RETURN WEIGHTS & BIAS OF LAYERS
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	CELL OF STRUCTS OF NETWORK WEIGHTS & BIAS
//-----------------------------------------------------------------------------
static mxArray* do_get_params() {
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
	const vector<string>& layer_names = net_->layer_names();
	// Step 1: count the number of layers with params
	int num_layers = 0;

	string prev_layer_name = "";
	for (unsigned int i = 0; i < layers.size(); ++i) {
		vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
		if (layer_blobs.size() == 0) {
			continue;
		}
		if (layer_names[i] != prev_layer_name) {
			prev_layer_name = layer_names[i];
			num_layers++;
		}
	}

	// Step 2: prepare output array of structures
	mxArray* mx_layers;

	const mwSize dims[2] = {num_layers, 1};
	const char* fnames[2] = {"data", "name"};
	mx_layers = mxCreateStructArray(2, dims, 2, fnames);

	// Step 3: copy params into output
	prev_layer_name = "";
	int mx_layer_index = 0;
	for (unsigned int i = 0; i < layers.size(); ++i) {
		vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
		if (layer_blobs.size() == 0) {
			continue;
		}
		mxArray* mx_layer_cells = NULL;
		if (layer_names[i] != prev_layer_name) {
			prev_layer_name = layer_names[i];
			const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
			mx_layer_cells = mxCreateCellArray(2, dims);
			mxSetField(mx_layers, mx_layer_index, "data", mx_layer_cells);
			mxSetField(mx_layers, mx_layer_index, "name",mxCreateString(layer_names[i].c_str()));
			mx_layer_index++;
		}
		for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
			// internally data is stored as (width, height, channels, num)
			// where width is the fastest dimension
			mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),layer_blobs[j]->channels(), layer_blobs[j]->num()};
			if(Caffe::mode()==Caffe::CPU){
				mxArray* mx_params = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
				mxSetCell(mx_layer_cells, j, mx_params);
				float* params_ptr = reinterpret_cast<float*>(mxGetPr(mx_params));				
				caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),params_ptr);
			}
			else if(Caffe::mode()==Caffe::GPU){
				mxGPUArray *mx_params = mxGPUCreateGPUArray(4, dims, mxSINGLE_CLASS, mxREAL,MX_GPU_DO_NOT_INITIALIZE);
				float* params_ptr = (float *)(mxGPUGetData(mx_params));
				caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),params_ptr);
				mxSetCell(mx_layer_cells, j, mxGPUCreateMxArrayOnGPU(mx_params));
				mxGPUDestroyGPUArray(mx_params);
			}
			else
				mex_error("Unknown Caffe mode");
		}
	}
	return mx_layers;
}
static void get_params(MEX_ARGS){
	plhs[0] = do_get_params();
}



//--------API CALL 'do_set_params'--------------------------------------------
//PURPOSE:
//	-SETS WEIGHTS AND BIAS OF THE NETWORK LAYERS
//ARGUMENTS:
//	-CELL OF STRUCTS WITH THE SAME STRUCTURE AS 'get_params' RETURNED 
//	OBJECT
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void do_set_params(const mxArray* mx_layers) {
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
	const vector<string>& layer_names = net_->layer_names();

	unsigned int input_layer_num = mxGetNumberOfElements(mx_layers);

	for (unsigned int i = 0; i < input_layer_num; ++i){

		// Step 1: get input layer information
		mxArray *mx_layer_name = mxGetField(mx_layers, i, "name");
		if (mx_layer_name == NULL){
			mexPrintf("layer %d has no field ""name"", ignore\n", i);
			continue;
		}
		char *layer_name = mxArrayToString(mx_layer_name);
		mxArray *mx_params_cell = mxGetField(mx_layers, i, "data");
		if (mx_params_cell == NULL){
			mexPrintf("layer %d has no field ""data"", ignore\n", i);
			continue;
		}
		if (!mxIsCell(mx_params_cell))
		{
			mexPrintf("layer %d field ""data"" is not cell, ignore\n", i);
			continue;
		}
		unsigned int weight_blob_num = mxGetNumberOfElements(mx_params_cell);

		// Step 2: scan model layers, and try to set layer
		string prev_layer_name = "";
		for (unsigned int j = 0; j < layers.size(); ++j) {
			vector<shared_ptr<Blob<float> > >& layer_blobs = layers[j]->blobs();
			if (layer_blobs.size() == 0)
				continue;
			if (layer_names[j] != string(layer_name))
				continue;
			if (weight_blob_num != layer_blobs.size()){
				mexPrintf("%s has % blobs, while model layer has %d blobs, ignore\n", layer_name, weight_blob_num, layer_blobs.size());
				continue;
			}
			for (unsigned int k = 0; k < layer_blobs.size(); ++k) {
				if(Caffe::mode()==Caffe::CPU){
					mxArray *mx_params = mxGetCell(mx_params_cell, k);
					const mwSize* input_blob_dims = mxGetDimensions(mx_params);
					int dim_num = mxGetNumberOfDimensions(mx_params);
					size_t input_dims[4] = {1, 1, 1, 1};
					for (int idim = 0; idim < dim_num; ++idim)
						input_dims[idim] = input_blob_dims[idim];
					if (layer_blobs[k]->width() != (int)input_dims[0] || layer_blobs[k]->height() != (int)input_dims[1] || layer_blobs[k]->channels() != (int)input_dims[2] || layer_blobs[k]->num() != (int)input_dims[3]){
						if(layer_blobs[k]->count()!=1){
							mexPrintf("%s blobs %d dims don't match, ignore\n", layer_name, k);
							continue;
						}
					}
					float* params_ptr = reinterpret_cast<float*>(mxGetPr(mx_params));					
					caffe_copy(layer_blobs[k]->count(), params_ptr, layer_blobs[k]->mutable_cpu_data());
				}
				else if(Caffe::mode()==Caffe::GPU){
					const mxGPUArray *mx_params 	= mxGPUCreateFromMxArray(mxGetCell(mx_params_cell, k));
					const size_t *input_dims 	= mxGPUGetDimensions(mx_params);
					
					if (layer_blobs[k]->width() != (int)input_dims[0] || layer_blobs[k]->height() != (int)input_dims[1] || layer_blobs[k]->channels() != (int)input_dims[2] || layer_blobs[k]->num() != (int)input_dims[3]){
						if(layer_blobs[k]->count()!=1){
							mexPrintf("%s blobs %d dims don't match, ignore\n", layer_name, k);
							continue;
						}
					}
					float const* params_ptr = (float const*)(mxGPUGetDataReadOnly(mx_params));					
					caffe_copy(layer_blobs[k]->count(), params_ptr, layer_blobs[k]->mutable_gpu_data());
					mxGPUDestroyGPUArray(mx_params);					
				}
				else{
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
				}
			}
		}
		mxFree(layer_name);
	}
}
static void set_params(MEX_ARGS) {
	if (nrhs != 1) {
		LOG(ERROR) << "Only given " << nrhs << " arguments";
		mexErrMsgTxt("caffe_mex : Wrong number of arguments");
	}

	if (net_.use_count() == 0)
	{
		mexPrintf("No solver inited!\n");
		plhs[0] = mxCreateDoubleScalar(-1);
		return;
	}

	do_set_params(prhs[0]);
}



//--------API CALL 'get_gradients'-------------------------------------------------
//PURPOSE:
//	RETURNS gradients COMPUTED BY BACKWARD PROPAGATION OF DATA THROUGH NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	CELL OF STRUCTS WITH THE SAME STRUCTURE AS 'get_params' BUT THIS TIME
//	IT CONTAINS THE gradients PRODUCED BY THE BACKWARD PROPAGATION
//-----------------------------------------------------------------------------
static mxArray* do_get_gradients(){
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
	const vector<string>& layer_names = net_->layer_names();
	// Step 1: count the number of layers with params
	int num_layers = 0;
	string prev_layer_name = "";
	for (unsigned int i = 0; i < layers.size(); ++i) {
		vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
		if (layer_blobs.size() == 0) {
			continue;
		}
		if (layer_names[i] != prev_layer_name) {
			prev_layer_name = layer_names[i];
			num_layers++;
		}
	}
	// Step 2: prepare output array of structures
	mxArray* mx_layers;
	const mwSize dims[2] = {num_layers, 1};
	const char* fnames[2] = {"diff", "name"};
	mx_layers = mxCreateStructArray(2, dims, 2, fnames);
	// Step 3: copy params into output
	prev_layer_name = "";
	int mx_layer_index = 0;
	for (unsigned int i = 0; i < layers.size(); ++i) {
		vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
		if (layer_blobs.size() == 0) {
			continue;
		}
		mxArray* mx_layer_cells = NULL;
		if (layer_names[i] != prev_layer_name) {
			prev_layer_name = layer_names[i];
			const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
			mx_layer_cells = mxCreateCellArray(2, dims);
			mxSetField(mx_layers, mx_layer_index, "diff", mx_layer_cells);
			mxSetField(mx_layers, mx_layer_index, "name", mxCreateString(layer_names[i].c_str()));
			mx_layer_index++;
		}
		for(unsigned int j = 0; j < layer_blobs.size(); ++j){
			// internally data is stored as (width, height, channels, num)
			// where width is the fastest dimension
			mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),layer_blobs[j]->channels(), layer_blobs[j]->num()};
			if(Caffe::mode()==Caffe::CPU){
				mxArray* mx_gradients = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
				mxSetCell(mx_layer_cells, j, mx_gradients);
				float* gradients_ptr = reinterpret_cast<float*>(mxGetPr(mx_gradients));
				caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_diff(),gradients_ptr);
			}
			else if(Caffe::mode()==Caffe::GPU){
				mxGPUArray *mx_gradients = mxGPUCreateGPUArray(4, dims, mxSINGLE_CLASS, mxREAL,MX_GPU_DO_NOT_INITIALIZE);
				float* gradients_ptr = (float *)(mxGPUGetData(mx_gradients));
				caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_diff(),gradients_ptr);
				mxSetCell(mx_layer_cells, j, mxGPUCreateMxArrayOnGPU(mx_gradients));
				mxGPUDestroyGPUArray(mx_gradients);
			}
			else
				mex_error("Unknown Caffe mode");
		}
	}
	return mx_layers;
}
static void get_gradients(MEX_ARGS) {
	if (nrhs != 0) {
		ostringstream error_msg;
		error_msg << "Expected 0 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}
	plhs[0] = do_get_gradients();
}



//--------API CALL 'set_phase'-------------------------------------------------
//PURPOSE:
//	RESETS NETWORK AND CHANGES PHASE (options: train/test)
//ARGUMENTS:
//	String: 'train' or 'test'
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void do_set_phase(){
	net_.reset(new Net<float>(string(param_file), phase));
}
static void set_phase(MEX_ARGS){
	if (nrhs != 1){
		ostringstream error_msg;
		error_msg << "Expected 1 argument, got " << nrhs;
		mex_error(error_msg.str());
	}
	char* phase_name = mxArrayToString(prhs[0]);
	if (strcmp(phase_name, "train") == 0) {
		phase = TRAIN;
	}
	else if (strcmp(phase_name, "test") == 0) {
		phase = TEST;
	} else {
		mex_error("Unknown phase.");
	}
	do_set_phase();
	mxFree(phase_name);
}



//--------API CALL 'set_mode_cpu'----------------------------------------------
//PURPOSE:
//	-SETS CAFFE MODE FOR CPU COMPUTATIONS
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void set_mode_cpu(MEX_ARGS){
	Caffe::set_mode(Caffe::CPU);
}

//--------API CALL 'set_mode_gpu'----------------------------------------------
//PURPOSE:
//	-SETS CAFFE MODE FOR GPU COMPUTATIONS
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void set_mode_gpu(MEX_ARGS) {
	Caffe::set_mode(Caffe::GPU);
}


//--------API CALL 'set_device'------------------------------------------------
//PURPOSE:
//	-SETS WHICH GPU DEVICE WILL BE USED FOR COMPUTATIONS
//ARGUMENTS:
//	-DEVICE_ID
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void set_device(MEX_ARGS) {
	if (nrhs != 1) {
		ostringstream error_msg;
		error_msg << "Expected 1 argument, got " << nrhs;
		mex_error(error_msg.str());
	}
	int device_id = static_cast<int>(mxGetScalar(prhs[0]));
	Caffe::SetDevice(device_id);
}



//--------API CALL 'get_init_key'----------------------------------------------
//PURPOSE:
//	-RETURNS THE KEY THAT IS USED TO IDENTIFY THE NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-INIT_KEY
//-----------------------------------------------------------------------------
static void get_init_key(MEX_ARGS){
	plhs[0] = mxCreateDoubleScalar(init_key);
}



//--------API CALL 'init'------------------------------------------------------
//PURPOSE:
//	-INITIALIZE THE NETWORK, SET LAYERS, I/O BLOBS, CONNECTIONS, PARAMETERS
//ARGUMENTS:
//	-PROTOTXT FILE WITH PARAMS, PHASE
//RETURNS:
//	-INIT_KEY
//-----------------------------------------------------------------------------
static void init(MEX_ARGS) {
	if (nrhs != 2) {
		ostringstream error_msg;
		error_msg << "Expected 2 arguments, got " << nrhs;
		mex_error(error_msg.str());
	}

	mxInitGPU();	
		
	param_file = mxArrayToString(prhs[0]);
	set_phase(0,NULL,1,&prhs[1]);
	init_key = random();  // NOLINT(caffe/random_fn)
	if (nlhs == 1) {
		plhs[0] = mxCreateDoubleScalar(init_key);
	}
}



//--------API CALL 'reset'-----------------------------------------------------
//PURPOSE:
//	-RESETS THE NETWORK IF ONE EXISTS. DEALLOCATE MEMORY RESOURCES
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void reset(MEX_ARGS) {
	if (net_) {
		net_.reset();
		init_key = -2;
		LOG(INFO) << "Network reset, call init before use it again";
	}
}

//--------API CALL 'is_initialized'--------------------------------------------
//PURPOSE:
//	-CHECKS IF AN INITIALIZED NETWORK ALREADY EXISTS
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-0: NETWORK DOES NOT EXIST
//	-1: NETWORK EXISTS
//-----------------------------------------------------------------------------
static void is_initialized(MEX_ARGS) {
	if (!net_) {
		plhs[0] = mxCreateDoubleScalar(0);
	}
	else{
		plhs[0] = mxCreateDoubleScalar(1);
	}
}

//--------API CALL: 'training_iter'--------------------------------------------
//PURPOSE:
//	PERFORM ONE TRAINING ITERATION:
//		UPLOAD INPUTS -> FORWARD -> BACKWARD -> GET PRODUCED gradients
//ARGUMENTS:
//	CELL OF DATA, ONE CELL PER INPUT (e.g. DATA+LABELS = CELL(2,1));
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void training_iter(MEX_ARGS){
	if (nrhs != 1) {
		ostringstream error_msg;
		error_msg << "Expected 1 argument, got " << nrhs;
		mex_error(error_msg.str());
	}
	do_upload_input(prhs[0]);
	do_forward_backward();
	plhs[0] = do_get_gradients();
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
	string cmd;
	void (*func)(MEX_ARGS);
};
static handler_registry handlers[] = {
	// Public API functions
	{	"forward",		forward		},
	{	"backward",		backward	},
	{	"forward_backward",	forward_backward},
	{	"training_iter",	training_iter	},
	{	"init",			init            },
	{	"is_initialized",	is_initialized  },
	{	"set_mode_cpu",		set_mode_cpu    },
	{	"set_mode_gpu",		set_mode_gpu    },
	{	"set_phase",		set_phase	},
	{	"set_device",		set_device      },
	{	"get_params",		get_params      },
	{	"set_params",		set_params	},
	{	"upload_input",		upload_input	},
	{	"download_output",	download_output },
	{	"get_gradients",	get_gradients	},
	{	"get_blobs_number",	get_blobs_number},
	{	"get_blob_data",	get_blob_data	},
	{	"get_blob_diff",	get_blob_diff	},
	{	"get_init_key",		get_init_key    },
	{	"reset",		reset           },
	// The end.
	{	"END",			NULL		},
};

/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
	mexLock();  // Avoid clearing the mex file.
	if (nrhs == 0) {
		mex_error("No API command given");
		return;
	}
	{
		// Handle input command
		char *cmd = mxArrayToString(prhs[0]);
		bool dispatched = false;
		// Dispatch to cmd handler
		for (int i = 0; handlers[i].func != NULL; i++) {
			if (handlers[i].cmd.compare(cmd) == 0) {
				handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
				dispatched = true;
				break;
			}
		}
		if (!dispatched) {
			ostringstream error_msg;
			error_msg << "Unknown command '" << cmd << "'";
			mex_error(error_msg.str());
		}
		mxFree(cmd);
	}
}