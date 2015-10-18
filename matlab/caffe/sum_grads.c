float *sum_grads;
unsigned int sum_grads_length;

static void do_init_sum_grads(){
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
	
	sum_grads_length=0;
	// Step 1: Find the length from network structure
	for (unsigned int i = 0; i < layers.size(); ++i) {
		vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
		if (layer_blobs.size() == 0) {
			continue;
		}
		for(unsigned int j = 0; j < layer_blobs.size(); ++j){
			sum_grads_length=sum_grads_length+layer_blobs[j]->count();
			//printf("grads[%d][%d]=%u\n",i,j,layer_blobs[j]->count());
		}
	}
	// Step 2: Allocate memory space
	switch(Caffe::mode()){
		case Caffe::CPU:	
			sum_grads=(float*)malloc(sum_grads_length*sizeof(float));
			break;
		case Caffe::GPU:
			cudaMalloc((void**)&sum_grads,sum_grads_length*sizeof(float));
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	}	
	LOG(INFO) << "Memory to store sum of grads: " << (sum_grads_length*sizeof(float))/1024/1024 << " MB";
//	printf("GRADS SPACE IS %u\n",sum_grads_length);
}

//--------API CALL 'zero_sum_grads'--------------------------------------------
//PURPOSE:
//	SET THE SUM OF GRADS TO ZERO
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	-NOTHING-
//-----------------------------------------------------------------------------
static void zero_sum_grads(MEX_ARGS) {
	switch(Caffe::mode()){
		case Caffe::CPU:	
			memset(sum_grads,0x00,sum_grads_length*sizeof(float));
			break;
		case Caffe::GPU:
			cudaMemset(sum_grads,0x00,sum_grads_length*sizeof(float));
			break;
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	}
}


static void do_sum_grads(){
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();

	unsigned int offset=0;
	for (unsigned int i = 0; i < layers.size(); ++i){
		vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
		if (layer_blobs.size() == 0) {
			continue;
		}
		for(unsigned int j = 0; j < layer_blobs.size(); ++j){
			const int local_size=layer_blobs[j]->count();
			switch(Caffe::mode()){
				case Caffe::CPU:	
					caffe_add(local_size,layer_blobs[j]->cpu_diff(),&sum_grads[offset],&sum_grads[offset]); //(num,a,b,c) c=a+b
					break;
				case Caffe::GPU:
					caffe_gpu_add(local_size,layer_blobs[j]->gpu_diff(),&sum_grads[offset],&sum_grads[offset]); //(num,a,b,c) c=a+b
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
			}
			offset+=local_size;
		}
	}
}

//--------API CALL 'get_sum_grads'---------------------------------------------
//PURPOSE:
//	RETURNS THE SUM OF GRADS COMPUTED BY BACKWARD PROPAGATION OF DATA THROUGH
//	NETWORK
//ARGUMENTS:
//	-NOTHING-
//RETURNS:
//	CELL OF STRUCTS WITH THE SAME STRUCTURE AS 'get_weights' BUT THIS TIME
//	IT CONTAINS THE MEAN OF GRADS PRODUCED BY THE SUM OF BACKWARD 
//	PROPAGATIONS
//-----------------------------------------------------------------------------
static mxArray* do_get_sum_grads(){
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
	const vector<string>& layer_names = net_->layer_names();
	// Step 1: count the number of layers with weights
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
	const char* fnames[2] = {"weights", "layer_names"};
	mx_layers = mxCreateStructArray(2, dims, 2, fnames);
	// Step 3: copy weights into output
	prev_layer_name = "";
	int mx_layer_index = 0;
	
	unsigned int offset=0;
	int local_size;
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
			mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
			mxSetField(mx_layers, mx_layer_index, "layer_names", mxCreateString(layer_names[i].c_str()));
			mx_layer_index++;
		}
		for(unsigned int j = 0; j < layer_blobs.size(); ++j){
			// internally data is stored as (width, height, channels, num)
			// where width is the fastest dimension
			mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),layer_blobs[j]->channels(), layer_blobs[j]->num()};
			if(Caffe::mode()==Caffe::CPU){
				mxArray* mx_weights = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
				mxSetCell(mx_layer_cells, j, mx_weights);
				float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));				
				caffe_copy(layer_blobs[j]->count(),&sum_grads[offset],weights_ptr);
			}
			else if(Caffe::mode()==Caffe::GPU){
					mxGPUArray *mx_weights = mxGPUCreateGPUArray(4, dims, mxSINGLE_CLASS, mxREAL,MX_GPU_DO_NOT_INITIALIZE);
					float* weights_ptr = (float *)(mxGPUGetData(mx_weights));				
					local_size=layer_blobs[j]->count();
					caffe_copy(local_size,&sum_grads[offset],weights_ptr);
					offset+=local_size;
					mxSetCell(mx_layer_cells, j, mxGPUCreateMxArrayOnGPU(mx_weights));
					mxGPUDestroyGPUArray(mx_weights);
			}
			else
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
	}
	return mx_layers;
}
static void get_sum_grads(MEX_ARGS) {
	plhs[0] = do_get_sum_grads();
}
