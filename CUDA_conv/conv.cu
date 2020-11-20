#include <cudnn.h>
#include <iostream>

#define checkCUDNN(expression)  \
{                               \
	cudnnStatus_t status = (expression); \
  if (status != CUDNN_STATUS_SUCCESS) { \
    std::cerr << "Error on line " << __LINE__ << ": " \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
}

int main(int argc, char const *argv[]) {
  cudnnHandle_t cudnn; // serve as a context object
  checkCUDNN(cudnnCreate(&cudnn));

  const int height = 380;
  const int width = 380;

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
          /*format*/CUDNN_TENSOR_NHWC,
          /*dataType*/CUDNN_DATA_FLOAT,
          /*batch_size*/1,
          /*channels*/3,
          /*image_height*/height,
          /*image_width*/width));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
          /*format*/CUDNN_TENSOR_NHWC,
          /*dataType*/CUDNN_DATA_FLOAT,
          /*batch_size*/1,
          /*channels*/3,
          /*image_height*/height,
          /*image_width*/width));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
          /*dataType*/CUDNN_DATA_FLOAT,
          /*format*/CUDNN_TENSOR_NCHW,
          /*out_channels*/3, 
          /*in_channels*/3,
          /*kernel_height*/3,
          /*kernel_width*/3));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNN_DATA_FLOAT));
  
  cudnnConvolutionFwdAlgoPerf_t perf;
  int algo_cnt;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
      requestedAlgoCount(cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            10, /*requestedAlgoCount*/
            &algo_cnt,
            &perf));

  convolution_algorithm = perf.algo;
  
  // In memory constrained environments, we may prefer CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
  
  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        output_descriptor,
        convolution_algorithm,
        &workspace_bytes));

  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
  
  // allocate memory
  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  int image_bytes = 1 * 3 * height * width * sizeof(float);

  float array[1][height][width][3];
  float* d_input{nullptr};
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, array, image_bytes, cudaMemcpyHostToDevice);

  float* d_output{nullptr};
  cudaMalloc(&d_output, image_bytes);
  cudaMemset(d_output, 0, image_bytes);  

  const float kernel_template[3][3] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
  };

  float h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel <3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  float* d_kernel{nullptr};
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);


  float time;
  cudaEvent_t start, stop;

  checkCUDNN( cudaEventCreate(&start) );
  checkCUDNN( cudaEventCreate(&stop) );
  checkCUDNN( cudaEventRecord(start, 0) );

  int loop_times = 10000;
  for (int i = 0; i < loop_times; i++) {
    const float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor,
                                    d_input,
                                    kernel_descriptor,
                                    d_kernel,
                                    convolution_descriptor,
                                    convolution_algorithm,
                                    d_workspace,
                                    workspace_bytes,
                                    &beta,
                                    output_descriptor,
                                    d_output));
    float* h_output = new float[image_bytes];
  }

  checkCUDNN( cudaEventRecord(stop, 0) );
  checkCUDNN( cudaEventSynchronize(stop) );
  checkCUDNN( cudaEventElapsedTime(&time, start, stop) );

  std::cout << "Run " << loop_times << " convolutions, run " << time << " ms";
  cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
  // Do something with h_output ...

  delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}
