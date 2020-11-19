CXX := /usr/local/cuda-10.2/bin/nvcc
TARGET := conv
CUDNN_PATH := /usr/
HEADERS := -I /usr/include
LIBS := -L/usr/lib/x86_64-linux-gnu/  -L/usr/local/lib
CXXFLAGS := -arch=sm_35 -std=c++11 -O2

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcudnn

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
