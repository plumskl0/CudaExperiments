memtest_gpu: memtest_gpu.cu
	nvcc memtest_gpu.cu -o memtest_gpu
	./memtest_gpu

memtest_cpu: memtest_cpu.cu
	nvcc memtest_cpu.cu -o memtest_cpu
	./memtest_cpu

memtest_managed: memtest_managed.cu
	nvcc memtest_managed.cu -o memtest_managed
	./memtest_managed
