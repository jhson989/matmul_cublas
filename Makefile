CC = nvcc
PROJECT = cublas_tutorial
PROGRAM = ${PROJECT}.out
MAIN = main.cu
INCS = include/debug.cuh include/kernel.cuh
SRCS = src/debug.cu src/kernel.cu
COMPILE_OPTS = -lcublas

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${MAIN} ${SRCS} ${INC} Makefile
	${CC} -o $@ ${MAIN} ${SRCS} ${COMPILE_OPTS}


run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}