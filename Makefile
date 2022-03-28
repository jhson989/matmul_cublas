CC = nvcc
PROJECT = cublas_tutorial
PROGRAM = ${PROJECT}.out
MAIN = main.cu
INCS = include/debug.cuh include/kernel.cuh
SRCS = src/debug.cu src/kernel.cu
COMPILE_OPTS = -lcublas
DEBUG=OFF

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${MAIN} ${SRCS} ${INC} Makefile
	${CC} -o $@ ${MAIN} ${SRCS} ${COMPILE_OPTS} -DDEBUG_${DEBUG}


run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}