# mpi-attention Makefile
# =====================
# 契约(不得破坏的 target 名与语义):
#   make            构建全部五个二进制
#   make bench      → bin/bench_mpi    (MPI 计时 harness + 选手 kernel)
#   make omp        → bin/bench_omp    (OpenMP 参照,不计分)
#   make serial     → bin/bench_serial (单核朴素参照,无 MPI/OpenMP,不计分)
#   make ref        → bin/gen_ref      (串行 fp64 参考生成,确定性)
#   make report     → bin/report       (聚合 results.json)
#   make test       小尺寸端到端自测
#   make run        以默认参数跑 scripts/run_benchmark.sh
#   make clean
#
# 选手可以修改本文件(例如追加自己的源文件到 KERNEL_SRC、调整 CFLAGS),
# 但必须保留以上 target 的语义。允许的编译参数自由度见 competition/README.md。
# 非默认路径 OpenMPI:  make CC=/usr/lib64/openmpi/bin/mpicc

CC       := mpicc
CC_SEQ   := gcc
CFLAGS   ?= -O2 -march=native -std=c11 -Wall -Wextra -fno-math-errno
OPENMP   := -fopenmp
LDLIBS   := -lm

INC := -Isrc/common -Isrc/reference -Isrc/harness

COMMON_SRC := src/common/prng.c src/common/data.c src/common/sha256.c \
              src/common/io_util.c src/common/verify.c
COMMON_HDR := src/common/att_types.h src/common/prng.h src/common/data.h \
              src/common/sha256.h src/common/io_util.h src/common/verify.h

# 选手优化面:可自由重写 src/reference/attention_baseline.c、追加自己的
# 源文件(任意位置,在此注册);内部结构完全自定
# (attention.h 的接口签名除外;其余冻结文件见 manifest.frozen_files)
KERNEL_SRC := src/reference/attention_baseline.c
KERNEL_HDR := src/reference/attention.h src/reference/README.md

REF_SRC := src/reference/attention_ref.c src/reference/attention_omp.c
REF_HDR := src/reference/attention_variants.h

.PHONY: all bench omp serial ref report test run clean

all: bench omp serial ref report
bench:  bin/bench_mpi
omp:    bin/bench_omp
serial: bin/bench_serial
ref:    bin/gen_ref
report: bin/report

bin:
	mkdir -p bin

HARNESS_COMMON := src/harness/harness_args.c src/harness/harness_args.h \
                  src/harness/results.c src/harness/results.h

bin/bench_mpi: src/harness/bench_main.c $(HARNESS_COMMON) $(COMMON_SRC) $(COMMON_HDR) $(KERNEL_SRC) $(KERNEL_HDR) | bin
	$(CC) $(CFLAGS) $(OPENMP) $(INC) -o $@ src/harness/bench_main.c $(HARNESS_COMMON) $(COMMON_SRC) $(KERNEL_SRC) $(LDLIBS)

bin/bench_omp: src/harness/omp_main.c $(HARNESS_COMMON) src/reference/attention_omp.c \
               src/reference/attention_variants.h $(COMMON_SRC) $(COMMON_HDR) | bin
	$(CC_SEQ) $(CFLAGS) $(OPENMP) $(INC) -o $@ src/harness/omp_main.c $(HARNESS_COMMON) src/reference/attention_omp.c $(COMMON_SRC) $(LDLIBS)

# 单核朴素参照:纯 gcc,不带 -fopenmp、无 MPI
bin/bench_serial: src/harness/serial_main.c $(HARNESS_COMMON) \
                  src/reference/attention_serial.c src/reference/attention_variants.h \
                  $(COMMON_SRC) $(COMMON_HDR) | bin
	$(CC_SEQ) $(CFLAGS) $(INC) -o $@ src/harness/serial_main.c $(HARNESS_COMMON) src/reference/attention_serial.c $(COMMON_SRC) $(LDLIBS)

# 串行 fp64 参考生成:故意不链 OpenMP / 不用 MPI,保持位级确定
bin/gen_ref: src/harness/ref_main.c src/harness/harness_args.c src/harness/harness_args.h \
             src/reference/attention_ref.c src/reference/attention_variants.h \
             $(COMMON_SRC) $(COMMON_HDR) | bin
	$(CC_SEQ) $(CFLAGS) $(INC) -o $@ src/harness/ref_main.c src/harness/harness_args.c \
	    src/reference/attention_ref.c $(COMMON_SRC) $(LDLIBS)

bin/report: src/harness/report_main.c src/harness/harness_args.c src/harness/harness_args.h \
            $(COMMON_SRC) $(COMMON_HDR) | bin
	$(CC_SEQ) $(CFLAGS) $(INC) -o $@ src/harness/report_main.c src/harness/harness_args.c $(COMMON_SRC) $(LDLIBS)

test: all
	bash scripts/selftest.sh

run: all
	bash scripts/run_benchmark.sh --manifest manifest.json \
	     --ref-dir data/reference --out-dir results

clean:
	rm -rf bin results
