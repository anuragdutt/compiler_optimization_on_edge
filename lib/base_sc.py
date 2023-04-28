import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
import time


M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

target = "llvm"
dev = tvm.device(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 100
np_runing_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

print("Numpy running time: %f" % (np_runing_time / np_repeat))

print("TimePreNumpyComputation --"+ str(time.time()))
for i in range(10):
    answer = numpy.dot(a.numpy(), b.numpy())
print("TimePostNumpyComputation --"+ str(time.time()))

time.sleep(20)

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
print("TimePreCallback(10 runs) --"+ str(time.time()))
for i in range(10):
    func(a, b, c)
print("TimePostCallback(10 runs) --"+ str(time.time()))
time.sleep(20)

# tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

# evaluator = func.time_evaluator(func.entry_name, dev, number=1)
# print("Baseline: %f" % evaluator(a, b, c).mean)

