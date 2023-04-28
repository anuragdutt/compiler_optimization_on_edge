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

answer = numpy.dot(a.numpy(), b.numpy())

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

bn = 32
kfactor = 4
s = te.create_schedule(C.op)

# Blocking by loop tiling
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

# Hoist reduction domain outside the blocking loop
s[C].reorder(mo, no, ko, ki, mi, ni)

# func = tvm.build(s, [A, B, C], target=target, name="mmult")
# assert func

#c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
#func(a, b, c)
#tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

# By simply tiling the loop 32x32, and hoisting ko, ki outside the blocking loops,
# we can see big speedup compared with the baseline.
#evaluator = func.time_evaluator(func.entry_name, dev, number=10)
#$print("Opt1: %f" % evaluator(a, b, c).mean)

packedB = te.compute(
    (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
)
C = te.compute(
    (M, N),
    lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, mi, ki, ni)
s[C].vectorize(ni)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
# assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)

time.sleep(20)
print("TimePreCallback(100 runs) --"+ str(time.time()))
for i in range(100):
    func(a, b, c)
print("TimePostCallback(100 runs) --"+ str(time.time()))
time.sleep(20)
#tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

#evaluator = func.time_evaluator(func.entry_name, dev, number=10)
#print("Opt4: %f" % evaluator(a, b, c).mean)
