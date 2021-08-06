
import numpy as np
import cv2 as cv
import time

rand = np.random.random((1024, 1024)).astype(np.float32)
h_array1 = np.stack([rand, rand],axis=2)
h_array2 = h_array1
d_array1 = cv.cuda_GpuMat()
d_array2 = cv.cuda_GpuMat()
d_array1.upload(h_array1)
d_array2.upload(h_array2)

start = time.time()
cv.cuda.gemm(d_array1, d_array2, 1, None, 0, None, 1)
end = time.time()
print("Time elapsed:", end - start, "sec")