import torch
from matplotlib import pyplot as plt

float_tensor = torch.randn(100000)

power_range = range(1, -15, -1)
mse_collection = []
l1_collection = []

for pwr in power_range:
    scale, zero_point = 10 ** (pwr), 0
    dtype = torch.qint8
    q_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype)
    rec = q_tensor.dequantize()

    # get mean square error
    mse = torch.nn.functional.mse_loss(float_tensor, rec)
    # get L1 error
    l1 = torch.nn.functional.l1_loss(float_tensor, rec)

    l1_collection.append(l1.item())
    mse_collection.append(mse.item())
    print(pwr, scale, mse, l1)

plt.plot(power_range, mse_collection)
plt.plot(power_range, l1_collection)
plt.show()
