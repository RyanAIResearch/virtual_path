from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import resource
import os
# from pytorch_memlab import MemReporter

# resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# torch.utils.checkpoint
# torch.utils.checkpoint.get_device_states()
# with torch.no_grad():
model = models.resnet18().cuda()
inputs = torch.randn(32, 3, 224, 224).cuda()
outputs = model(inputs)

# with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], profile_memory=True, with_modules=True, use_cuda=True) as prof:
#     outputs = model(inputs)
#     loss = F.cross_entropy(outputs, torch.tensor([0, 3, 6, 10, 100]).cuda())
#     loss.backward()

# reporter = MemReporter(model)
# outputs = model(inputs)
# print(torch.cuda.memory_summary())
# print(torch.cuda.memory_snapshot())
# print('----- before -----\n')
# print(torch.cuda.max_memory_allocated())
# print(torch.cuda.memory_stats()['allocated_bytes.all.current'])
# reporter.report()
# loss = F.cross_entropy(outputs, torch.tensor([0, 3, 6, 10, 100]).cuda())
# loss.backward()
# print('----- after -----\n')
# reporter.report()
print(torch.cuda.max_memory_allocated() / (1024 * 1024))
print(torch.cuda.memory_allocated() / (1024 * 1024))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
print(os.system('free --mega'))

loss = F.cross_entropy(outputs, torch.zeros(32).long().cuda())
loss.backward()
print(torch.cuda.max_memory_allocated() / (1024 * 1024))
print(torch.cuda.memory_allocated() / (1024 * 1024))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
print(os.system('free --mega'))
print()
# print(prof.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=10))
