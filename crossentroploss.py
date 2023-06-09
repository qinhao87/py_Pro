import torch
import torch.nn as nn

x_input = torch.randn(3, 3)
print('x_input:\n', x_input)

y_target = torch.tensor([1, 2, 0])
print('y_target:\n', y_target)

softmax_func = nn.Softmax(dim=1)

softmax_output = softmax_func(x_input)
print('soft_output:\n', softmax_output)

log_output = torch.log(softmax_output)
print('log_out:\n', log_output)

logsoftmax_func = nn.LogSoftmax(dim=1)
logsoftmax_output = logsoftmax_func(x_input)
print('logsoftmax_output:\n', logsoftmax_output)

nllloss_func = nn.NLLLoss()
nlloss_output = nllloss_func(logsoftmax_output, y_target)
print('nlloss_output:\n', nlloss_output)

crossentroploss = nn.CrossEntropyLoss()
crossentroploss_output = crossentroploss(x_input, y_target)
print('crossentroploss_output:\n', crossentroploss_output)