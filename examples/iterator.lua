local dl = require 'dataload'

inputs, targets = torch.range(1,20), torch.range(1,20)
dataloader = dl.TensorLoader(inputs, targets)

for k, inputs, targets in dataloader:sampleiter(inputs, targets) do

end
