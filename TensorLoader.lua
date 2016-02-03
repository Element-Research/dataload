local dl = require 'dataload._env'
local TensorLoader = torch.class('dl.TensorLoader', 'dl.DataLoader', dl)

function TensorLoader:__init(inputs, targets)
   self.inputs = inputs
   self.targets = targets
   
   assert(torchx.recursiveBatchSize(self.inputs) == torchx.recursiveBatchSize(self.targets))
end

function TensorLoader:index(indices, inputs, targets)   
   inputs = torchx.recursiveIndex(inputs, self.inputs, 1, indices)
   targets = torchx.recursiveIndex(targets, self.targets, 1, indices)
   return inputs, targets
end

function TensorLoader:shuffle()
   local indices = torch.LongTensor():randperm(self:size())
   self.inputs = torchx.recursiveIndex(nil, self.inputs, 1, indices)
   self.targets = torchx.recursiveIndex(nil, self.targets, 1, indices)
   return self, indices
end

function TensorLoader:split(ratio)
   assert(ratio > 0 and ratio < 1, "Expecting 0 < arg < 1")
   
   local size = self:size()
   local sizeA = math.floor(size*ratio)
   
   local loaders = {}
   for i,split in ipairs{{1,sizeA},{sizeA+1,size}} do
      local start, stop = unpack(split)
      local inputs = torchx.recursiveSub(self.inputs, start, stop)
      local targets = torchx.recursiveSub(self.targets, start, stop)
      local loader = dl.TensorLoader(inputs, targets)
      assert(loader:size() == stop - start + 1)
      loaders[i] = loader
   end
   return unpack(loaders)
end

function TensorLoader:size()
   return torchx.recursiveBatchSize(self.inputs)
end

function TensorLoader:isize(excludedim)
   -- by default, batch dimension is excluded
   excludedim = excludedim == nil and 1 or excludedim
   return torchx.recursiveSize(self.inputs, excludedim)
end

function TensorLoader:tsize(excludedim)
   -- by default, batch dimension is excluded
   excludedim = excludedim == nil and 1 or excludedim
   return torchx.recursiveSize(self.targets, excludedim)
end

