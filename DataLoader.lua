local dl = require 'dataload._env'
local DataLoader = torch.class('dl.DataLoader', dl)

function DataLoader:index(indices, inputs, targets)
   error"Not Implemented"
end

function DataLoader:sample(batchsize, inputs, targets)
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchsize):random(1,self:size())
   return self:index(self._indices, inputs, targets)
end

function DataLoader:sub(start, stop, inputs, targets)
   self._indices = self._indices or torch.LongTensor()
   self._indices:range(start, stop)
   return self:index(self._indices, inputs, targets)
end

function DataLoader:shuffle()
   error"Not Implemented"
end

function DataLoader:split(ratio)
   error"Not Implemented"
end

-- number of samples
function DataLoader:size()
   error"Not Implemented"
end

-- size of inputs
function DataLoader:isize(excludedim)
   -- by default, batch dimension is excluded
   excludedim = excludedim == nil and 1 or excludedim
   error"Not Implemented"
end

-- size of targets
function DataLoader:tsize(excludedim)
   -- by default, batch dimension is excluded
   excludedim = excludedim == nil and 1 or excludedim
   error"Not Implemented"
end

