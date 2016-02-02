local dl = require 'dataload._env'
local DataLoader = torch.class('dl.DataLoader', dl)

function DataLoader:index(indices, inputs, targets)
   error"Not Implemented"
end

function DataLoader:sample(batchsize, inputs, targets)
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchsize):random(1,self:nSample())
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

function DataLoader:size()
   error"Not Implemented"
end

function DataLoader:inputSize()
   error"Not Implemented"
end

function DataLoader:targetSize()
   error"Not Implemented"
end

