local dl = require 'dataload._env'
local DataLoader = torch.class('dl.DataLoader', dl)

function DataLoader:index(indices, inputs, targets, ...)
   error"Not Implemented"
end

function DataLoader:sample(batchsize, inputs, targets, ...)
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchsize):random(1,self:size())
   return self:index(self._indices, inputs, targets, ...)
end

function DataLoader:sub(start, stop, inputs, targets, ...)
   self._indices = self._indices or torch.LongTensor()
   self._indices:range(start, stop)
   return self:index(self._indices, inputs, targets, ...)
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

-- called by AsyncIterator before serializing the DataLoader to threads
function DataLoader:reset()
   self._indices = nil
   self._start = nil
end

-- collect garbage every self.gcdelay times this method is called 
function DataLoader:collectgarbage()
   self.gcdelay = self.gcdelay or 200
   self.gccount = (self.gccount or 0) + 1
   if self.gccount >= self.gcdelay then
      collectgarbage()
      self.gccount = 0
   end
end

function DataLoader:clone(...)
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   if select('#',...) > 0 then
      clone:share(self,...)
   end
   return clone
end

-- iterators : subiter, sampleiter

-- subiter : for iterating over validation and test sets
function DataLoader:subiter(batchsize, epochsize, ...)
   batchsize = batchsize or 32
   local dots = {...}
   local size = self:size()
   epochsize = epochsize or -1 
   epochsize = epochsize > 0 and epochsize or self:size()
   self._start = self._start or 1
   local nsampled = 0
   local stop
   
   local inputs, targets
   
   -- build iterator
   return function()
      if nsampled >= epochsize then
         return
      end
      
      local bs = math.min(nsampled+batchsize, epochsize) - nsampled
      stop = math.min(self._start + bs - 1, size)
      -- inputs and targets
      local batch = {self:sub(self._start, stop, inputs, targets, unpack(dots))}
      -- allows reuse of inputs and targets buffers for next iteration
      inputs, targets = batch[1], batch[2]
      
      bs = stop - self._start + 1
      nsampled = nsampled + bs
      self._start = self._start + bs
      if self._start > size then
         self._start = 1
      end
      
      self:collectgarbage()
      
      return nsampled, unpack(batch)
   end
end

-- sampleiter : for iterating over training sets
function DataLoader:sampleiter(batchsize, epochsize, ...)
   batchsize = batchsize or 32
   local dots = {...}
   local size = self:size()
   epochsize = epochsize or -1 
   epochsize = epochsize > 0 and epochsize or self:size()
   local nsampled = 0
   
   local inputs, targets
   
   -- build iterator
   return function()
      if nsampled >= epochsize then
         return
      end
      
      local bs = math.min(nsampled+batchsize, epochsize) - nsampled
      
      -- inputs and targets
      local batch = {self:sample(bs, inputs, targets, unpack(dots))}
      -- allows reuse of inputs and targets buffers for next iteration
      inputs, targets = batch[1], batch[2]
      
      nsampled = nsampled + bs
      
      self:collectgarbage()
      
      return nsampled, unpack(batch)
   end
end
