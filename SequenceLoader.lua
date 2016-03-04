local dl = require 'dataload._env'
local SequenceLoader = torch.class('dl.SequenceLoader', 'dl.DataLoader', dl)

function SequenceLoader:__init(data, batchsize, bidirectional)
   assert(torch.isTensor(data))
   assert(torch.type(batchsize) == 'number')
   -- data is a tensor where the first dimension indexes the sequence
   
   self.batchsize = batchsize
   self.bidirectional = bidirectional 
   
   local seqlen = data:size(1)
   local size = data:size():totable()
   table.remove(size, 1)
   assert(#size == data:dim() - 1)
   
   self.data = data.new()
   -- note that some data will be lost
   local seqlen2 = torch.floor(seqlen / batchsize)
   self.data = data:sub(1,seqlen2*batchsize):view(seqlen2, batchsize):contiguous()
end

-- inputs : seqlen x batchsize [x inputsize]
-- targets : seqlen x batchsize [x inputsize]
function SequenceLoader:sub(start, stop, inputs, targets)
   local seqlen = stop - start + 1
   
   inputs = inputs or self.data.new()
   targets = targets or inputs.new()
   
   if self.bidirectional then
      assert(stop <= self.data:size(1))
      inputs:set(self.data:sub(start, stop))
      targets:set(inputs)
   else
      assert(stop < self.data:size(1))
      inputs:set(self.data:sub(start, stop))
      targets:set(self.data:sub(start+1, stop+1))
   end
   
   return inputs, targets 
end

function SequenceLoader:sample()
   error"Not Implemented"
end

-- returns size of sequences
function SequenceLoader:size()
   if self.bidirectional then
      return self.data:size(1)
   else
      return self.data:size(1)-1
   end
end

function SequenceLoader:isize(excludedim)
   -- by default, sequence dimension is excluded
   excludedim = excludedim == nil and 1 or excludedim
   local size = torchx.recursiveSize(self.inputs, excludedim)
   if excludedim ~= 1 then
      size[1] = self:size()
   end
   return size
end

function SequenceLoader:tsize(excludedim)
   return self:isize(excludedim)
end

