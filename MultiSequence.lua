local dl = require 'dataload._env'
local MultiSequence, parent = torch.class('dl.MultiSequence', 'dl.DataLoader', dl)

-- used by Billion Words dataset to encapsulate unordered sentences.
-- The inputs and targets for sequence of sequences look as follows:
-- target : [ ] E L L O [ ] C R E E N ...
-- input  : [ ] H E L L [ ] S C R E E ...
-- Note that [ ] is a zero mask used to forget between sequences.
function MultiSequence:__init(sequences, batchsize)
   assert(torch.isTensor(sequences[1]))
   assert(torch.type(batchsize) == 'number')
   -- sequence is a list of tensors where the first dimension indexes time
   self.sequences = sequences
   
   self.batchsize = batchsize
   
   self.seqlen = 0
   for i, seq in ipairs(self.sequences) do
      self.seqlen = self.seqlen + seq:size(1)
   end
   
   self.seqlen = torch.ceil(self.seqlen/batchsize)
   
   self:reset()
end

function MultiSequence:reset()
   parent.reset(self)   
   self.trackers = {nextseq=1}
end

-- inputs : seqlen x batchsize [x inputsize]
-- targets : seqlen x batchsize [x inputsize]
function MultiSequence:sub(start, stop, inputs, targets)
   local seqlen = stop - start + 1
   
   inputs = inputs or self.sequences[1].new()
   inputs:resize(seqlen, self.batchsize, unpack(self:isize())):zero()
   
   targets = targets or inputs.new()
   targets:resize(seqlen, self.batchsize, unpack(self:tsize())):zero()
   
   for i=1,self.batchsize do
   
      local input = inputs:select(2,i)
      local target = targets:select(2,i)
      
      local tracker = self.trackers[i] or {}
      self.trackers[i] = tracker
      
      local start = 1
      while start <= seqlen do
      
         if not tracker.seqid then
            tracker.idx = 1
            
            -- each sequence is separated by a zero input and -1 target. 
            -- this should make the model forget between sequences
            -- (use with AbstractRecurrent:maskZero() and LookupTableMaskZero)
            if input:dim() == 1 then
               input[start] = 0
               target[start] = 1
            else
               input[start]:fill(0)
               target[start]:fill(0)
            end
            
            start = start + 1
            
            if self.randseq then 
               tracker.seqid = math.random(1,#self.sequences)
            else
               tracker.seqid = self.trackers.nextseq
               self.trackers.nextseq = self.trackers.nextseq + 1
               if self.trackers.nextseq > #self.sequences then
                  self.trackers.nextseq = 1
               end
            end

         end
         
         if start <= seqlen then
            local sequence = self.sequences[tracker.seqid]
            
            local stop = math.min(tracker.idx+seqlen-start, sequence:size(1) - 1) 
            local size = stop - tracker.idx + 1
            
            input:narrow(1,start,size):copy(sequence:sub(tracker.idx, stop))
            target:narrow(1,start,size):copy(sequence:sub(tracker.idx+1, stop+1))
            
            start = start + size
            tracker.idx = stop+1
            
            if stop == sequence:size(1) - 1 then
               tracker.seqid = nil
            end
         
         end
         
      end
        
      assert(start-1 == seqlen)
      
   end
   
   return inputs, targets 
end

function MultiSequence:sample()
   error"Not Implemented"
end

-- returns size of sequences
function MultiSequence:size()
   return self.seqlen
end

function MultiSequence:isize(excludedim)
   -- by default, sequence dimension is excluded
   excludedim = excludedim == nil and 1 or excludedim
   local size = torchx.recursiveSize(self.sequences[1], excludedim)
   if excludedim ~= 1 then
      size[1] = self:size()
   end
   return size
end

function MultiSequence:tsize(excludedim)
   return self:isize(excludedim)
end

function MultiSequence:subiter(seqlen, epochsize, ...)
   return parent.subiter(self, seqlen, epochsize, ...)
end

function MultiSequence:sampleiter(seqlen, epochsize, ...)
   error"Not Implemented. Use subiter instead."
end

