------------------------------------------------------------------------
--[[ MultiImageSequence ]]--
-- input : a sequence of images
-- targets : a sequence of images
-- Example dataset : 
-- input is a sequence of video frames where targets are binary masks
--
-- Directory is organized as :
-- [datapath]/[seqid]/[input|target][1,2,3,...,T].jpg
-- So basically, the datapath contains a folder for each sequence.
-- Each sequence has T input and target images where T can vary.
-- The target of input[t].jpg is target[t].jpg.
--
-- The return inputs and targets will be separated by mask tokens ([ ]):
-- [ ] target11, target12, target13, ..., target1T [ ] target21, ...
-- [ ] input11,  input12,  input13,  ..., input1T  [ ] input21, ...
--
-- The mask tokens [ ] represent images with nothing but zeros.
-- For large datasets use Lua5.2 instead of LuaJIT to avoid mem errors.
------------------------------------------------------------------------
local dl = require 'dataload._env'
local MultiImageSequence, parent = torch.class('dl.MultiImageSequence', 'dl.DataLoader', dl)

function MultiImageSequence:__init(datapath, batchsize, loadsize, samplesize, samplefunc, verbose)
   -- 1. post-init arguments
   
   -- samples a random uniform crop location every time-step (instead of once per sequence)
   self.cropeverystep = false
   -- random-uniformly samples a loadsize between samplesize and loadsize (this effectively scales the croped location)
   self.varyloadsize = false
   -- varies load size every step instead of once per sequence
   self.scaleeverystep = false
   -- each new sequence is chosen random uniformly
   self.randseq = false
   
   -- modify this to use different patterns for input and target files
   self.inputpattern = 'input%d.jpg'
   self.targetpattern = 'target%d.jpg'

   -- 2. arguments
   
   -- path containing a folder for each sequence
   self.datapath = datapath
   assert(torch.type(self.datapath) == 'string')
   
   -- number of sequences per batch
   self.batchsize = batchsize
   assert(torch.type(batchsize) == 'number')
   
   -- size to load the images to, initially
   self.loadsize = loadsize
   assert(torch.type(self.loadsize) == 'table')
   assert(torch.type(self.loadsize[1]) == 'table', 'Missing inputs loadsize')
   assert(torch.type(self.loadsize[2]) == 'table', 'Missing targets loadsize')
   
   -- consistent sample size to resize the images. 
   local inputsamplesize = samplesize or self.loadsize
   assert(torch.type(inputsamplesize) == 'table')
   assert(torch.type(inputsamplesize[1]) == 'number', 'Provide samplesize for inputs only (target samplesize will be proportional)')
   assert(inputsamplesize[2] <= self.loadsize[1][2])
   assert(inputsamplesize[3] <= self.loadsize[1][3])
   -- make target samplesize proportional to input samplesize w.r.t. loadsize
   local h = math.min(self.loadsize[2][2], math.max(1, torch.round(inputsamplesize[2]*self.loadsize[2][2]/self.loadsize[1][2])))
   local w = math.min(self.loadsize[2][2], math.max(1, torch.round(inputsamplesize[3]*self.loadsize[2][3]/self.loadsize[1][3])))
   local targetsamplesize = {self.loadsize[2][1], h, w}
   self.samplesize = {inputsamplesize, targetsamplesize}
   
   -- function f(self, dst, path) used to create a sample(s) from 
   -- an image path. Stores them in dst. Strings "sampleDefault"
   -- "sampleTrain" or "sampleTest" can also be provided as they
   -- refer to existing functions
   self.samplefunc = samplefunc or 'sampleDefault'

   -- display verbose messages
   self.verbose = verbose == nil and true or verbose
   
   self:reset()
end

function MultiImageSequence:buildIndex(cachepath, overwrite)
   if cachepath and (not overwrite) and paths.filep(cachepath) then
      if self.verbose then
         print("loading cached index")
      end
      local cache = torch.load(cachepath, 'ascii')
      for k,v in pairs(cache) do
         self[k] = v
      end
   else
      -- will need this package later to load images (faster than image package)
      require 'graphicsmagick'
      local _ = require 'moses'
      
      if self.verbose then
         print(string.format("Building index. Counting number of frames"))
      end
      
      -- index files
      local a = torch.Timer()
      local files = paths.indexdir(self.datapath, nil, nil, 'target*')
      
      local seqdirs = {}
      for i=1,files:size() do
         local filepath = files:filename(i)
         local seqdir, idx = filepath:match("/([^/]+)/input([%d]*)[.][^/]+$")
         if seqdir then
            local seq = seqdirs[seqdir]
            if not seq then
               seq = {}
               seqdirs[seqdir] = seq
            end
            seq[tonumber(idx)] = true
         end
      end
      
      self.seqdirs = {}
      self.nframe = 0
      for seqdir, seq in pairs(seqdirs) do
         if #seq > 0 then
            table.insert(self.seqdirs, {seqdir, #seq})
            self.nframe = self.nframe + #seq
         end
      end
      
      -- +#seqdirs because of masks between sequences
      self.seqlen = torch.ceil((self.nframe + #self.seqdirs)/self.batchsize)
      assert(#self.seqdirs > 0)
      
      if cachepath then
         local cache = {seqdirs=self.seqdirs, nframe=self.nframe, seqlen=self.seqlen}
         torch.save(cachepath, cache, "ascii")
      end
   end
   
   if self.verbose then
      print(string.format("Found %d sequences with a total of %d frames", #self.seqdirs, self.nframe))
   end
end

function MultiImageSequence:reset()
   parent.reset(self)
   self.trackers = {nextseq=1}
end

function MultiImageSequence:size()
   if not self.seqdirs then
      self:buildIndex()
   end
   return self.seqlen
end

-- size of input images
function MultiImageSequence:isize(excludedim)
   excludedim = excludedim == nil and 1 or excludedim
   assert(excludedim == 1)
   return {unpack(self.samplesize[1])}
end

-- size of target images
function MultiImageSequence:tsize(excludedim)
   excludedim = excludedim == nil and 1 or excludedim
   assert(excludedim == 1)
   return {unpack(self.samplesize[2])}
end

-- inputs : seqlen x batchsize x c x h x w
-- targets : seqlen x batchsize x c x h x w
function MultiImageSequence:sub(start, stop, inputs, targets, samplefunc)
   if not self.seqdirs then
      self:buildIndex()
   end
   local seqlen = stop - start + 1
   
   inputs = inputs or torch.FloatTensor()
   inputs:resize(seqlen, self.batchsize, unpack(self.samplesize[1])):zero()
   
   targets = targets or inputs.new()
   targets:resize(seqlen, self.batchsize, unpack(self.samplesize[2])):zero()
   
   -- samplefunc is a function that generates a sample input and target
   -- given their commensurate paths
   local samplefunc = samplefunc or self.samplefunc
   if torch.type(samplefunc) == 'string' then
      samplefunc = self[samplefunc]
   end
   assert(torch.type(samplefunc) == 'function')
   
   for i=1,self.batchsize do
   
      local input = inputs:select(2,i)
      local target = targets:select(2,i)
      
      local tracker = self.trackers[i] or {}
      self.trackers[i] = tracker
      
      local start = 1
      while start <= seqlen do
      
         if not tracker.seqid then
            tracker.idx = 1
            
            -- each sequence is separated by a zero input and target. 
            -- this should make the model forget between sequences
            -- (use with AbstractRecurrent:maskZero())
            input[start]:fill(0)
            target[start]:fill(0)
            
            start = start + 1
            
            if self.randseq then 
               tracker.seqid = math.random(1,#self.seqdirs)
            else
               tracker.seqid = self.trackers.nextseq
               self.trackers.nextseq = self.trackers.nextseq + 1
               if self.trackers.nextseq > #self.seqdirs then
                  self.trackers.nextseq = 1
               end
            end

         end
         
         if start <= seqlen then
            local seqdir, nframe = unpack(self.seqdirs[tracker.seqid])
            local seqpath = paths.concat(self.datapath, seqdir)
            
            local size = 0
            for i=tracker.idx,nframe do
               local inputpath = paths.concat(seqpath, string.format(self.inputpattern, i))
               local targetpath = paths.concat(seqpath, string.format(self.targetpattern, i))
               
               if i == nframe then
                  -- move on to next sequence
                  tracker.seqid = nil
               end
               
               size = size + 1
               samplefunc(self, input[start + size - 1], target[start + size - 1], inputpath, targetpath, tracker)
               tracker.idx = tracker.idx + 1
               
               if start + size - 1 == seqlen then
                  break
               end
            end
            
            start = start + size
         
         end
         
      end
        
      assert(start-1 == seqlen)
      
   end
   
   self:collectgarbage()
   return inputs, targets 
end

function MultiImageSequence:loadImage(path, idx, tracker)
   -- https://github.com/clementfarabet/graphicsmagick#gmimage
   local gm = require 'graphicsmagick'
   local lW, lH
   if self.varyloadsize then
      if self.scaleeverystep or tracker.idx == 1 then
         tracker.lW = tracker.lW or {}
         tracker.lH = tracker.lH or {}
         
         lW, lH = self.loadsize[idx][3], self.loadsize[idx][2]
         local sW, sH = self.samplesize[idx][3], self.samplesize[idx][2]
         -- sample a loadsize between samplesize and loadsize (same scale for input and target)
         tracker.scale = idx == 1 and math.random() or tracker.scale
         tracker.lW[idx] = torch.round(sW + tracker.scale*(lW-sW))
         tracker.lH[idx] = torch.round(sH + tracker.scale*(lH-sH))
      end
      lW, lH = tracker.lW[idx], tracker.lH[idx]
   else
      lW, lH = self.loadsize[idx][3], self.loadsize[idx][2]
   end
   assert(lW and lH)
   -- load image with size hints
   local input = gm.Image():load(path, lW, lH)
   -- resize by imposing the smallest dimension (while keeping aspect ratio)
   input:size(nil, math.min(lW, lH))
   return input
end

-- just load the image and resize it
function MultiImageSequence:sampleDefault(input, target, inputpath, targetpath, tracker)
   local input_ = self:loadImage(inputpath, 1, tracker)
   local colorspace = self.samplesize[1][1] == 1 and 'I' or 'RGB'
   local out = input_:toTensor('float',colorspace,'DHW', true)
   input:resize(out:size(1), self.samplesize[1][2], self.samplesize[1][3])
   image.scale(input, out)

   local target_ = self:loadImage(targetpath, 2, tracker)
   local colorspace = self.samplesize[2][1] == 1 and 'I' or 'RGB'
   local out = target_:toTensor('float',colorspace,'DHW', true)
   target:resize(out:size(1), self.samplesize[2][2], self.samplesize[2][3])
   image.scale(target, out)
   return input, target
end

-- function to load the image, jitter it appropriately (random crops, etc.)
function MultiImageSequence:sampleTrain(input, target, inputpath, targetpath, tracker)   
   local input_ = self:loadImage(inputpath, 1, tracker)
   local target_ = self:loadImage(targetpath, 2, tracker)
   
   -- do random crop once per sequence (unless self.cropeverystep)
   if tracker.idx == 1 or self.cropeverystep then
      tracker.cH = math.random()
      tracker.cW = math.random()
   end
   assert(tracker.cH and tracker.cW)
   
   local iW, iH = input_:size()
   local oW, oH = self.samplesize[1][3], self.samplesize[1][2]
   local h1, w1 = math.ceil(tracker.cH*(iH-oH)), math.ceil(tracker.cW*(iW-oW))
   local out = input_:crop(oW, oH, w1, h1)
   local colorspace = self.samplesize[1][1] == 1 and 'I' or 'RGB'
   out = out:toTensor('float',colorspace,'DHW', true)
   input:copy(out)
   
   local iW, iH = target_:size()
   local oW, oH = self.samplesize[2][3], self.samplesize[2][2]
   local h1, w1 = math.ceil(tracker.cH*(iH-oH)), math.ceil(tracker.cW*(iW-oW))
   local out = target_:crop(oW, oH, w1, h1)
   local colorspace = self.samplesize[2][1] == 1 and 'I' or 'RGB'
   out = out:toTensor('float',colorspace,'DHW', true)
   target:copy(out)
   
   return input, target
end
