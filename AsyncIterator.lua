------------------------------------------------------------------------
--[[ AsyncIterator ]]--
-- Decorates a DataLoader to make it multi-threaded.
------------------------------------------------------------------------
local dl = require 'dataload._env'
local AsyncIterator, parent = torch.class('dl.AsyncIterator', 'dl.DataLoader', dl)

function AsyncIterator:__init(dataset, nthread, verbose)
   self.dataset = dataset
   assert(torch.isTypeOf(self.dataset, "dl.DataLoader"))
   self.nthread = nthread or 2
   self.verbose = verbose == 'nil' and true or verbose
   
   -- reset that shouldn't be shared by threads
   self.dataset:reset()
   
   -- upvalues should be local
   local verbose = self.verbose
   local datasetstr = torch.serialize(dataset, "ascii")
   local mainSeed = os.time()
   
   -- the torch threads library handles the multi-threading for us 
   local threads = require 'threads'
   -- so that the tensors don't get serialized, i.e. threads share them
   threads.Threads.serialization('threads.sharedserialize')
   
   
   -- build a Threads pool
   self.threads = threads.Threads(
      nthread, -- the following functions are executed in each thread
      function()
         dl = require 'dataload'
      end,
      function(idx)
         local success, err = pcall(function()
            t = {}
            t.id = idx
            local seed = mainSeed + idx
            math.randomseed(seed)
            torch.manualSeed(seed)
            if verbose then
               print(string.format('Starting worker thread with id: %d seed: %d', t.id, seed))
            end
         
            -- dataset is serialized in main thread and deserialized in dataset
            t.dataset = torch.deserialize(datasetstr, "ascii")
         end)
         if not success then
            print(err)
            error""
         end
      end
   )
   
   self.recvqueue = torchx.Queue()
   self.ninprogress = 0
end

function AsyncIterator:synchronize()
   self.threads:synchronize()
   while not self.recvqueue:empty() do
     self.recvqueue:get()
   end
   self.ninprogress = 0
   self.querymode = nil
   collectgarbage()
end

function AsyncIterator:reset()
   self:synchronize()
   parent.reset(self)
end

-- send request to worker : put request into queue
function AsyncIterator:asyncPut(fn, args, size)   
   assert(torch.type(fn) == 'string')
   assert(torch.type(args) == 'table')
   assert(torch.type(size) == 'number') -- size of batch
   for i=1,1000 do
      if self.threads:acceptsjob() then
         break
      else
         sys.sleep(0.01)
      end
      if i==1000 then
         error"infinite loop"
      end
   end
   
   self.ninprogress = (self.ninprogress or 0) + 1
   
   self.threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         local success, res = pcall(function()
            -- fn, args and size are upvalues
            local res = {t.dataset[fn](t.dataset, unpack(args))}
            res.size = size
            return res
         end)
         if not success then
            print(res)
            error""
         end
         return res
      end,
      -- the endcallback (runs in the main thread)
      function(res)      
         assert(torch.type(res) == 'table')
         self.recvqueue:put(res)
         self.ninprogress = self.ninprogress - 1
      end
   )
end

-- recv results from worker : get results from queue
function AsyncIterator:asyncGet()
   -- necessary because Threads:addjob sometimes calls dojob...
   if self.recvqueue:empty() then
      self.threads:dojob()
   end
   assert(not self.recvqueue:empty())
   return self.recvqueue:get()
end

-- iterators : subpairs, samplepairs
function AsyncIterator:subiter(batchsize, epochsize, ...)
   batchsize = batchsize or 32
   local dots = {...}
   
   -- empty the async queue
   self:synchronize()
   self.querymode = 'subpairs'
   
   local size = self:size()
   epochsize = epochsize or self:size()
   self._start = self._start or 1
   local nput = 0
   local nget = 0 -- nsampled
   local stop
   
   local previnputs, prevtargets
   
   local putOnly = true
     
   -- build iterator
   local iterate = function()
      assert(self.querymode == 'subpairs', "Can only support one iterator per synchronize()")
      
      if nget >= epochsize then
         return
      end
      
      
      -- asynchronously send (put) a task to a worker to be retrieved (get)
      -- by a later iteration
      if nput < epochsize then
         local bs = math.min(nput+batchsize, epochsize) - nput
         stop = math.min(self._start + bs - 1, size)
         bs = stop - self._start + 1
         
         -- if possible, reuse buffers previnputs and prevtargets
         self:asyncPut('sub', {self._start, stop, previnputs, prevtargets, unpack(dots)}, bs)
         
         nput = nput + bs
         self._start = self._start + bs
         if self._start >= size then
            self._start = 1
         end
      end
      
      if not putOnly then
         local batch = self:asyncGet()
         -- we will resend these buffers to the workers next call
         previnputs, prevtargets = batch[1], batch[2]
         nget = nget + batch.size
         self:collectgarbage()
         return nget, unpack(batch)
      end
   end
   
   -- fill task queue with some batch requests
   for tidx=1,self.nthread do
      iterate()
   end
   putOnly = false
   
   return iterate
end

function AsyncIterator:size(...)
   return self.dataset:size(...)
end
