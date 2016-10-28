local dl = require 'dataload._env'

local dltest = {}
local precision_forward = 1e-6
local precision_backward = 1e-6
local nloop = 50
local mytester

--e.g. usage: th -e "dl = require 'dataload'; dl.test()"

function dltest.loadMNIST()
   -- this unit test also tests TensorLoader to some extent.
   -- To test download, the data/mnist directory should be deleted
   local train, valid, test = dl.loadMNIST()
   
   -- test size and split
   mytester:assert(train:size()+valid:size()+test:size() == 70000)
   mytester:assert(torch.pointer(train.inputs:storage():data()) == torch.pointer(valid.inputs:storage():data()))
   
   -- test sub (and index incidently)
   local inputs, targets = train:sub(1,100)
   mytester:assertTableEq(inputs:size():totable(), {100,1,28,28}, 0.000001)
   mytester:assertTableEq(targets:size():totable(), {100}, 0.000001)
   mytester:assert(targets:min() >= 1)
   mytester:assert(targets:max() <= 10)
   
   -- test sample (and index)
   local inputs_, targets_ = inputs, targets
   inputs, targets = train:sample(100, inputs, targets)
   mytester:assert(torch.pointer(inputs:storage():data()) == torch.pointer(inputs_:storage():data()))
   mytester:assert(torch.pointer(targets:storage():data()) == torch.pointer(targets_:storage():data()))
   mytester:assertTableEq(inputs:size():totable(), {100,1,28,28}, 0.000001)
   mytester:assertTableEq(targets:size():totable(), {100}, 0.000001)
   mytester:assert(targets:min() >= 1)
   mytester:assert(targets:max() <= 10)
   mytester:assert(inputs:view(100,-1):sum(2):min() > 0)
   
   -- test shuffle
   local isum, tsum = train.inputs:sum(), train.targets:sum()
   local isum25, tsum25 = train.inputs:sub(2,5):sum(), train.targets:sub(2,5):sum()
   train:shuffle()
   mytester:assert(math.abs(isum - train.inputs:sum()) < 0.0000001)
   mytester:assert(math.abs(tsum - train.targets:sum()) < 0.0000001)
   mytester:assert(math.abs(isum25 - train.inputs:sub(2,5):sum()) > 0.00001)
   mytester:assert(math.abs(tsum25 - train.targets:sub(2,5):sum()) > 0.00001)
   
   -- test inputSize and outputSize
   local isize, tsize = train:isize(), train:tsize()
   mytester:assertTableEq(isize, {1,28,28}, 0.0000001)
   mytester:assert(#tsize == 0)
   mytester:assertTableEq(train:isize(false), {50000,1,28,28}, 0.0000001)
   mytester:assertTableEq(train:tsize(false), {50000}, 0.0000001)
end

function dltest.loadCIFAR10()
   -- To test download, the data/mnist directory should be deleted
   local train, valid, test = dl.loadCIFAR10()
   
   -- test size and split
   mytester:assert(train:size()+valid:size()+test:size() == 60000)
   mytester:assert(torch.pointer(train.inputs:storage():data()) == torch.pointer(valid.inputs:storage():data()))
   
   -- test sub (and index incidently)
   local inputs, targets = train:sub(1,100)
   mytester:assertTableEq(inputs:size():totable(), {100,3,32,32}, 0.000001)
   mytester:assertTableEq(targets:size():totable(), {100}, 0.000001)
   mytester:assert(targets:min() >= 1)
   mytester:assert(targets:max() <= 10)
   
   -- test sample (and index)
   local inputs_, targets_ = inputs, targets
   inputs, targets = train:sample(100, inputs, targets)
   mytester:assert(torch.pointer(inputs:storage():data()) == torch.pointer(inputs_:storage():data()))
   mytester:assert(torch.pointer(targets:storage():data()) == torch.pointer(targets_:storage():data()))
   mytester:assertTableEq(inputs:size():totable(), {100,3,32,32}, 0.000001)
   mytester:assertTableEq(targets:size():totable(), {100}, 0.000001)
   mytester:assert(targets:min() >= 1)
   mytester:assert(targets:max() <= 10)
   mytester:assert(inputs:view(100,-1):sum(2):min() > 0)
   
   -- test shuffle
   local isum, tsum = train.inputs:mean(), train.targets:mean()
   local isum25, tsum25 = train.inputs:sub(2,5):sum(), train.targets:sub(2,5):sum()
   train:shuffle()
   mytester:assert(math.abs(isum - train.inputs:mean()) < 0.0000001)
   mytester:assert(math.abs(tsum - train.targets:mean()) < 0.0000001)
   mytester:assert(math.abs(isum25 - train.inputs:sub(2,5):sum()) > 0.00001)
   mytester:assert(math.abs(tsum25 - train.targets:sub(2,5):sum()) > 0.00001)
   
   -- test inputSize and outputSize
   local isize, tsize = train:isize(), train:tsize()
   mytester:assertTableEq(isize, {3,32,32}, 0.0000001)
   mytester:assert(#tsize == 0)
   mytester:assertTableEq(train:isize(false), {41666,3,32,32}, 0.0000001)
   mytester:assertTableEq(train:tsize(false), {41666}, 0.0000001)
end

function dltest.TensorLoader()
   -- the tensor inputs and targets are tested by loadMNIST
   -- so we test the nested tensors here.
   local inputs = {torch.randn(100,3,4),{torch.randn(100,2)}}
   local targets = {torch.randn(100),{torch.randn(100,1)}}
   
   -- test size, isize, tsize
   local ds = dl.TensorLoader(inputs, targets)
   mytester:assert(ds:size() == 100)
   mytester:assert(#ds:isize() == 2)
   mytester:assertTableEq(ds:isize()[1], {3,4}, 0.0000001)
   mytester:assertTableEq(ds:isize()[2][1], {2}, 0.0000001)
   mytester:assertTableEq(ds:tsize()[2][1], {1}, 0.0000001)
   mytester:assert(#ds:tsize() == 2 )
   mytester:assert(#ds:tsize()[1] == 0)
   
   -- test sub (and index)
   local inputs_, targets_ = ds:sub(2,5)
   local inputs2_ = {inputs[1]:sub(2,5), {inputs[2][1]:sub(2,5)}}
   local targets2_ = {targets[1]:sub(2,5), {targets[2][1]:sub(2,5)}}
   mytester:assertTensorEq(inputs_[1], inputs2_[1], 0.00000001)
   mytester:assertTensorEq(inputs_[2][1], inputs2_[2][1], 0.00000001)
   mytester:assertTensorEq(targets_[1], targets2_[1], 0.00000001)
   mytester:assertTensorEq(targets_[2][1], targets2_[2][1], 0.00000001)
   
   -- test shuffle
   local isum = {inputs[1]:sum(), {inputs[2][1]:sum()}}
   local tsum = {targets[1]:sum(), {targets[2][1]:sum()}}
   local isum25 = {inputs[1]:sub(2,5):sum(), {inputs[2][1]:sub(2,5):sum()}}
   local tsum25 = {targets[1]:sub(2,5):sum(), {targets[2][1]:sub(2,5):sum()}}
   ds:shuffle()
   mytester:assert(math.abs(isum[1] - ds.inputs[1]:sum()) < 0.0000001)
   mytester:assert(math.abs(isum[2][1] - ds.inputs[2][1]:sum()) < 0.0000001)
   mytester:assert(math.abs(tsum[1] - ds.targets[1]:sum()) < 0.0000001)
   mytester:assert(math.abs(tsum[2][1] - ds.targets[2][1]:sum()) < 0.0000001)
   mytester:assert(math.abs(isum25[1] - ds.inputs[1]:sub(2,5):sum()) > 0.00001)
   mytester:assert(math.abs(isum25[2][1] - ds.inputs[2][1]:sub(2,5):sum()) > 0.00001)
   mytester:assert(math.abs(tsum25[1] - ds.targets[1]:sub(2,5):sum()) > 0.00001)
   mytester:assert(math.abs(tsum25[2][1] - ds.targets[2][1]:sub(2,5):sum()) > 0.00001)
   
   -- test split
   local ds1, ds2 = ds:split(0.2)
   mytester:assertTensorEq(ds1.inputs[1], ds.inputs[1]:sub(1,20), 0.0000001)
   mytester:assertTensorEq(ds1.targets[1], ds.targets[1]:sub(1,20), 0.0000001)
   mytester:assertTensorEq(ds2.inputs[1], ds.inputs[1]:sub(21,100), 0.0000001)
   mytester:assertTensorEq(ds2.targets[1], ds.targets[1]:sub(21,100), 0.0000001)
   mytester:assertTensorEq(ds1.inputs[2][1], ds.inputs[2][1]:sub(1,20), 0.0000001)
   mytester:assertTensorEq(ds1.targets[2][1], ds.targets[2][1]:sub(1,20), 0.0000001)
   mytester:assertTensorEq(ds2.inputs[2][1], ds.inputs[2][1]:sub(21,100), 0.0000001)
   mytester:assertTensorEq(ds2.targets[2][1], ds.targets[2][1]:sub(21,100), 0.0000001)
   
   -- test DataLoader:subiter
   
   -- should stop after 28 samples
   local batchsize = 10
   local epochsize = 28
   local start, stop = 1, 10
   local nsampled
   for k, inputs, targets in ds:subiter(batchsize, epochsize) do
      local inputs2 = {ds.inputs[1]:sub(start, stop), {ds.inputs[2][1]:sub(start, stop)}}
      mytester:assertTensorEq(inputs[1], inputs2[1], 0.0000001)
      mytester:assertTensorEq(inputs[2][1], inputs2[2][1], 0.0000001)
      start = stop + 1
      stop = math.min(epochsize, start + batchsize - 1)
      nsampled = k
   end
   mytester:assert(start-1 == 28)
   mytester:assert(nsampled == epochsize)
   
   -- should continue from previous state :
   local batchsize = 8
   local epochsize = 16
   stop = start + batchsize - 1
   for k, inputs, targets in ds:subiter(batchsize, epochsize) do
      local inputs2 = {ds.inputs[1]:sub(start, stop), {ds.inputs[2][1]:sub(start, stop)}}
      mytester:assertTensorEq(inputs[1], inputs2[1], 0.0000001)
      mytester:assertTensorEq(inputs[2][1], inputs2[2][1], 0.0000001)
      start = stop + 1
      stop = math.min(ds:size(), start + batchsize - 1)
      nsampled = k
   end
   mytester:assert(start-1 == 28+16)
   mytester:assert(nsampled == epochsize)
   
   -- should loop back to begining
   local batchsize = 32
   local epochsize = 100
   stop = start + batchsize - 1
   local i = 0
   for k, inputs, targets in ds:subiter(batchsize, epochsize) do
      if start == ds:size() + 1 then
         start, stop = 1, 32
      end
      i = i + 1
      local inputs2 = {ds.inputs[1]:sub(start, stop), {ds.inputs[2][1]:sub(start, stop)}}
      mytester:assertTensorEq(inputs[1], inputs2[1], 0.0000001)
      mytester:assertTensorEq(inputs[2][1], inputs2[2][1], 0.0000001)
      start = stop + 1
      stop = math.min(ds:size(), start + batchsize - 1)
      if i == 3 then
         stop = start + 11
      end
      nsampled = k
   end
   
   mytester:assert(start-1 == 28+16)
   mytester:assert(nsampled == epochsize)
   
   -- test sampleiter 
   
   local rowsums = {}
   for i=1,ds.inputs[1]:size(1) do
      local sum = ds.inputs[1][i]:sum()
      assert(not rowsums[sum])
      rowsums[sum] = {
         inputs={ds.inputs[1][i],{ds.inputs[2][1][i]}}, 
         targets={ds.targets[1][i],{ds.targets[2][1][i]}},
         idx = i
      }
   end
   
   local batchsize = 24
   local epochsize = 1000
   
   local rowcounts = torch.Tensor(ds.inputs[1]:size(1)):zero()
   local nsampled = 0
   for k, inputs, targets in ds:sampleiter(batchsize, epochsize) do
      for i=1,inputs[1]:size(1) do
         local sum = inputs[1][i]:sum()
         local row = rowsums[sum]
         mytester:assert(row ~= nil)
         rowcounts[row.idx] = rowcounts[row.idx] + 1
         
         mytester:assertTensorEq(row.inputs[1], inputs[1][i], 0.000001)
         mytester:assertTensorEq(row.inputs[2][1], inputs[2][1][i], 0.000001)
         mytester:assert(math.abs(row.targets[1] - targets[1][i]) < 0.000001)
         mytester:assertTensorEq(row.targets[2][1], targets[2][1][i], 0.000001)
      end
      nsampled = k
   end
   mytester:assert(nsampled == epochsize)
   mytester:assert(rowcounts:min() > 0)
   local std = rowcounts:std()
   mytester:assert(std > 2.3 and std < 4)
   
   -- simple subiter test 
   local dataloader = dl.TensorLoader(torch.range(1,5), torch.range(1,5))

   local inputs_ = {}
   table.insert(inputs_, dataloader.inputs:sub(1,2))
   table.insert(inputs_, dataloader.inputs:sub(3,4))
   table.insert(inputs_, dataloader.inputs:sub(5,5))
   table.insert(inputs_, dataloader.inputs:sub(1,1))

   local i = 0
   
   for k, inputs, targets in dataloader:subiter(2,6) do
      i = i + 1
      mytester:assertTensorEq(inputs_[i], inputs, 0.0000001)
   end

end

function dltest.ImageClass()
   local datapath = paths.concat(dl.DATA_PATH, "_unittest_")
   local buffer
   
   if not paths.dirp(datapath) then
      -- create a dummy dataset based on MNIST
      local mnist = dl.loadMNIST()
      
      os.execute("rm -r "..datapath)
      
      paths.mkdir(datapath)
      
      local inputs, targets
      for i=1,10 do
         local classpath = paths.concat(datapath, "class"..i)
         paths.mkdir(classpath)
         inputs, targets = mnist:sample(100, inputs, targets)
         for j=1,100 do
            local input = inputs[j]
            if math.random() < 0.5 then
               buffer = buffer or inputs.new()
               if math.random() < 0.5 then 
                  buffer:resize(1, 32, 28)
               else
                  buffer:resize(1, 28, 32)
               end
               image.scale(buffer, input)
               input = buffer
            end
            image.save(paths.concat(classpath, "image"..j..".jpg"), input)
         end
      end
   end
   
   local ignorepath = paths.concat(datapath, 'class1/ignore')
   if not paths.dirp(ignorepath) then
      local mnist = dl.loadMNIST()
      paths.mkdir(ignorepath)
      local inputs, targets = mnist:sample(10, inputs, targets)
      for j=1,10 do
         local input = inputs[j]
         if math.random() < 0.5 then
            buffer = buffer or inputs.new()
            if math.random() < 0.5 then 
               buffer:resize(1, 32, 28)
            else
               buffer:resize(1, 28, 32)
            end
            image.scale(buffer, input)
            input = buffer
         end
         image.save(paths.concat(ignorepath, "image"..j..".jpg"), input)
      end
   end
      
   
   -- Note that I can't really test scaling as gm and image scale differently
   local ds = dl.ImageClass(datapath, {1, 28, 28}, {1, 28, 28}, nil, nil, false, nil, "*/ignore/*")
   
   -- test index
   local inputs, targets = ds:index(torch.LongTensor():range(201,300))
   local inputs2, targets2 = inputs:clone():zero(), targets:clone():zero()
   local buffer
   
   for i=201,300 do
      local imgpath = ffi.string(torch.data(ds.imagePath[i]))      
      local img = image.load(imgpath):float()
      
      -- also make sure that the resize happens the right way
      local gimg = ds:loadImage(imgpath)
      local gimg2 = gimg:toTensor('float','R','DHW', true)
      
      buffer = buffer or torch.FloatTensor()
      buffer:resizeAs(img) 
      image.scale(buffer, img)
      
      mytester:assertTensorEq(gimg2, buffer, 0.00001)
      
      image.scale(inputs2[i-200], buffer)
      targets2[i-200] = ds.iclasses[string.match(imgpath, "(class%d+)/image%d+[.]jpg$")]
   end
   
   mytester:assertTensorEq(inputs, inputs2, 0.000001)
   mytester:assertTensorEq(targets, targets2, 0.000001)
   
   -- test size
   mytester:assert(ds:size() == 1000)
   
   -- test sample
   local inputs_, targets_ = inputs, targets
   inputs, targets = ds:sample(100, inputs, targets)
   mytester:assert(torch.pointer(inputs:storage():data()) == torch.pointer(inputs_:storage():data()))
   mytester:assert(torch.pointer(targets:storage():data()) == torch.pointer(targets_:storage():data()))
   mytester:assertTableEq(inputs:size():totable(), {100,1,28,28}, 0.000001)
   mytester:assertTableEq(targets:size():totable(), {100}, 0.000001)
   mytester:assert(targets:min() >= 1)
   mytester:assert(targets:max() <= 10)
   mytester:assert(inputs:view(100,-1):sum(2):min() > 0)
   
   -- test sampleTrain
   ds.samplesize = {1,14,14}
   inputs, targets = ds:sub(1, 5, inputs, targets, ds.sampleTrain)
   local inputs2, targets2 = ds:sub(1, 5)
   for i=1,5 do
      mytester:assertTensorNe(inputs, inputs2, 0.0000001)
   end
   mytester:assertTensorEq(targets, targets2, 0.0000001)
   
   -- test sampleTest
   inputs, targets = ds:sub(1, 5, inputs, targets, ds.sampleTest)
   mytester:assertTableEq(inputs:size():totable(), {50, 1, 14, 14}, 0.0000001)
   mytester:assertTableEq(targets:size():totable(), {50}, 0.0000001)
   mytester:assert(targets:view(5,10):float():std(2):sum() < 0.0000001)
end

function dltest.AsyncIterator()
   if not pcall(function() require 'threads' end) then
      return
   end
   local inputs, targets = torch.randn(100,3), torch.randn(100, 10)
   local ds1 = dl.TensorLoader(inputs, targets)
   local ds2 = dl.AsyncIterator(ds1,2)
   mytester:assert(true)
   
   local batches = {}
   for i, inputs, targets in ds1:subiter() do
      assert(not batches[inputs:sum()])
      batches[inputs:sum()] = {inputs=inputs:clone(), targets=targets:clone()}
   end
   
   local n = 0
   local bidx = 0
   for i, inputs, targets in ds2:subiter() do
      bidx = bidx + 1
      n = n + inputs:size(1)
      local batch2 = batches[inputs:sum()]
      mytester:assert(batch2 ~= nil)
      mytester:assertTensorEq(batch2.inputs, inputs, 0.0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0.0000001)
      batches[inputs:sum()] = nil
   end
   mytester:assert(bidx == 4)
   
   ds1:reset()
   ds2:reset()
   
   -- should stop after 28 samples
   local batchsize = 10
   local epochsize = 28
   
   local batches = {}
   for i, inputs, targets in ds1:subiter(batchsize, epochsize) do
      assert(not batches[inputs:sum()])
      batches[inputs:sum()] = {inputs=inputs:clone(), targets=targets:clone()}
   end
   
   local nsampled
   for k, inputs, targets in ds2:subiter(batchsize, epochsize) do
      local batch2 = batches[inputs:sum()]
      mytester:assert(batch2 ~= nil)
      mytester:assertTensorEq(batch2.inputs, inputs, 0.0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0.0000001)
      batches[inputs:sum()] = nil
      nsampled = k
   end
   mytester:assert(nsampled == epochsize)
   
   -- should continue from previous state :
   local batchsize = 8
   local epochsize = 16
   
   local batches = {}
   for i, inputs, targets in ds1:subiter(batchsize, epochsize) do
      assert(not batches[inputs:sum()])
      batches[inputs:sum()] = {inputs=inputs:clone(), targets=targets:clone()}
   end
   
   for k, inputs, targets in ds2:subiter(batchsize, epochsize) do
      local batch2 = batches[inputs:sum()]
      mytester:assert(batch2 ~= nil)
      mytester:assertTensorEq(batch2.inputs, inputs, 0.0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0.0000001)
      batches[inputs:sum()] = nil
      nsampled = k
   end
   mytester:assert(nsampled == epochsize)
   
   -- should loop back to begining
   local batchsize = 32
   local epochsize = 100
   
   local batches = {}
   for i, inputs, targets in ds1:subiter(batchsize, epochsize) do
      assert(not batches[inputs:sum()])
      batches[inputs:sum()] = {inputs=inputs:clone(), targets=targets:clone()}
   end
   
   for k, inputs, targets in ds2:subiter(batchsize, epochsize) do
      local batch2 = batches[inputs:sum()]
      mytester:assert(batch2 ~= nil)
      mytester:assertTensorEq(batch2.inputs, inputs, 0.0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0.0000001)
      batches[inputs:sum()] = nil
      nsampled = k
   end
   mytester:assert(nsampled == epochsize)
   mytester:assert(ds2.querymode == 'subiter')
   
   -- test sampleiter 
   
   local rowsums = {}
   for i=1,ds1.inputs:size(1) do
      local sum = ds1.inputs[i]:sum()
      assert(not rowsums[sum])
      rowsums[sum] = {inputs=ds1.inputs[i], targets=ds1.targets[i], idx=i}
   end
   
   local batchsize = 24
   local epochsize = 1000
   
   local rowcounts = torch.Tensor(ds1.inputs:size(1)):zero()
   local nsampled = 0
   for k, inputs, targets in ds2:sampleiter(batchsize, epochsize) do
      for i=1,inputs:size(1) do
         local sum = inputs[i]:sum()
         local row = rowsums[sum]
         mytester:assert(row ~= nil)
         rowcounts[row.idx] = rowcounts[row.idx] + 1
         
         mytester:assertTensorEq(row.inputs, inputs[i], 0.000001)
         mytester:assertTensorEq(row.targets, targets[i], 0.000001)
      end
      nsampled = k
   end
   mytester:assert(nsampled == epochsize)
   mytester:assert(rowcounts:min() > 0)
   local std = rowcounts:std()
   mytester:assert(std > 2.3 and std < 4)
   mytester:assert(ds2.querymode == 'sampleiter')
end

function dltest.SequenceLoader()
   local data = torch.LongTensor(1003)
   local batchsize = 50
   local seqlen = 5
   local ds = dl.SequenceLoader(data, batchsize)
   local data2 = data:sub(1,1000):view(50, 1000/50):t()
   mytester:assertTensorEq(data:narrow(1,1,1000/50), data2:select(2,1), 0.0000001)
   mytester:assertTensorEq(ds.data, data2, 0.000001)
   
   local inputs, targets = ds:sub(1, 5)
   mytester:assertTensorEq(ds.data:sub(1,5), inputs, 0.0000001)
   mytester:assertTensorEq(ds.data:sub(2,6), targets, 0.0000001)
   
   local start2 = 1
   for start, inputs, targets in ds:subiter(seqlen) do
      local stop2 = math.min(start2+seqlen-1, data2:size(1)-1)
      local inputs2 = data2:sub(start2,stop2)
      local targets2 = data2:sub(start2+1,stop2+1)
      
      mytester:assertTensorEq(inputs, inputs2, 0.000001)
      mytester:assertTensorEq(targets, targets2, 0.000001)
      start2 = start2 + seqlen
   end
   
   mytester:assert(start2 == 1000/50 + 1)
end   

-- each subfolder (seqpath) has an input and target sequence.
-- tests :
-- each image is loaded in the correct order;
-- each input is aligned to its target;
-- the mask images are inserted correctly between sequences;

function dltest.MultiImageSequence()
   local datapath = paths.concat(dl.DATA_PATH, "_unittest2_")
   local buffer
   
   local function getid(img)
      img:view(1,-1)
      local maxid, maxval = 0, -9999
      local i = 0
      img:apply(function(x)
         i = i + 1
         if x > maxval then
            maxid = i
            maxval = x
         end
      end)
      return maxid
   end
   
   -- create dummy dataset
   if paths.dirp(datapath) then
      os.execute("rm -r "..datapath)
   end
   local inputs = torch.FloatTensor(256,1,16*16):zero()
   local targets = torch.FloatTensor(256,1,16*16):zero()
   for i=1,256 do
      inputs[{i,1,i}] = 1
      targets[{i,1,256-i+1}] = 1
   end
   
   paths.mkdir(datapath)
   
   local seqs = {}
   local ids = {}
   local nframe, nseq = 0, 0
   for seqid=1,256-6,7 do
      nseq = nseq + 1
      local seqdir = "seq"..seqid
      local seqpath = paths.concat(datapath, seqdir)
      paths.mkdir(seqpath)
      
      local seqlen = math.random(1,7)
      for j=1,seqlen do
         nframe = nframe + 1
         local id = seqid+j-1
         table.insert(ids, id)
         image.save(paths.concat(seqpath, "input"..j..".png"), inputs[id]:view(1,16,16))
         image.save(paths.concat(seqpath, "target"..j..".png"), targets[id]:view(1,16,16))
         local input = image.load(paths.concat(seqpath, "input"..j..".png"))
         assert(math.abs(id - getid(input)) < 0.001)
      end
      
      seqs[seqdir] = {inputs:narrow(1,seqid,seqlen), targets:narrow(1,seqid,seqlen)}
   end
   local nsample = nframe + nseq
   
   -- init MultiImageSequence
   local loadsize = {{1, 16, 16},{1, 16, 16}}
   local samplesize = {1, 16, 16}
   local batchsize = 4
   local ds = dl.MultiImageSequence(datapath, batchsize, loadsize, samplesize)
   ds.verbose = false
   ds.inputpattern = 'input%d.png'
   ds.targetpattern = 'target%d.png'
   ds:buildIndex()
   
   mytester:assert(ds.nframe == nframe)
   mytester:assert(#ds.seqdirs == nseq)
   mytester:assert(ds:size() == torch.ceil(nsample/batchsize))
   
   local sequences = {}
   for i, seqtbl in ipairs(ds.seqdirs) do
      local seqdir, nframe = unpack(seqtbl)
      local tensor = seqs[seqdir]
      assert(tensor)
      table.insert(sequences, tensor)
   end
   
   -- test sub   
   local inputs, targets = ds:sub(1, 15)
   
   local seqid, seqidx = 0, -1
   local seq
   for i=1,inputs:size(2) do
      local inputs_ = inputs:select(2,i)
      local targets_ = targets:select(2,i)
      if seqidx ~= 0 then
         seqid = seqid + 1
         seq = sequences[seqid]
         seqidx = 0
      end
      
      for j=1,inputs:size(1) do
         local inimg = inputs_[j]
         local outimg = targets_[j]
         
         if seqidx == 0 then
            --print(seqidx, inimg:sum())
            mytester:assert(inimg:sum() == 0, "zero input mask "..i..':'..j)
            mytester:assert(outimg:sum() == 0, "zero target mask "..i..':'..j)
            seqidx = seqidx + 1
         else
            --print(seqidx, getid(seq[1][seqidx]), getid(inimg))
            mytester:assert(getid(seq[1][seqidx]) == getid(inimg))
            mytester:assert(getid(seq[2][seqidx]) == getid(outimg))
            seqidx = seqidx + 1
            if seqidx > seq[1]:size(1) then
               seqidx = 0
               seqid = seqid + 1
               seq = sequences[seqid]
            end
         end
         
      end
   end
   
   local tensor = torch.LongTensor(ds:size(), 1, 16, 16)
   
   -- test subiter
   ds:reset()
   local nstart = 0
   local startidx = 1
   for i, inputs, targets in ds:subiter(15) do
      for k=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            if inputs[k][j]:sum() == 0 then
               nstart = nstart + 1
            end
         end
      end
      local stop = math.min(ds:size(), startidx + 15 - 1)
      local size = stop - startidx + 1
      -- copies first sequence into tensor over all iterations
      tensor:narrow(1, startidx, size):copy(inputs:select(2,1))
      startidx = startidx + size
   end
   mytester:assert(nstart >= nseq and nstart <= nseq+(batchsize*2))
   mytester:assert(startidx == ds:size() + 1)
   
   local eq = torch.LongTensor()
   
   local tensors = {}
   local startidx = 1
   for idx=1,tensor:size(1) do
      local x = tensor[idx]
      if x:sum() == 0 and idx > 1 then
         table.insert(tensors, tensor:sub(startidx+1,idx-1))
         startidx = idx
      end
   end
   for i, tensor in ipairs(tensors) do
      local found = false
      for k,sequence in pairs(sequences) do
         if tensor:size(1) == sequence[1]:size(1) then
            if eq:eq(tensor, sequence[1]:view(sequence[1]:size(1),1,16,16):long()):min() == 1 then
               found = true
               sequences[k] = nil
               break
            end
         end
      end
      mytester:assert(found, "missing sequence "..i)
   end
   
   -- create dummy images
   local imgpath = paths.concat(datapath, 'lenna.png')
   local lenna = image.lena():float()
   image.save(imgpath, lenna)
   
   -- test sampleDefault with self.varyloadsize and rescale
   local loadsize = {{3, 128, 128},{3, 96, 96}}
   local samplesize = {3, 64, 64}
   local batchsize = 4
   local ds = dl.MultiImageSequence(datapath, batchsize, loadsize, samplesize)
   mytester:assertTableEq(ds.samplesize[2], {3,48,48}, 0.000001)
   ds.inputpattern = 'input%d.png'
   ds.targetpattern = 'target%d.png'
   ds.verbose = false
   ds:buildIndex()
   ds.varyloadsize = true
   
   local input = torch.FloatTensor(1,64,64)
   local target = torch.FloatTensor(1,48,48)
   local tracker = {idx=1}
   ds:sampleDefault(input, target, imgpath, imgpath, tracker)
   local input2 = input:clone()
   local target2 = target:clone()
   image.scale(input2, lenna)
   image.scale(target2, lenna)
   mytester:assertTensorEq(input2, input, 0.1)
   mytester:assertTensorEq(target2, target, 0.1)
   
   image.save(paths.concat(datapath, 'lenna1_input.png'), input)
   image.save(paths.concat(datapath, 'lenna1_target.png'), target)
   
   -- test sampleTrain
   
   input:zero()
   target:zero()
   tracker = {idx=1}
   ds:sampleTrain(input, target, imgpath, imgpath, tracker)
   tracker.idx = 2
   local lenna_ = ds:loadImage(imgpath, 1, tracker)
   local iW, iH = lenna_:size()
   local oW, oH = 64, 64
   local h1, w1 = math.ceil(tracker.cH*(iH-oH)), math.ceil(tracker.cW*(iW-oW))
   local out = lenna_:crop(oW, oH, w1, h1)
   local colorspace = 'RGB'
   out = out:toTensor('float',colorspace,'DHW', true)
   mytester:assertTensorEq(out, input, 0.000001)
   
   image.save(paths.concat(datapath, 'lenna2_input.png'), input)
   image.save(paths.concat(datapath, 'lenna2_target.png'), target)
end

function dltest.loadPTB()
   local batchsize = 20
   local seqlen = 5
   local train, valid, test = dl.loadPTB(20)
   
   mytester:assert(#train.ivocab == 10000)
   local textsize, vocabsize = 0, 0
   for word, wordid in pairs(train.vocab) do
      textsize = textsize + train.wordfreq[word]
      vocabsize = vocabsize + 1
   end
   mytester:assert(vocabsize == 10000)
   mytester:assert(train:size() == math.floor(textsize/batchsize)-1)
   mytester:assert(not train.vocab['<OOV>'])
   mytester:assert(valid ~= nil)
   mytester:assert(test ~= nil)
   
   if false then
      local sequence = {}
      for i,inputs,targets in train:subiter(seqlen) do
         for k=1,inputs:size(1) do
            table.insert(sequence, train.ivocab[inputs[{k,1}]] or 'WTF?')
         end
      end
      print(table.concat(sequence, ' '))
   end
end

function dltest.loadSentiment140()
   train, valid, test = dl.loadSentiment140()
   mytester:assert(train ~= nil)
   mytester:assert(valid ~= nil)
   mytester:assert(test ~= nil)
   mytester:assert(test:size() == 359)
   
   mytester:assert(train.targets:min() == 1)
   mytester:assert(train.targets:max() == 2)
   
   if false then
      for i=1,10 do
         print(i, train.targets[i], train:tensor2text(train.inputs[i]))
      end
   end
end

function dltest.loadImageNet()
   local nthread = 2
   local batchsize = 200
   local epochsize = 20000
   local datapath = '/data2/ImageNet/'
   
   if not paths.dirp(datapath) then
      return
   end
   
   local train, valid = dl.loadImageNet(datapath, 2, nil, nil, true)
   
   -- test subiter
   local a = torch.Timer()
   local isum, tsum = 0, 0
   for i, inputs, targets in valid:subiter(batchsize/10, epochsize) do
      isum = isum + inputs:sum()
      tsum = tsum + targets:sum()
   end
   print("async subiter", a:time().real)
   
   local a = torch.Timer()
   local isum2, tsum2 = 0, 0
   valid.dataset:reset()
   for i, inputs, targets in valid.dataset:subiter(batchsize/10, epochsize) do
      isum2 = isum2 + inputs:sum()
      tsum2 = tsum2 + targets:sum()
   end
   print("sync subiter", a:time().real)
   
   mytester:assert(math.abs(isum - isum2) < 0.000001)
   mytester:assert(math.abs(tsum - tsum2) < 0.000001)
   
   -- test sampleiter
   local a = torch.Timer()
   for i, inputs, targets, imagepaths in train.dataset:subiter(batchsize/10, epochsize) do
      -- pass
   end
   print("sync sampleiter", a:time().real)
   
   local a = torch.Timer()
   for i, inputs, targets, imagepaths in train:sampleiter(batchsize/10, epochsize) do
      -- pass
   end
   print("async sampleiter", a:time().real)
   
   -- save some images from train and valid set loaders
   
   local samplepath = paths.concat(datapath, 'unittest')
   paths.mkdir(samplepath)
   for i, inputs, targets, imagepaths in train:sampleiter(batchsize/10, 400) do
      for idx=1,inputs:size(1) do
         local input, target = inputs[idx], targets[idx]
         image.save(paths.concat(samplepath, target..'_t_'..paths.basename(imagepaths[idx])), input)
      end
   end
   
   for i, inputs, targets, imagepaths in valid:subiter(batchsize/10, 400) do
      local inputs = inputs:view(-1, 10, 3, 224, 224)
      local targets = targets:view(-1, 10)
      for idx=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local input, target = inputs[idx][j], targets[idx][j]
            image.save(paths.concat(samplepath, target..'_v_'..paths.basename(imagepaths[idx]))..j..'.jpg', input)
         end
      end
   end
end

function dltest.fitImageNormalize()
   local trainset, validset, testset = dl.loadMNIST()
   local ppf = dl.fitImageNormalize(trainset, 5000)
   
   local inputs = validset:sample(100)
   ppf(inputs)
   mytester:assert(math.abs(inputs:mean()) < 0.05)
   mytester:assert(math.abs(inputs:std() - 1) < 0.05) 
end

function dltest.MultiSequence()
   local sequences = {}
   for i=1,200 do
      table.insert(sequences, torch.LongTensor(math.random(3,20)):random(1,100))
   end
   local batchsize = 4
   local ds = dl.MultiSequence(sequences, 8)
   local inputs, targets = ds:sub(1, 15)
   
   local seqid, seqidx = 0, -1
   local seq
   for i=1,inputs:size(2) do
      local inputs_ = inputs:select(2,i)
      local targets_ = targets:select(2,i)
      if seqidx ~= 0 then
         seqid = seqid + 1
         seq = sequences[seqid]
         seqidx = 0
      end
      
      for j=1,inputs:size(1) do
         local inid = inputs_[j]
         local outid = targets_[j]
         if seqidx == 0 then
            mytester:assert(inid == 0)
            mytester:assert(outid == 1)
            seqidx = seqidx + 1
         else
            mytester:assert(seq[seqidx] == inid)
            mytester:assert(seq[seqidx+1] == outid)
            if seq[seqidx+1] ~= outid then
               print(i, j)
               print(seqid, seqidx)
               print(seq)
               print(inputs:t())
               return
            end
            seqidx = seqidx + 1
            if seqidx == seq:size(1) then
               seqidx = 0
               seqid = seqid + 1
               seq = sequences[seqid]
            end
         end
         
      end
   end
   
   local tensor = torch.LongTensor(ds:size())
   
   ds:reset()
   local nstart = 0
   local startidx = 1
   for i, inputs, targets in ds:subiter(15) do
      inputs:apply(function(x)
         if x == 0 then
            nstart = nstart + 1
         end
      end)
      local stop = math.min(ds:size(), startidx + 15 - 1)
      local size = stop - startidx + 1
      tensor:narrow(1, startidx, size):copy(inputs:select(2,1))
      startidx = startidx + size
   end
   mytester:assert(nstart >= 200 and nstart <= 200+(batchsize*2))
   mytester:assert(startidx == ds:size() + 1)
   
   local eq = torch.LongTensor()
   
   local tensors = {}
   local startidx = 1
   local idx = 0
   tensor:apply(function(x)
      idx = idx + 1
      if x == 0 and idx > 1 then
         table.insert(tensors, tensor:sub(startidx+1,idx-1))
         startidx = idx
      end
   end)
   for i, tensor in ipairs(tensors) do
      local found = false
      mytester:assert(tensor:min() > 0)
      for k,sequence in pairs(sequences) do
         if tensor:size(1) == sequence:size(1)-1 then
            if eq:eq(tensor, sequence:sub(1,-2)):min() == 1 then
               found = true
               sequences[k] = nil
               break
            end
         end
      end
      mytester:assert(found)
   end
   
   -- test that it works with multi-dimensional sequences
   local sequences = {}
   for i=1,200 do
      table.insert(sequences, torch.LongTensor(math.random(3,20), 2):random(1,100))
   end
   local batchsize = 4
   local ds = dl.MultiSequence(sequences, 3)
   local inputs, targets = ds:sub(1, 15) 
   mytester:assertTableEq(inputs:size():totable(), {15, 3, 2}, 0.00001)
end

function dltest.loadGBW()
   local batchsize = {50,1,1}
   local trainfile = 'train_tiny.th7'
   local a = torch.Timer()
   local trainset, validset, testset = dl.loadGBW(batchsize, trainfile, nil, nil, false)
   
   local words = {}
   local seqlen = 20
   for i,inputs, targets in trainset:subiter(seqlen) do
      for j=1,inputs:size(1) do
         local word = trainset.ivocab[inputs[{j,3}]]
         if word then
            table.insert(words, word)
         end
      end
   end
   local words = table.concat(words, ' ')
   mytester:assert(words:find('M3 money supply growth , which ran at 8.9pc in the year to August , and to bring policy interest rates , currently 2pc , above the reported inflation rate of 2.2pc. <S> THE Federal Reserve Board may want to scrutinize another statistic to gauge the health of the economy :') ~= nil)
end

function dltest.buildBigrams()
   local trainset = dl.loadPTB(20)
   local bigrams = dl.buildBigrams(trainset)
   mytester:assert(#bigrams == #trainset.ivocab)
end

--e.g. usage: th -e "dl = require 'dataload'; dl.test()"

function dl.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dltest)
   mytester:run(tests)
end
