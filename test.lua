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
         mytester:assert(row)
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
   
   if not paths.dirp(datapath) then
      -- create a dummy dataset based on MNIST
      local mnist = dl.loadMNIST()
      
      
      os.execute("rm -r "..datapath)
      
      paths.mkdir(datapath)
      
      local buffer
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
   
   -- Note that I can't really test scaling as gm and image scale differently
   local ds = dl.ImageClass(datapath, {1, 28, 28}, {1, 28, 28}, nil, nil, false)
   
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
      mytester:assert(batch2)
      mytester:assertTensorEq(batch2.inputs, inputs, 0,0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0,0000001)
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
      mytester:assert(batch2)
      mytester:assertTensorEq(batch2.inputs, inputs, 0,0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0,0000001)
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
      mytester:assert(batch2)
      mytester:assertTensorEq(batch2.inputs, inputs, 0,0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0,0000001)
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
      mytester:assert(batch2)
      mytester:assertTensorEq(batch2.inputs, inputs, 0,0000001)
      mytester:assertTensorEq(batch2.targets, targets, 0,0000001)
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
         mytester:assert(row)
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
   mytester:assert(valid)
   mytester:assert(test)
   
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

function dl.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dltest)
   mytester:run(tests)
end
