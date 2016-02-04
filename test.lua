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


function dl.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dltest)
   mytester:run(tests)
end
