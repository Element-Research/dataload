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
end


function dl.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dltest)
   mytester:run(tests)
end
