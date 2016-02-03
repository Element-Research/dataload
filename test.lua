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
   train:shuffle()
   mytester:assert(math.abs(isum - train.inputs:sum()) < 0.0000001)
   mytester:assert(math.abs(tsum - train.targets:sum()) < 0.0000001)
   
   -- test inputSize and outputSize
   local isize, tsize = train:isize(), train:tsize()
   mytester:assertTableEq(isize, {1,28,28}, 0.0000001)
   mytester:assert(#tsize == 0)
end


function dl.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dltest)
   mytester:run(tests)
end
