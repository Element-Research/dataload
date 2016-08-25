local dl = require 'dataload._env'

-- Load CIFAR10, a subset of the 80 million tiny images dataset.
-- Returns train, valid, test sets
function dl.loadCIFAR10(datapath, validratio, scale, srcurl)
   -- 1. arguments and defaults
   
   -- path to directory containing CIFAR10 dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH, 'cifar10')
   -- proportion of training set to use for cross-validation.
   validratio = validratio or 1/6 
   -- scales the values between this range
   scale = scale == nil and {0,1} or scale
   -- URL from which to download dataset if not found on disk.
   srcurl = srcurl or 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   
   -- 2. load raw data
   
   -- download and decompress the file if necessary
   local existartfile = paths.concat(datapath, 'cifar-10-torch.tar.gz')
   dl.downloadfile(datapath, srcurl, existartfile)
   local existtorchfile = paths.concat(datapath, 'cifar-10-batches-t7', 'test_batch.t7')
   dl.decompressfile(datapath, paths.concat(datapath, 'cifar-10-torch.tar.gz'), existtorchfile)
   
   -- load train/test files
   
   local rawtraindata = {data = {},
                         batch_label = {},
                         labels = {}}
   for i=1,5 do
      local batch = dl.load(paths.concat(datapath, 'cifar-10-batches-t7', 'data_batch_' ..  i .. '.t7'))
      rawtraindata.data[i] = batch.data
      rawtraindata.batch_label[i] = batch.batch_label
      rawtraindata.labels[i] = batch.labels
   end
      
   local traindata = {data = torch.concat(rawtraindata.data, 2), 
                      batch_label = torch.concat(rawtraindata.batch_label, 2), 
                      labels = torch.concat(rawtraindata.labels, 2)}
   local validdata = dl.load(paths.concat(datapath, 'cifar-10-batches-t7', 'test_batch.t7'))

   -- 3. build into TensorLoader
   
   local loaders = {}
   for i,batch in ipairs{traindata, validdata} do
      local inputs, targets = batch.data:transpose(1, 2):float(),
                              batch.labels:squeeze():float()
      if scale then
         dl.rescale(inputs, unpack(scale))
      end
   
      -- class 0 will have index 1, class 1 index 2, and so on.
      targets:add(1)
      
      -- from bhwc to bchw
      inputs:resize(inputs:size(1), 3, 32, 32)
      
      -- wrap into loader
      local loader = dl.TensorLoader(inputs, targets)
      
      -- set classes
      loader.classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                        'frog', 'horse', 'ship', 'truck'}
      
      loaders[i] = loader
   end
   
   -- 4. split into train, valid test

   local train, valid = loaders[1]:split(1-validratio)
   valid.classes = train.classes
   local test = loaders[2]

   return train, valid, test
end
