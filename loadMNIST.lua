local dl = require 'dataload._env'

-- Load MNIST, a Handwritten digit classification dataset.
-- Returns train, valid, test sets
function dl.loadMNIST(datapath, validratio, scale, srcurl)
   -- 1. arguments and defaults
   
   -- path to directory containing MNIST dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH, 'mnist')
   -- proportion of training set to use for cross-validation.
   validratio = validratio or 1/6 
   -- scales the values between this range
   scale = scale == nil and {0,1} or scale
   -- URL from which to download dataset if not found on disk.
   srcurl = srcurl or 'https://stife076.files.wordpress.com/2015/02/mnist4.zip'
   
   -- 2. load raw data
   
   -- download and decompress the file if necessary
   local existfile = paths.concat(datapath, 'test.th7')
   dl.downloadfile(datapath, srcurl, existfile)
   dl.decompressfile(datapath, paths.concat(datapath, 'mnist4.zip'), existfile)
   
   -- load train/test files
   local traindata = dl.load(paths.concat(datapath, 'train.th7'))
   local validdata = dl.load(paths.concat(datapath, 'test.th7'))
   
   -- 3. build into TensorLoader
   
   local loaders = {}
   for i,data in ipairs{traindata, validdata} do
      local inputs, targets = data[1], data[2]
      if scale then
         dl.rescale(inputs, unpack(scale))
      end
   
      -- class 0 will have index 1, class 1 index 2, and so on.
      targets:add(1)
      
      -- from bhwc to bchw
      inputs:resize(inputs:size(1), 1, 28, 28)
      
      -- wrap into loader
      local loader = dl.TensorLoader(inputs, targets)
      
      -- set classes
      loader.classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
      
      loaders[i] = loader
   end
   
   -- 4. split into train, valid test

   local train, valid = loaders[1]:split(1-validratio)
   valid.classes = train.classes
   local test = loaders[2]

   return train, valid, test
end
