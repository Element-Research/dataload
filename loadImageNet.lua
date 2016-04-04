local dl = require 'dataload._env'

-- load ILSVRC2012-14 image classification dataset (A.K.A. ImageNet)
-- http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php
-- Wraps the Large Scale Visual Recognition Challenge 2014 (ILSVRC2014)
-- classification dataset (commonly known as ImageNet). The dataset
-- hasn't changed from 2012-2014.
-- Due to its size, the data first needs to be prepared offline :
-- 1. use scripts/downloadimagenet.lua to download and extract the data
-- 2. use scripts/harmonizeimagenet.lua to harmonize train/valid sets
function dl.loadImageNet(datapath, nthread, loadsize, samplesize, verbose)
   -- 1. arguments and defaults
   
   -- path containing 3 folders : ILSVRC2012_img_train, ILSVRC2012_img_val and metadata
   assert(torch.type(datapath) == 'string' and paths.dirp(datapath), "Expecting path to ILSVRC2012 data at arg one") 
   -- number of threads to use per set
   nthread = nthread or 2
   -- approximate size to load the images to before cropping.
   loadsize = loadsize or {3, 256, 256}
   -- consistent size for cropped patches from loaded images.
   samplesize = samplesize or {3, 224, 224}
   -- verbose initialization
   verbose = verbose == nil and true or verbose

   -- path to training images
   local trainpath = paths.concat(datapath, 'ILSVRC2012_img_train')
   -- path to validation images
   local validpath = paths.concat(datapath, 'ILSVRC2012_img_val')
   -- path to meta data
   local metapath = paths.concat(datapath, 'metadata')
   
   local sortfunc = function(x,y)
      return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
   end
   
   local train = dl.ImageClass(trainpath, loadsize, samplesize, 'sampleTrain', sortfunc, verbose)
   local valid = dl.ImageClass(validpath, loadsize, samplesize, 'sampleTest', sortfunc, verbose)
   
   train = dl.AsyncIterator(train, nthread)
   valid = dl.AsyncIterator(valid, nthread)
   
   local classinfopath = paths.concat(metapath, 'classInfo.th7')
   if paths.filep(classinfopath) then
      local classinfo = torch.load(classinfopath)
      train.classinfo = classinfo
      valid.classinfo = classinfo
   else
      if verbose then
         print("ImageNet: skipping "..classnfopath)
         print("To avoid this message use harmonizeimagenet.lua "..
               "script and pass correct metapath")
      end
   end

   return train, valid
end
