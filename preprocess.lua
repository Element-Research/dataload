local dl = require 'dataload._env'

-- Returns normalize preprocessing function (PPF)
-- Estimate the per-channel mean/std on training set and caches results
function dl.fitImageNormalize(trainset, nsample, cachepath, verbose, dim, iterator)
   nsample = nsample or 10000
   dim = dim or 2
   iterator = iterator or 'sampleiter'
   local mean, std
   if cachepath and paths.filep(cachepath) then
      local meanstd = torch.load(cachepath)
      mean = meanstd.mean
      std = meanstd.std
      if verbose then
         print('Loaded mean and std from cache.')
      end
   else
      local tm = torch.Timer()
      if verbose then
         print('Estimating the mean,std (per-channel, shared for all pixels) over ' 
               .. nsample .. ' randomly sampled training images')
      end
      
      local batch
      for i, inputs, targets in trainset[iterator](trainset, math.min(32, nsample), nsample) do
         assert(torch.isTensor(inputs))
         mean = mean or torch.zeros(inputs:size(dim))
         std = std or mean:clone()
         for j=1,inputs:size(dim) do
            mean[j] = mean[j] + inputs:select(dim,j):mean()
            std[j] = std[j] + inputs:select(dim,j):std()
         end
      end
      
      for j=1,mean:size(1) do
         mean[j] = mean[j]*32 / nsample
         std[j] = std[j]*32 / nsample
      end
      
      if cachepath then
         local cache = {mean=mean,std=std}
         torch.save(cachepath, cache)
      end
      
      if verbose then
         print('Time to estimate:', tm:time().real)
      end
   end
   
   if verbose then
      print('Mean: ', mean)
      print('Std:', std)
   end
   
   local function ppf(inputs)
      for i=1,inputs:size(dim) do -- channels
         inputs:select(dim,i):add(-mean[i]):div(std[i]) 
      end
      return inputs
   end

   if verbose then
      -- just check if mean/std look good now
      local inputs = trainset:sub(1,100)
      ppf(inputs)
      print('Stats of 100 randomly sampled images after normalizing. '..
            'Mean: ' .. inputs:mean().. ' Std: ' .. inputs:std())
   end
   return ppf
end
