local dl = require 'dataload._env'

-- Temporarily changes the current working directory to call fn, 
-- returning its result.
function dl.withcwd(path, fn)
   local curdir = lfs.currentdir()
   lfs.chdir(path)
   local res = fn()
   lfs.chdir(curdir)
   return res
end

-- Downloads srcurl into dstdir.
-- If existfile is provided then 
-- download only if that file doesn't exist
function dl.downloadfile(dstdir, srcurl, existfile)
   if (not existfile) or (not paths.filep(existfile)) then
      paths.mkdir(dstdir)
      dl.withcwd(
         dstdir, 
         function() 
            local protocol, scpurl, filename = srcurl:match('(.-)://(.*)/(.-)$')
            if protocol == 'scp' then
                os.execute(string.format('%s %s %s', 'scp', scpurl .. '/' .. filename, filename))
            else
                os.execute('wget ' .. srcurl)
            end
         end
      )
   end
   return dstdir
end

-- Decompress a .tar, .tgz or .tar.gz file.
function dl.untar(srcpath, dstpath)
   local dstpath = dstpath or '.'
   paths.mkdir(dstpath)
   if srcpath:match("%.tar$") then
      os.execute('tar -xvf ' .. srcpath .. ' -C ' .. dstpath)
   else
      os.execute('tar -xvzf ' .. srcpath .. ' -C ' .. dstpath)
   end
end

-- unzip a .zip file
function dl.unzip(srcpath, dstpath)
   local dstpath = dstpath or '.'
   paths.mkdir(dstpath)
   os.execute('unzip ' .. srcpath .. ' -d ' .. dstpath)
end

-- gunzip a .gz file
function dl.gunzip(srcpath, dstpath)
   assert(not dstpath, "destination path not supported with gunzip")
   os.execute('gunzip ' .. srcpath)
end

-- Decompresses srczip into dstdir.
-- If existfile is provided then 
-- decompress only if that file doesn't exist.
-- Supported extensions for srczip are : .zip, .tar, .tgz, .gz, .gzip
function dl.decompressfile(dstdir, srczip, existfile, verbose)
   paths.mkdir(dstdir)
   
   if (not existfile) or (not paths.filep(existfile)) then
      dl.withcwd(dstdir,
         function()
            if verbose then 
               print("decompressing file: ", srczip) 
            end
            if string.find(srczip, ".zip") then
               dl.unzip(srczip)
            elseif string.find(srczip, ".tar") or string.find(srczip, ".tgz") then
               dl.untar(srczip)
            elseif string.find(srczip, ".gz") or string.find(srczip, ".gzip") then
               dl.gunzip(srczip)
            else
               error("Don't know how to decompress file: ", srczip)
            end
         end
      )
   end
end

-- loads data as ascii, if that doesn't work, loads as binary.
function dl.load(path)
   local success, data = pcall(function() return torch.load(path, "ascii") end)
   if not success then
      data = torch.load(path, "binary")
   end
   return data
end


function dl.rescale(data, min, max, dmin, dmax)
   local range = max - min
   local dmin = dmin or data:min()
   local dmax = dmax or data:max()
   local drange = dmax - dmin

   data:add(-dmin)
   data:mul(range)
   data:mul(1/drange)
   data:add(min)
   
   return data
end

function dl.binarize(x, threshold)
   x[x:lt(threshold)] = 0
   x[x:ge(threshold)] = 1
   return x
end

-- text utility functions

function dl.buildVocab(tokens, minfreq)
   assert(torch.type(tokens) == 'table', 'Expecting table')
   assert(torch.type(tokens[1]) == 'string', 'Expecting table of strings')
   minfreq = minfreq or -1
   assert(torch.type(minfreq) == 'number')
   local wordfreq = {}
   
   for i=1,#tokens do
      local word = tokens[i]
      wordfreq[word] = (wordfreq[word] or 0) + 1
   end
   
   local vocab, ivocab = {}, {}
   local wordseq = 0
   
   local oov = 0
   for word, freq in pairs(wordfreq) do
      if freq >= minfreq then
         wordseq = wordseq + 1
         vocab[word] = wordseq
         ivocab[wordseq] = word
      else
         oov = oov + freq
      end
   end
   
   if oov > 0 then
      wordseq = wordfreq + 1
      wordfreq['<OOV>'] = oov
      vocab['<OOV>'] = wordseq
      ivocab[wordseq] = '<OOV>'
   end
   
   return vocab, ivocab, wordfreq
end

function dl.text2tensor(tokens, vocab)
   local oov = vocab['<OOV>']
   
   local tensor = torch.IntTensor(#tokens):fill(0)
   
   for i, word in ipairs(tokens) do
      local wordid = vocab[word] 
      
      if not wordid then
         assert(oov)
         wordid = oov
      end
      
      tensor[i] = wordid
   end
   
   return tensor
end

