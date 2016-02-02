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
            local protocol, scpurl, filename = url:match('(.-)://(.*)/(.-)$')
            if protocol == 'scp' then
                os.execute(string.format('%s %s %s', 'scp', scpurl .. '/' .. filename, filename))
            else
                os.execute('wget ' .. url)
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
