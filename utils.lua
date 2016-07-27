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
function dl.decompressfile(dstdir, srczip, existfile, verbose, dstfile)
   paths.mkdir(dstdir)
   
   if (not existfile) or (not paths.filep(existfile)) then
      dl.withcwd(dstdir,
         function()
            if verbose then 
               print("decompressing file: ", srczip) 
            end
            if string.find(srczip, ".zip") then
               dl.unzip(srczip, dstfile)
            elseif string.find(srczip, ".tar") or string.find(srczip, ".tgz") then
               dl.untar(srczip, dstfile)
            elseif string.find(srczip, ".gz") or string.find(srczip, ".gzip") then
               dl.gunzip(srczip, dstfile)
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

-- this can be inefficent, I wouldn't use it for datasets larger than PTB
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
   
   local _ = require 'moses'
   -- make sure ordering is consistent
   local words = _.sort(_.keys(wordfreq))
   
   local oov = 0
   for i, word in ipairs(words) do
      local freq = wordfreq[word]
      if freq >= minfreq then
         wordseq = wordseq + 1
         vocab[word] = wordseq
         ivocab[wordseq] = word
      else
         oov = oov + freq
      end
   end
   
   if oov > 0 then
      wordseq = wordseq + 1
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

function dl.splitString(str,sep)
   local sep, fields = sep or ":", {}
   local pattern = string.format("([^%s]+)", sep)
   str:gsub(pattern, function(c) fields[#fields+1] = c end)
   return fields
end

function dl.getNumberOfLines(filename)
   local ctr = 0
   for _ in io.lines(filename) do
      ctr = ctr +1
   end
   return ctr
end

-- misc.

function dl.hostname()
   local f = io.popen ("/bin/hostname")
   if not f then 
      return 'localhost'
   end
   local hostname = f:read("*a") or ""
   f:close()
   hostname =string.gsub(hostname, "\n$", "")
   return hostname
end

-- Generates a globally unique identifier.
-- If a namespace is provided it is concatenated with 
-- the time of the call, and the next value from a sequence
-- to get a pseudo-globally-unique name.
-- Otherwise, we concatenate the linux hostname
local counter = 1
function dl.uniqueid(namespace, separator)
   local separator = separator or ':'
   local namespace = namespace or dl.hostname()
   local uid = namespace..separator..os.time()..separator..counter
   counter = counter + 1
   return uid
end

-- table

--http://lua-users.org/wiki/TableUtils
function table.val_to_str ( v )
   if "string" == type( v ) then
      v = string.gsub( v, "\n", "\\n" )
      if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
         return "'" .. v .. "'"
      end
      return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
   else
      return "table" == type( v ) and table.tostring( v ) or tostring( v )
   end
end

function table.key_to_str ( k )
   if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
      return k
   else
      return "[" .. table.val_to_str( k ) .. "]"
   end
end

function table.tostring(tbl, newline, sort)
   local result, done = {}, {}
   for k, v in ipairs( tbl ) do
      table.insert( result, table.val_to_str( v ) )
      done[ k ] = true
   end
   local s = "="
   if newline then
      s = " : "
   end
   for k, v in pairs( tbl ) do
      if not done[ k ] then
         local line = table.key_to_str( k ) .. s .. table.val_to_str( v )
         table.insert(result, line)
      end
   end
   if sort then
      local _ = require 'moses'
      _.sort(result)
   end
   local res
   if newline then
      res = "{\n   " .. table.concat( result, "\n   " ) .. "\n}"
   else
      res = "{" .. table.concat( result, "," ) .. "}"
   end
   return res
end

function table.print(tbl)
   assert(torch.type(tbl) == 'table', "expecting table")
   print(table.tostring(tbl, true, true))
end
