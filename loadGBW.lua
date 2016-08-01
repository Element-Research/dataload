local dl = require 'dataload._env'

local sentence_start = 793470 --"<S>"
local sentence_end = 793471 --"</S>"
local unknown_word = 793469 --"<UNK>"

-- Loads Google Billion Words train, valid and test sets.
-- Since sentences are shuffled, each is seperated by a 0 token.
-- Models should use LookupMaskZero, MaskZero and MaskZeroCriterion.
-- The corpus is derived from the 
-- training-monolingual.tokenized/news.20??.en.shuffled.tokenized data 
-- distributed at http://statmt.org/wmt11/translation-task.html.
-- We use the preprocessing suggested by 
-- https://code.google.com/p/1-billion-word-language-modeling-benchmark
function dl.loadGBW(batchsize, trainfile, datapath, srcurl, verbose)
   -- 1. arguments and defaults
   
   -- the size of the batch is fixed for SequenceLoaders
   batchsize = torch.type(batchsize) == 'table' and batchsize or {batchsize, batchsize, batchsize}
   assert(torch.type(batchsize[1]) == 'number')
   -- path to directory containing Penn Tree Bank dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH, 'BillionWords')
   -- train file can be train_[tiny|small|data].th7
   trainfile = trainfile or 'train_data.th7'
   -- URL from which to download dataset if not found on disk.
   srcurl = srcurl or 'http://lisaweb.iro.umontreal.ca/transfert/lisa/users/leonardn/billionwords.tar.gz'
   -- verbose initialization
   verbose = verbose == nil and true or verbose
   
   local vocabpath = paths.concat(datapath, 'word_map.th7') -- wordid -> wordstring
   local wordfreqpath = paths.concat(datapath, 'word_freq.th7')
   
   -- download file if necessary
   dl.downloadfile(datapath, srcurl, vocabpath)
   dl.decompressfile(datapath, paths.concat(datapath, 'billionwords.tar.gz'), vocabpath)
   
   local ivocab = torch.load(vocabpath)
   local wordfreq = torch.load(wordfreqpath)
   local vocab = {}
   for i,word in ipairs(ivocab) do
      vocab[word] = i
   end
   
   local loaders = {}
   for i,filename in ipairs{trainfile, 'valid_data.th7', 'test_data.th7'} do
      local loader = dl.MultiSequenceGBW(datapath, filename, batchsize[i], verbose)
      -- append vocabulary and such
      loader.vocab = vocab
      loader.ivocab = ivocab
      loader.wordfreq = wordfreq
      table.insert(loaders, loader)
   end
   
   return unpack(loaders)
end

function dl.MultiSequenceGBW(datapath, filename, batchsize, verbose)
   local success, tds = pcall(function() return require "tds" end)
   if not success then
      error"Missing package tds : luarocks install tds"
   end
   
   -- 2. load preprocessed tensor of text
   
   local filepath = paths.concat(datapath, filename)
   local cachepath = filepath:gsub('th7','cache.t7')
   
   local seqs
   if paths.filep(cachepath) then
      if verbose then
         print("loading cached "..cachepath)
      end
      seqs = torch.load(cachepath)
   else
      if verbose then
         print("loading "..filepath)
      end
      -- A torch.tensor with 2 columns. First col is for storing 
      -- start indices of sentences. Second col is for storing the
      -- sequence of words as shuffled sentences. Sentences are
      -- only seperated by the sentence_end delimiter.
       
      local raw = torch.load(filepath)
      
      -- count number of sentences
      
      local nsentence = 0
      local sentencestart = -1
      raw:select(2,1):apply(function(startid)
         if startid ~= sentencestart then
            nsentence = nsentence+1
            sentencestart = startid
         end
      end)
      
      if verbose then
         print"Formatting raw tensor into table of sequences"
      end
      
      -- + nsentence because we are adding the <S> tokens
      local data = torch.LongTensor(raw:size(1)+nsentence)
      seqs = tds.Vec()
      
      local sentencestart = 1
      local rawidx = 0
      local dataidx = 1
      raw:select(2,1):apply(function(startid)
         rawidx = rawidx + 1
         local islastelement = (rawidx == raw:size(1))
         if startid ~= sentencestart or islastelement then
            if islastelement then
               rawidx = rawidx + 1
            end
            
            local size = rawidx - sentencestart + 1 -- + 1 for <S>
            local sequence = data:narrow(1, dataidx, size)
            sequence[1] = sentence_start
            
            sequence:narrow(1,2,size-1):copy(raw[{{sentencestart, rawidx-1},2}])
            
            dataidx = dataidx + size
            sentencestart = startid
            
            assert(sequence[sequence:size(1)] == sentence_end)
            seqs:insert(sequence)
         end
      end)
      
      if verbose then
         print("saving cache "..cachepath)
      end
      torch.save(cachepath, seqs)
   end
      
   -- 3. encapsulate into MultiSequence loader
   
   local loader = dl.MultiSequence(seqs, batchsize)
   
   return loader
end

