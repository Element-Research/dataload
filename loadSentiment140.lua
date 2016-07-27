local dl = require 'dataload._env'

--[[
  Ref. A http://help.sentiment140.com/for-students/
  Load Twitter sentiment analysis dataset (i.e. Sentiment140, ref. A)
  Returns train, valid, test sets

  Description from http://help.sentiment140.com/for-students/
   The data has been processed so that the emoticons are stripped off.
   Also, it's in a regular CSV format.

   Data file format has 6 fields:
   0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
   1 - the id of the tweet (2087)
   2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
   3 - the query (lyx). If there is no query, then this value is NO_QUERY.
   4 - the user that tweeted (robotickilldozr)
   5 - the text of the tweet (Lyx is cool)

  Warning:
   It might give an out-of-memory error to run this script on Torch with LuaJIT.
   Recommended to use Torch with Lua installation
   (please refer to https://github.com/torch/distro for installation steps).
--]]
function dl.loadSentiment140(datapath, minfreq, seqlen, validratio, srcurl, progress)
   -- path to directory containing Twitter dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH , "Twitter")

   -- Drop words with frequency less than minfreq
   minfreq = minfreq or 3

   -- Max Sequence length
   seqlen = seqlen or 50

   -- proportion of training set to use for cross-validation.
   validratio = validratio or 1/8

   -- URL from which to download dataset if not found on disk.
   srcurl = srcurl or 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
   
   -- debug
   progress = progress == nil and true or progress
   
   local cachepath = paths.concat(datapath, 'cache_'..minfreq..'_'..seqlen..'.t7')
   local trainset, testset
   
   if dl.overwrite or not paths.filep(cachepath) then
      
      -- download and decompress the file if necessary
      local testDataFile = paths.concat(datapath, 'testtokens.manual.2009.06.14.csv')
      local trainDataFile = paths.concat(datapath, 'traintokens.1600000.processed.noemoticon.csv')
      if not paths.filep(testDataFile) then
         print('not found ' .. testDataFile .. ', start fresh downloading...')
         local origTrainDataFile = paths.concat(datapath, 'training.1600000.processed.noemoticon.csv')
         local origTestDataFile = paths.concat(datapath, 'testdata.manual.2009.06.14.csv')
         dl.downloadfile(datapath, srcurl, origTestDataFile)
         dl.decompressfile(datapath, paths.concat(datapath, 'trainingandtestdata.zip'), origTestDataFile)

         -- run tokenizer to generate training/testing data in a new CSV format
         print("twokenizing data")
         local cmdstr = 'python twitter/twokenize.py -i ' ..origTestDataFile 
         cmdstr = cmdstr .. ' -o ' ..testDataFile
         local res = sys.execute(cmdstr)
         cmdstr = 'python twitter/twokenize.py -i ' ..origTrainDataFile
         cmdstr = cmdstr .. ' -o ' ..trainDataFile
         res = sys.execute(cmdstr)
      end
      
      -- Load Train File
      if progress then print("Load & processing training data.") end
      local a = torch.Timer()
      
      trainset = dl.Sentiment140Loader(trainDataFile, minfreq, seqlen)
      
      if progress then 
         print("Number of tweets: "..trainset:size())
         print("Vocabulary size: "..#trainset.ivocab) 
         print("Number of occurences replaced with <OOV> token: "..trainset.unigram[1])
         print("Tweet corpus size (in number of tokens): "..trainset.unigram:sum())
      end
      
      -- shuffle so that the training/validation split is balanced
      trainset:shuffle()
      print("trainset set processed in "..a:time().real.."s")
      collectgarbage()

      -- Load Test File      
      testset = dl.Sentiment140Loader(testDataFile, trainset.vocab, seqlen)
      
      testset.ivocab = assert(trainset.ivocab)
      testset.unigram = assert(trainset.unigram)
      
      torch.save(cachepath, {trainset, testset})
   else
      -- load cached training and test sets
      trainset, testset = unpack(torch.load(cachepath))
   end
   
   local trainset, validset = trainset:split(1-validratio)
   
   collectgarbage()
   
   for i, loader in ipairs{trainset, validset, testset} do
   
      function loader:tensor2text(tensor)
         assert(tensor:dim() == 1)
         local text = {}
         for i=1,tensor:size(1) do
            local wordid = tensor[i]
            if wordid > 0 then
               table.insert(text, self.ivocab[wordid])
            end
         end
         return table.concat(text, ' ')
      end
      
      loader.vocab = testset.vocab
      loader.ivocab = testset.ivocab
      loader.unigram = testset.unigram

   end

   return trainset, validset, testset
end

function dl.Sentiment140Loader(filename, vocab, maxlen)
   maxlen = maxlen or 50
   
   local wordfreq, minfreq
   if torch.type(vocab) == 'number' then
      wordfreq = {}
      minfreq = vocab
   else
      assert(maxlen > 0, "Expecting max tweet len at arg 3")
   end
   
   local nline = 0
   local inputstrings, inputsizes, targetindices = {}, {}, {}
   
   for line in io.lines(filename) do
      nline = nline + 1
      local i = 0
      
      for text in line:gmatch('[^",]+') do
         i = i + 1
         if i == 1 then
            targetindices[nline] = tonumber(text)
         elseif i == 6 then
            inputstrings[nline] = text
            
            local len = 0
            for word in text:gmatch('[^%s]+') do 
                if wordfreq then 
                  -- count frequency of words
                  wordfreq[word] = (wordfreq[word] or 0) + 1
               end
               len = len + 1
            end
            
            inputsizes[nline] = len
         end
      end
   end
   
   -- build vocabulary
   
   local ivocab, unigram
   if wordfreq then
      local _ = require 'moses'
      -- make sure ordering is consistent
      local words = _.sort(_.keys(wordfreq))
      
      vocab, ivocab = {}, {}
      vocab['<OOV>'] = 1
      ivocab[1] = '<OOV>'
      local oov = 0
      local wordseq = 1
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
         wordfreq['<OOV>'] = oov
      end
      
      -- make unigram distribution (e.g. for NCE)
      unigram = torch.LongTensor(#ivocab):zero()
      
      for i,word in ipairs(ivocab) do
         unigram[i] = wordfreq[word] or 0
      end
   end
   
   -- encapsulated inputs and targets into tensors
   
   local inputs = torch.LongTensor(nline, maxlen):zero()
   local targets = torch.LongTensor(nline):zero()
   local indx = 1   
   for i=1,nline do
      local inputstring = inputstrings[i]
      local inputsize = inputsizes[i]
      local input = inputs[indx]

      local j = math.max(1, maxlen - inputsize + 1)
      for word in inputstring:gmatch('[^%s]+') do 
         local wordid = vocab[word] or 1
         input[j] = wordid
         j = j + 1
         
         if j > maxlen then
            -- truncate long tweets
            break
         end
      end
      
      local target = targetindices[i]
      
      if target == 0 then
         targets[indx] = 1
         indx = indx + 1
      elseif target == 2 then
         -- targets[indx] = 2 Neutral tweet not considered.
      elseif target == 4 then
         targets[indx] = 2
         indx = indx + 1
      else
         error("Unrecognized target: "..target)
      end
      
   end

   inputs = inputs[{{1, indx-1}}]
   targets = targets[{{1, indx-1}}]
   
   local loader = dl.TensorLoader(inputs, targets)
   loader.vocab = vocab
   
   if wordfreq then
      loader.ivocab = ivocab
      loader.unigram = unigram
   end
   
   return loader, maxlen
end
