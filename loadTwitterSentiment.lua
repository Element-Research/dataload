local dl = require 'dataload._env'

--[[
  Load Twitter sentiment analysis dataset
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
function dl.loadTwitterSentiment(datapath, minFreq, seqLen, validRatio, srcUrl, showProgress)
   -- path to directory containing Twitter dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH , "Twitter")

   -- Drop words with frequency less than minFreq
   minFreq = minFreq or 0

   -- Max Sequence length
   seqLen = seqLen or 0 -- If 0 then use maxTweetLen

   -- proportion of training set to use for cross-validation.
   validRatio = validRatio or 1/8

   -- URL from which to download dataset if not found on disk.
   srcUrl = srcUrl or 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
   -- debug
   showProgress = showProgress == nil and true or showProgress
   
   local cachepath = paths.concat(datapath, 'cache_'..minFreq..'_'..seqLen..'.t7')
   
   local trainset, validset
   
   if dl.overwrite or not paths.filep(cachepath) then
      
      -- download and decompress the file if necessary
      local testDataFile = paths.concat(datapath, 'testtokens.manual.2009.06.14.csv')
      local trainDataFile = paths.concat(datapath, 'traintokens.1600000.processed.noemoticon.csv')
      if not paths.filep(testDataFile) then
         print('not found ' .. testDataFile .. ', start fresh downloading...')
         local origTrainDataFile = paths.concat(datapath, 'training.1600000.processed.noemoticon.csv')
         local origTestDataFile = paths.concat(datapath, 'testdata.manual.2009.06.14.csv')
         dl.downloadfile(datapath, srcUrl, origTestDataFile)
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
      if showProgress then print("Load & processing training data.") end
      local a = torch.Timer()
      local trainTweetsInfo, trainTweets, allTrainWords, maxTweetLen  = dl.processTwitterCSV(trainDataFile)
      print("buildVocab", a:time().real)
      local vocab, ivocab, wordFreq = dl.buildVocab(allTrainWords, minFreq)
      print(a:time().real)
      allTrainWords = nil
      collectgarbage()

      -- Load Test File
      if showProgress then print("Load & processing testing data.") end
      local testTweetsInfo, testTweets, allTestWords, maxTweetLen = dl.processTwitterCSV(testDataFile, maxTweetLen, false)

      -- Convert Tweets to Tensor using vocabulary
      if showProgress then print("Tweet/text to vector") end
      if seqLen == 0 then seqLen = maxTweetLen end
      local trainTweetsTensor = torch.LongTensor(#trainTweets, seqLen):zero()
      local trainTweetsTarget = torch.IntTensor(#trainTweets):zero()
      for i, tweet in pairs(trainTweets) do
         for j, token in pairs(tweet) do
            trainTweetsTensor[i][j] = vocab[token] or 1 -- 1 -> '<OOV>'
            if j == seqLen then break end
         end
         if trainTweetsInfo[i]['polarity'] == 0 then
            trainTweetsTarget[i] = 1
         elseif trainTweetsInfo[i]['polarity'] == 2 then
            trainTweetsTarget[i] = 2
         elseif trainTweetsInfo[i]['polarity'] == 4 then
            trainTweetsTarget[i] = 3
         end
      end

      local testTweetsTensor = torch.LongTensor(#testTweets, seqLen):zero()
      local testTweetsTarget = torch.IntTensor(#testTweets):zero()
      for i, tweet in pairs(testTweets) do
         for j, token in pairs(tweet) do
            testTweetsTensor[i][j] = vocab[token] or 1 -- 1 -> '<OOV>'
            if j == seqLen then break end
         end
         if testTweetsInfo[i]['polarity'] == 0 then
            testTweetsTarget[i] = 1
         elseif testTweetsInfo[i]['polarity'] == 2 then
            testTweetsTarget[i] = 2
         elseif testTweetsInfo[i]['polarity'] == 4 then
            testTweetsTarget[i] = 3
         end
      end

      -- Unigram
      local unigram = torch.LongTensor(#ivocab):zero()
      -- make unigram distribution for NCE
      for i,word in ipairs(ivocab) do
         unigram[i] = wordFreq[word] or 0
      end

      -- Wrap with TensorLoader class
      trainset = dl.TensorLoader(trainTweetsTensor, trainTweetsTarget)
      testset = dl.TensorLoader(testTweetsTensor, testTweetsTarget)

      -- Adding vocab, ivocab and unigram to sets      
      testset.vocab = vocab
      testset.ivocab = ivocab
      testset.unigram = unigram
      
      torch.save(cachepath, {trainset, testset})
   else
      
      trainset, testset = unpack(torch.load(cachepath))
      
   end
   
   local trainset, validset = trainset:split(1-validRatio)
   
   trainset.vocab = testset.vocab
   trainset.ivocab = testset.ivocab
   trainset.unigram = testset.unigram
      
   validset.vocab = testset.vocab
   validset.ivocab = testset.ivocab
   validset.unigram = testset.unigram
   collectgarbage()

   return trainset, validset, testset
end

function dl.processTwitterCSV(filename, maxTweetLen, returnAllWords)
   local maxTweetLen = maxTweetLen or 0
   local tweetsInfo = {}
   local tweets = {}
   local allWords = {}
   local returnAllWords = (returnAllWords==nil and true) or returnAllWords
   local filelines = io.open(filename):lines()
   for line in filelines do
      line = line:gsub('"', '')
      local infoTokens = string.split(line, ',')
      local tweetInfo = {}
      tweetInfo['polarity'] = tonumber(infoTokens[1])
      tweetInfo['id'] = tonumber(infoTokens[2])
      local tweetId = tweetInfo['id']
      tweetInfo['date'] = infoTokens[3]
      tweetInfo['query'] = infoTokens[4]
      tweetInfo['user'] = infoTokens[5]

      local tweet = {}
      local tempLine = ''
      for i=6,#infoTokens do
         if i==#infoTokens then
            tempLine = tempLine .. infoTokens[i]
         else
            tempLine = tempLine .. infoTokens[i] .. ','
         end
      end

      for word in tempLine:gmatch("([^%s]+)") do
         table.insert(tweet, word)
         if returnAllWords then table.insert(allWords, word) end
      end
      if maxTweetLen < #tweet then maxTweetLen = #tweet end
      table.insert(tweetsInfo, tweetInfo)
      table.insert(tweets, tweet)
   end
   return tweetsInfo, tweets, allWords, maxTweetLen
end
