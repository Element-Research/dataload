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
function dl.loadTwitterSentiment(datapath, validRatio, scale, srcUrl,
                                                         showProgress)
   -- 1. arguments and defaults
   
   -- path to directory containing Twitter dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH, 'twitter')
   -- proportion of training set to use for cross-validation.
   validRatio = validRatio or 1/8
   -- scales the values between this range
   scale = scale == nil and {0,1} or scale
   -- URL from which to download dataset if not found on disk.
   srcUrl = srcUrl or 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
   -- debug
   local showProgress = showProgress or true --false
   
   -- 2. load raw data
   
   -- download and decompress the file if necessary
   local testDataFile = paths.concat(datapath, 
                                     'testtokens.manual.2009.06.14.csv')
   local trainDataFile = paths.concat(datapath,
                             'traintokens.1600000.processed.noemoticon.csv')
   if not paths.filep(testDataFile) then
      print('not found ' .. testDataFile .. ', start fresh downloading...')
      local origTrainDataFile = paths.concat(datapath, 
				'training.1600000.processed.noemoticon.csv')
      local origTestDataFile = paths.concat(datapath,
					    'testdata.manual.2009.06.14.csv')
      dl.downloadfile(datapath, srcUrl, origTestDataFile)
      dl.decompressfile(datapath, paths.concat(datapath,
				 'trainingandtestdata.zip'), origTestDataFile)

      -- run tokenizer to generate training/testing data in a new CSV format
      local cmdstr = 'python twitter/twokenize.py -i ' ..origTestDataFile 
      cmdstr = cmdstr .. ' -o ' ..testDataFile
      local res = sys.execute(cmdstr)
      cmdstr = 'python twitter/twokenize.py -i ' ..origTrainDataFile
      cmdstr = cmdstr .. ' -o ' ..trainDataFile
      res = sys.execute(cmdstr)
   end
   
   -- Load Train File
   if showProgress then print("Load & processing training data.") end
   local trainTweetInfos, trainTweets, allTrainWords, maxTweetLen = 
                                          dl.processTwitterCSV(trainDataFile)
   print(maxTweetLen)
   train.vocab, train.ivocab, train.wordfreq = dl.buildVocab(allTrainWords)
   allTrainWords = nil
   collectgarbage()

   -- Load Test File
   if showProgress then print("Load & processing testing data.") end
   local testTweetInfos, testTweets, allTrainWords, maxTweetLen = 
                         dl.processTwitterCSV(testDataFile, maxTweetLen, false)
   print(maxTweetLen)

   return train, valid, test
end

function dl.loadTwitterCSV(filename, sep, contentsep, showprogress)
   local sep = sep or '","'
   local contentsep = contentsep or ' '
   local showprogress = showprogress or false
   if showprogress then
      print('loading ' .. filename, sep, contentsep)
   end
   local nlines = dl.getNumberOfLines(filename)
   local filelines = io.open(filename):lines()
   local fieldstable, contenttable = {}, {}
   local tablesize=0
   for line in filelines do
      local vs = dl.splitString(line, sep)
      vs[1] = vs[1]:sub(2,#vs[1])
      vs[6] = vs[6]:sub(1,#vs[6]-1)
      vs[6] = dl.splitString(vs[6], contentsep)
      table.insert(fieldstable, vs)
      table.insert(contenttable, vs[6])
      tablesize = tablesize + 1
      if showprogress and math.fmod(tablesize, 100)==0 then
         xlua.progress(tablesize, nlines)
      end
   end
   if showprogress then
      print('# items loaded ' .. tablesize)
   end
   return fieldstable, contenttable
end

function dl.processTwitterCSV(filename, maxTweetLen, returnAllWords)
   local maxTweetLen = maxTweetLen or 0
   local tweetsInfo = {}
   local tweets = {}
   local allWords = {}
   local returnAllWords = returnAllwords or true
   local filelines = io.open(filename):lines()
   for line in filelines do
      local infoTokens = string.split(line, ',')

      local tweetInfo = {}
      tweetInfo['polarity'] = tonumber(infoTokens[1]:tosymbol())
      tweetInfo['id'] = tonumber(infoTokens[2]:tosymbol())
      local tweetId = tweetInfo['id']
      tweetInfo['date'] = infoTokens[3]:tosymbol()
      tweetInfo['query'] = infoTokens[4]:tosymbol()
      tweetInfo['user'] = infoTokens[5]:tosymbol()

      local tweet = {}
      local tempLine = ''
      for i=6,#infoTokens do
         if i==#infoTokens then
            tempLine = tempLine .. infoTokens[i]
         else
            tempLine = tempLine .. infoTokens[i] .. ','
         end
      end
      tempLine = tempLine:sub(2, -2)
      for word in tempLine:gmatch("([^%s]+)") do
         table.insert(tweet, word)
         if returnAllWords then table.insert(allWords, word) end
      end
      if maxTweetLen < #tweet then maxTweetLen = #tweet end
      tweetsInfo[tweetId] = tweetInfo
      tweets[tweetId] = tweet
   end
   return tweetsInfo, tweets, allWords, maxTweetLen
end

function dl.buildContentVocab(contenttable)
   local tokens = contenttable --stringx.split(text)
   vocab, ivocab, wordfreq = dl.buildVocab(tokens)
   return vocab, ivocab, wordfreq
end
