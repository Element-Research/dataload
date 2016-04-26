local dl = require 'dataload._env'

--[[
  Load Twitter sentiment analysis dataset
  Returns train, valid, test sets

  Description from http://help.sentiment140.com/for-students/
   The data has been processed so that the emoticons are stripped off. Also, it's in a regular CSV format.
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
function dl.loadTwitterSentiment(datapath, validratio, scale, srcurl, showprogress)
   -- 1. arguments and defaults
   
   -- path to directory containing Twitter dataset on disk
   datapath = datapath or paths.concat(dl.DATA_PATH, 'twitter')
   -- proportion of training set to use for cross-validation.
   validratio = validratio or 1/8
   -- scales the values between this range
   scale = scale == nil and {0,1} or scale
   -- URL from which to download dataset if not found on disk.
   srcurl = srcurl or 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
   -- debug
   local showprogress = showprogress or true --false
   
   -- 2. load raw data
   
   -- download and decompress the file if necessary
   local testdatafile = paths.concat(datapath, 'testtokens.manual.2009.06.14.csv')
   local traindatafile = paths.concat(datapath, 'traintokens.1600000.processed.noemoticon.csv')
   if not paths.filep(testdatafile) then
      print('not found ' .. testdatafile .. ', start fresh downloading...')
      local origtraindatafile = paths.concat(datapath, 'training.1600000.processed.noemoticon.csv')
      local origtestdatafile = paths.concat(datapath, 'testdata.manual.2009.06.14.csv')
      dl.downloadfile(datapath, srcurl, origtestdatafile)
      dl.decompressfile(datapath, paths.concat(datapath, 'trainingandtestdata.zip'), origtestdatafile)
      -- run tokenizer to generate training/testing data in a new CSV format this script requires
      local cmdstr = 'python twitter/twokenize.py -i ' ..origtestdatafile.. ' -o ' ..testdatafile
      local res = sys.execute(cmdstr)
      cmdstr = 'python twitter/twokenize.py -i ' ..origtraindatafile.. ' -o ' ..traindatafile
      res = sys.execute(cmdstr)
   end
   
   -- load train file
   local traindata, traincontent = dl.loadTwitterCSV(traindatafile, '","', ' ', showprogress)
   
   -- 3. split into train, valid test
   print('build training vocabulary')
   local train, trainwords = {}, {}
   local trainnum = math.floor((1-validratio)*#traindata)
   for i = 1,trainnum do
      for j = 1,#traincontent[i] do
         table.insert(trainwords, traincontent[i][j])
      end
      if showprogress and math.fmod(i, 100)==0 then
         xlua.progress(i, trainnum)
      end
   end
   local train_vocab, train_ivocab, train_wordfreq = dl.buildVocab(trainwords)
   trainwords = nil
   collectgarbage()
   for i = 1,trainnum do
      table.insert(train, traindata[i])
      if showprogress and math.fmod(i, 100)==0 then
         xlua.progress(i, trainnum)
      end
   end
   train.vocab, train.ivocab, train.wordfreq = train_vocab, train_ivocab, train_wordfreq 
   collectgarbage()
    
   if showprogress then
      print('build validation vocabulary')
   end
   local valid, validwords = {}, {}
   for i = trainnum+1,#traindata do
      table.insert(valid, traindata[i])
      for j = 1,#traincontent[i] do
         table.insert(validwords, traincontent[i][j])
      end
      if showprogress and math.fmod(i, 100)==0 then
         xlua.progress(i, #traindata)
      end
   end
   valid.vocab, valid.ivocab, valid.wordfreq = dl.buildVocab(validwords)
   validwords = nil
   traincontent = nil
   collectgarbage()

   -- load test file
   local testdata, testcontent = dl.loadTwitterCSV(testdatafile, '","', ' ', showprogress)
   if showprogress then
      print('build testing vocabulary')
   end
   local test, testwords = testdata, {}
   for i,tc in ipairs(testcontent) do
      for j = 1,#tc do
         table.insert(testwords, tc[j])
      end
      if showprogress and math.fmod(i, 100)==0 then
         xlua.progress(i, #testcontent)
         --print(i, #testcontent[i])
      end
   end
   test.vocab,  test.ivocab,  test.wordfreq  = dl.buildVocab(testwords)
   print(test.vocat, test.ivocat, test.wordfreq)
   testwords = nil
   testcontent = nil
   collectgarbage()

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

function dl.buildContentVocab(contenttable)
   local tokens = contenttable --stringx.split(text)
   vocab, ivocab, wordfreq = dl.buildVocab(tokens)
   return vocab, ivocab, wordfreq
end
