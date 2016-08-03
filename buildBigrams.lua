local dl = require 'dataload._env'

-- build a table of bigrams for a dataset where targets is w[t+1] and inputs is w[t]
function dl.buildBigrams(dataset)

   local bigram_map = {}  -- finds all the bigram index and their frequency

   -- generate a information for bigram index and prob.
   for j, input, target in dataset:subiter(1) do
      -- input and target have size : 1 x batchsize
      assert(input:dim() == 2)
      assert(input:size(1) == 1)
      assert(target:dim() == 2)
      assert(target:size(1) == 1)
      
      for i=1,input:size(2) do
         local cur_w = input[{1,i}]
         local next_w = target[{1,i}]
         
         local tab = bigram_map[cur_w]
         if not tab then
            tab = {}
            bigram_map[cur_w] = tab
         end
         
         tab[next_w] = (tab[next_w] or 0) + 1
         
      end

   end
   
   local bigram = {}      -- creates a tensor for index and probability
   local _ = require 'moses'
   
   -- create index and probability using map information
   for wid, map in pairs(bigram_map) do
       local tab = {}
       local len = _.size(map)
       
       tab.index = torch.LongTensor(len)
       tab.prob = torch.FloatTensor(len)
       
       local sum = 0
       local count = 0
       for wid2, freq in pairs(map) do
           tab.index[count+1] = wid2 
           tab.prob[count+1] = freq
           sum = sum + freq
           count = count + 1   
       end
       tab.prob:div(sum)
       bigram[wid] = tab
   end

   return bigram 
end

