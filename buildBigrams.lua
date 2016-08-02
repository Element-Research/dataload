local dl = require 'dataload._env'

function length(table) 
 count = 0
 for _, _ in pairs(table) do
   count = count + 1
 end
 return count
end


function dl.buildBigrams(dataset)

local bigram_map = {}  -- finds all the bigram index and their frequency
local bigram = {}      -- creates a tensor for index and probability

-- generate a information for bigram index and prob.
for j, input, _ in dataset:subiter(1) do
   for i = 1, input:size(2)-1 do
       if(bigram_map[input[1][i]] == nil) then
         local tab = {}
         tab[input[1][i+1]] = 1
         bigram_map[input[1][i]] = tab
       else
         if(bigram_map[input[1][i]][input[1][i+1]]== nil) then
             bigram_map[input[1][i]][input[1][i+1]] = 1  
         else
             bigram_map[input[1][i]][input[1][i+1]] = bigram_map[input[1][i]][input[1][i+1]] + 1
         end
       end
   end
end

-- create index and probability using map information
for uniI, map in pairs(bigram_map) do
    local tab = {}
    local len = length(map)
    tab['index'] = torch.LongTensor(len)
    tab['prob'] = torch.FloatTensor(len)
    local sum = 0
    local count = 0
    for bigI, freq in pairs(map) do
        tab['index'][count+1] = bigI 
        tab['prob'][count+1] = freq
        sum = sum + freq
        count = count + 1   
    end
    tab['prob']:div(sum)
    bigram[uniI] = tab
end

return bigram 
end



--[[
require('dataload')
local train, valid, test = dl.loadPTB(20)
local bigram = dl.buildBigrams(train)

print(bigram[1]['index'])
print(bigram[1]['prob'])
]]--
