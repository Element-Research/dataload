require 'paths'
require 'xlua'
require 'torchx'
require 'string'
require 'os'
require 'sys'
require 'image'
require 'lfs'

local moses = require 'moses'

local dl = require 'dataload._env'

require 'dataload.config'
require 'dataload.utils'
require 'dataload.DataLoader'
require 'dataload.TensorLoader'
require 'dataload.MNIST'
require 'dataload.test'

return dl
