require 'paths'
require 'xlua'
require 'torchx'
require 'string'
require 'os'
require 'sys'
require 'image'
require 'lfs'

-- these actually return local variables but we will re-require them
-- when needed. This is just to make sure they are loaded.
require 'moses'
require 'ffi'

local dl = require 'dataload._env'

require 'dataload.config'
require 'dataload.utils'
require 'dataload.DataLoader'
require 'dataload.TensorLoader'
require 'dataload.ImageClass'
require 'dataload.MNIST'
require 'dataload.test'

return dl
