local dl = require 'dataload._env'

-- environment variables
dl.TORCH_PATH = os.getenv('TORCH_DATA_PATH') or os.getenv('HOME')
-- where the downloaded data will be saved
dl.DATA_PATH = os.getenv('DEEP_DATA_PATH') or paths.concat(dl.TORCH_PATH, 'data')
-- where to save your models
dl.SAVE_PATH = os.getenv('DEEP_SAVE_PATH') or paths.concat(dl.TORCH_PATH, 'save')

paths.mkdir(dl.SAVE_PATH)
paths.mkdir(dl.DATA_PATH)
