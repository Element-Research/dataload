# dataload

A collection of Torch dataset loaders. 
The library provides the following generic data loader classes :
 
 * [DataLoader](#dl.DataLoader) : an abstract class inherited by the following classes;
 * [TensorLoader](#dl.TensorLoader) : for tensor or nested (i.e. tables of) tensor datasets;
 * [ImageClass](#dl.ImageClass) : for image classification datasets stored in a flat folder structure;
 * [AsyncIterator](#dl.AsyncIterator) : decorates a `DataLoader` for asynchronou multi-threaded iteration.

The library also provides functions for downloading and wrapping 
specific datasets using the above loaders :

 * loadMNIST

<a name='dl.DataLoader'></a>
## DataLoader

```lua
dataloader = dl.DataLoader()
``` 

An abstract class inherited by all `DataLoader` instances. 
It wraps a data set to provide methods for accessing
`inputs` and `targets`. The data itself may be loaded from disk or memory.

### [n] size()

Returns the number of samples in the `dataloader`.

### [size] isize([excludedim])

Returns the `size` of `inputs`. When `excludedim` is 1 (the default), 
the batch dimension is excluded from `size`. 
When `inputs` is a tensor, the returned `size` is 
a table of numbers. When it is a table of tensors, the returned `size` 
is a table of table of numbers.

### [size] tsize([excludedim])

Returns the `size` of `targets`. When `excludedim` is 1 (the default), 
the batch dimension is excluded from `size`. 
When `targets` is a tensor, the returned `size` is 
a table of numbers. When it is a table of tensors, the returned `size` 
is a table of table of numbers. 

<a name='dl.DataLoad.index'></a>
### [inputs, targets] index(indices, [inputs, targets])

Returns `inputs` and `targets` containing samples indexed by `indices`.

So for example :

```lua
indices = torch.LongTensor{1,2,3,4,5}
inputs, targets = dataloader:index(indices)
``` 

would return a batch of `inputs` and `targets` containing samples 1 through 5.
When `inputs` and `targets` are provided as arguments, they are used as 
memory buffers for the returned `inputs` and `targets`, 
i.e. their allocated memory is reused.

<a name='dl.DataLoad.sample'></a>
### [inputs, targets] sample(batchsize, [inputs, targets])

Returns `inputs` and `targets` containing `batchsize` random samples.
This method is equivalent to : 

```lua
indices = torch.LongTensor(batchsize):random(1,dataloader:size())
inputs, targets = dataloader:index(indices)
``` 

<a name='dl.DataLoad.sub'></a>
### [inputs, targets] sub(start, stop, [inputs, targets])

Returns `inputs` and `targets` containing `stop-start+1` samples between `start` and `stop`.
This method is equivalent to : 

```lua
indices = torch.LongTensor():range(start, stop)
inputs, targets = dataloader:index(indices)
``` 
  
### shuffle()

Internally shuffles the `inputs` and `targets`. Note that not all 
subclasses support this method.

### [ds1, ds2] split(ratio)

Splits the `dataloader` into two new `DataLoader` instances
where `ds1` contains the first `math.floor(ratio x dataloader:size())` samples,
and `ds2` contains the remainder. 
Useful for splitting a training set into a new training set and validation set.

<a name='dl.DataLoad.subiter'></a>
### [iterator] subiter([batchsize, epochsize, ...])

Returns an iterator over a validation and test sets.
Each iteration returns 3 values : 
 
 * `k` : the number of samples processed so far. Each iteration returns a maximum of `batchsize` samples.
 * `inputs` : a tensor (or nested table thereof) containing a maximum of `batchsize` inputs.
 * `targets` : a tensor (or nested table thereof) containing targets for the commensurate inputs. 
 
The iterator will return batches of `inputs` and `targets` of size at most `batchsize` until 
`epochsize` samples have been returned.

Note that the default implementation of this iterator is to call [sub](#dl.DataLoad.sub) for each batch.
Sub-classes may over-write this behavior.

Example :

```lua
local dl = require 'dataload'

inputs, targets = torch.range(1,5), torch.range(1,5)
dataloader = dl.TensorLoader(inputs, targets)

local i = 0
for k, inputs, targets in dataloader:subiter(2,6) do
   i = i + 1
   print(string.format("batch %d, nsampled = %d", i, k))
   print(string.format("inputs:\n%stargets:\n%s", inputs, targets))
end
``` 

Output :

```lua
batch 1, nsampled = 2	
inputs:
 1
 2
[torch.DoubleTensor of size 2]
targets:
 1
 2
[torch.DoubleTensor of size 2]
	
batch 2, nsampled = 4	
inputs:
 3
 4
[torch.DoubleTensor of size 2]
targets:
 3
 4
[torch.DoubleTensor of size 2]
	
batch 3, nsampled = 5	
inputs:
 5
[torch.DoubleTensor of size 1]
targets:
 5
[torch.DoubleTensor of size 1]
	
batch 4, nsampled = 6	
inputs:
 1
[torch.DoubleTensor of size 1]
targets:
 1
[torch.DoubleTensor of size 1]
``` 

Note how the last two batches are of size 1 while those before are of size `batchsize = 2`.
The reason for this is that the `dataloader` only has 5 samples.  
So the last batch is split between the last sample and the first.

<a name='dl.DataLoad.sampleiter'></a>
### [iterator] sampleiter([batchsize, epochsize, ...])

Returns an iterator over a training set. 
Each iteration returns 3 values : 
 
 * `k` : the number of samples processed so far. Each iteration returns a maximum of `batchsize` samples.
 * `inputs` : a tensor (or nested table thereof) containing a maximum of `batchsize` inputs.
 * `targets` : a tensor (or nested table thereof) containing targets for the commensurate inputs. 
 
The iterator will return batches of `inputs` and `targets` of size at most `batchsize` until 
`epochsize` samples have been returned.

Note that the default implementation of this iterator is to call [sample](#dl.DataLoad.sample) for each batch.
Sub-classes may over-write this behavior.

Example :

```lua
local dl = require 'dataload'

inputs, targets = torch.range(1,5), torch.range(1,5)
dataloader = dl.TensorLoader(inputs, targets)

local i = 0
for k, inputs, targets in dataloader:sampleiter(2,6) do
   i = i + 1
   print(string.format("batch %d, nsampled = %d", i, k))
   print(string.format("inputs:\n%stargets:\n%s", inputs, targets))
end
``` 

Output :

```lua
batch 1, nsampled = 2	
inputs:
 1
 2
[torch.DoubleTensor of size 2]
targets:
 1
 2
[torch.DoubleTensor of size 2]
	
batch 2, nsampled = 4	
inputs:
 4
 2
[torch.DoubleTensor of size 2]
targets:
 4
 2
[torch.DoubleTensor of size 2]
	
batch 3, nsampled = 6	
inputs:
 4
 1
[torch.DoubleTensor of size 2]
targets:
 4
 1
[torch.DoubleTensor of size 2]
``` 


### reset()

Resets all internal counters such as those used for iterators.
Called by AsyncIterator before serializing the `DataLoader` to threads.

### collectgarbage()

Collect garbage every `self.gccdelay` times this method is called.

### [copy] clone()

Returns a deep `copy` clone of `self`.

<a name='dl.TensorLoader'></a>
## TensorLoader

```lua
dataloader = dl.TensorLoader(inputs, targets) 
``` 

The `TensorLoader` can be used to encapsulate tensors of `inputs` and `targets`.
As an example, consider a dummy `3 x 8 x 8` image classification dataset consisting of 1000 samples and 10 classes: 

```lua
inputs = torch.randn(1000, 3, 8, 8)
targets = torch.LongTensor(1000):random(1,10)
dataloader = dl.TensorLoader(inputs, targets)
``` 

The `TensorLoader` can also be used to encapsulate nested tensors of `inputs` and `targets`.
It uses recursive functions to handle nestings of arbitrary depth. As an example, let us 
modify the above example to include `x,y` GPS coordinates in the `inputs` and 
a parallel set of classification `targets` (7 classes): 

```lua
inputs = {torch.randn(1000, 3, 8, 8), torch.randn(1000, 2)}
targets = {torch.LongTensor(1000):random(1,10), torch.LongTensor(1000):random(1,7)}
dataloader = dl.TensorLoader(inputs, targets)
``` 

<a name='dl.ImageClass'></a>
## ImageClass

```lua
dataloader = dl.ImageClass(datapath, loadsize, [samplesize, samplefunc, sortfunc, verbose])
``` 

For loading an image classification data set stored in a flat folder structure :

```
(datapath)/(classdir)/(imagefile).(jpg|png|etc)
``` 

So directory `classdir` is expected to contain the all images belonging to that class.
All image files are indexed into an efficient `CharTensor` during initialization.
Images are only loaded into `inputs` and `targets` tensors upon calling 
batch sampling methods like [index](#dl.DataLoad.index), [sample](#dl.DataLoad.index) and [sub](#dl.DataLoad.index).

Note that for asynchronous loading of images (i.e. loading batches of images in different threads), 
the `ImageClass` loader can be decorated with an [AsyncIterator](#dl.AsyncIterator). 
Images on disk can have different height, width and number of channels. 

Constructor arguments are as follows :
 
 * `datapath` : one or many paths to directories of images;
 * `loadsize` : initialize size to load the images to. Example : `{3, 256, 256}`;
 * `samplesize` : consistent sample size to resize the images to. Defaults to `loadsize`;
 * `samplefunc` : `function f(self, dst, path)` used to create a sample(s) from an image path. Stores them in `CharTensor` `dst`. Strings `"sampleDefault"` (the default), `"sampleTrain"` or `"sampleTest"` can also be provided as they refer to existing functions
 * `verbose` : display verbose message (default is `true`);
 * `sortfunc` : comparison operator used for sorting `classdir` to get class indices. Defaults to the `<` operator.

<a name='dl.AsyncIterator'></a>
## AsyncIterator

<a name='dl.loadMNIST'></a>
## loadMNIST
