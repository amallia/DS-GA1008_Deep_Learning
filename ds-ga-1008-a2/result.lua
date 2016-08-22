require 'xlua'
require 'optim'
require 'image'
require 'nn'
require 'cudnn'
require 'cunn'
require 'nn'

c = require 'trepl.colorize'
if not paths.dirp('monty_python') then
print(c.blue '==>'..'downloading normalized testing data')
os.execute('mkdir '.. "monty_python")
os.execute('wget ' .. 'https://googledrive.com/host/0B5Em-_9MHL-4OEN2ekFSam44amc' .. '; '.. 'mv 0B5Em-_9MHL-4OEN2ekFSam44amc monty_python/test.t7')
print(c.blue '==>'..'downloading model.net')
os.execute('wget ' .. 'https://googledrive.com/host/0B5Em-_9MHL-4SFhFZGJHLTZLRTA' .. '; '.. 'mv 0B5Em-_9MHL-4SFhFZGJHLTZLRTA monty_python/model.net')
else 
print('model found start loading data..')
end

print("loading testing data...")
t  = torch.load('monty_python/test.t7')
test.data = t.data
test.label = t.labels

print("loading model...")
model = torch.load('monty_python/model.net')



function testing_t()
  local confusion = optim.ConfusionMatrix(10)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  res = torch.Tensor((#test.data)[1],10):float():cuda()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,(#test.data)[1],bs do
    local outputs = model:forward(test.data:narrow(1,i,bs):cuda())
    confusion:batchAdd(outputs, test.label:narrow(1,i,bs):cuda())
    res[{{i,i+bs-1},{}}] = outputs
  end
  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)
	return res
end

function prediction(data)
res = {}
for i = 1, (#data)[1] do
pred = data[i]:float()
res[i] = torch.nonzero(pred:eq(pred:max()))[1][1]  
end
return res
end

res = prediction(testing_t())
print("generating predictions ...")
f = io.open('monty_python/predictions.csv', 'w')
f:write('Id'.. ','.. 'Prediction\n' )

for i = 1, #res do
   f:write(i ..','..res[i]..'\n')
end
f:close()
print('finished...')

