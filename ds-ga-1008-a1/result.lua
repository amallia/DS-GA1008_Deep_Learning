require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'csvigo'



print '==> processing options'
cmd = torch.CmdLine()
cmd:option('-size', 'full', 'how many samples do we load: small | full')
cmd:text()
opt = cmd:parse(arg or {})

model = torch.load('model.net')

--classes = {'1','2','3','4','5','6','7','8','9','0'}
--confusion = optim.ConfusionMatrix(classes)

print '==> downloading testing dataset'

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
test_file = paths.concat('mnist.t7', 'test_32x32.t7')

if not paths.filep(test_file) then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end


print '==> loading testing dataset'


if opt.size == 'full' then
   print '==> using regular, full testing data'
   tesize = 10000
elseif opt.size == 'small' then
   print '==> using reduced testing data, for fast experiments'
   tesize = 1000
end


loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

testData.data = testData.data:float()

mean = 25.423883007812	
std = 70.046886078148	
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)




function test()
 
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   res = {}
   -- test over test data
   print('==> generating prediction result:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      input = testData.data[t]
      input = input:double()
      pred = model:forward(input)
      --confusion:add(pred,testData.labels[t])
      res[t] = torch.nonzero(pred:eq(pred:max()))[1][1]
      
      
   end

end



test()
f = io.open('predictions.csv', 'w')
f:write('Id \tPrediction' .. '\n')

for i = 1, #res do
   f:write(i)
   f:write('\t')
   f:write(res[i])
   f:write('\n')
end
f:close()



