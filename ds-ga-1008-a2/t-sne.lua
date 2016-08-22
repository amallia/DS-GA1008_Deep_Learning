m = require 'manifold'
require "cudnn"
require 'cunn'
require 'nn'
require 'image'
require 'xlua'
iterm = require "iterm"
model  = torch.load('logs/model.net')
test_ori = torch.load("test_raw.t7")
testset= torch.load('test.t7')

ind = torch.randperm(8000)[{{1,1000}}]

test_ori = test_ori:index(1,ind:long()):double()
testset.data = testset.data:index(1,ind:long()):double()
testset.labels = testset.labels:index(1,ind:long()):double()


function trans(data)
n_instance = data:size(1)
n_col = data:size(2)*data:size(3)*data:size(4)
res = torch.Tensor(n_intance,n_col)
for i = 1,n_instance do 
res[{i}] = data[{{i}}]:resize(n_col)
end
return res
end




function evalu(data,model,n_cha,n_pix)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  res = torch.Tensor((#data)[1],n_cha,n_pix,n_pix)
  local bs = 25
  for i=1,(#data)[1],bs do
     print(i)
    outputs = model:forward(data:narrow(1,i,bs):cuda()):float()
    res[{{i,i+bs-1},{},{},{}}] = outputs
  end
  return res
end


N = 1000
testset.size  = N
testset.data  = testset.data[{{1,N}}]
testset.labels = testset.labels[{{1,N}}]

x = testset.data:clone()
y = test_ori
x:resize(x:size(1), x:size(2) * x:size(3)*x:size(4))
--y:resize(y:size(1), y:size(2) * y:size(3)*y:size(4))

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
--mapped_x1 = m.embedding.tsne(x, opts)

im_size = 1024
--map_im = m.draw_image_map(mapped_x1, y:resize(1000,3,28,28), im_size, 0, true)
--image.save('raw.PNG',map_im)

model_f = nn.Sequential()
for i= 1,14 do
	model_f:add(model:get(i))
end

testset.data = evalu(testset.data,model_f,256,8)
x = testset.data:clone()
x:resize(x:size(1), x:size(2) * x:size(3)*x:size(4))
mapped_x1 = m.embedding.tsne(x, opts)
map_im = m.draw_image_map(mapped_x1, y:resize(1000,3,96,96), im_size, 0, true)





--iterm.image(map_im)


