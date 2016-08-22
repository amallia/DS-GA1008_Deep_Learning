
require "nn"
require 'xlua'
require 'image'
require 'unsup'



function kmeans_tri(x, k, niter, batchsize, callback, verbose)
 
end



function patchify(X_input,size,num,p,save)
	num_data = (#X_input)[1]
	num_cha = (#X_input)[2]
	num_dim = (#X_input)[3]
	res = torch.Tensor(num,num_cha,size,size):zero()
	
	local kernel = torch.Tensor({{0,-1,0},{-1,0,1},{0,1,0}}) 
	local i  = 1
	repeat
		xlua.progress(i, num)
		repeat
		item = torch.random(1,num_data)
		ite = 0
		local grad = image.convolve(X_input[item], kernel):abs()
		
		repeat
		
		x_loc = torch.random(2,num_dim-size-1)
		y_loc = torch.random(2,num_dim-size-1)
		cur_loc = {{},{x_loc,x_loc+size-1},{y_loc,y_loc+size-1}}
		gra_avg = grad:mean()
		gra_std = grad:std()
		gra_loc = grad[cur_loc]:mean()
		ite  =ite +1
		converge = (gra_loc > gra_avg +p *gra_std)
		until ite > 100 or converge
		
		until  converge
		
		tar = X_input[{{item},{},{x_loc,x_loc+size-1},{y_loc,y_loc+size-1}}][1]
		mean = tar:mean()
		std = tar:std() 
		if std == 0 then res[i] = tar:zero()
		else
		res[i] = tar:add(-mean):div(std)
		end
		i = i+1
	until i >= num
	if save then torch.save('unlab',res) end 
	return res
end

function parseData(d, numSamples, numChannels, height, width)
  res = torch.Tensor(numSamples,numChannels,height,width)
  for i =1, numSamples do
  	xlua.progress(i, numSamples)
	res[i] = d[i]
  end
  return res
end


function kmeans_data(data,n_clu,nite)
	n_data = (#data)[1]
	n_col = (#data)[2]*(#data)[4]*(#data)[3]
	data_mat = torch.reshape(data,n_data,n_col)
	return torch.reshape(unsup.kmeans(data_mat,n_clu,nite,verbose),n_clu,(#data)[2],(#data)[3],(#data)[4])
end



function norm(data)
 normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,data:size()[1] do
     xlua.progress(i, data:size()[1] )
     -- rgb -> yuv
     local rgb = data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     data[i] = yuv
  end
  -- normalize u globally:
  mean_u = data:select(2,2):mean()
  std_u = data:select(2,2):std()
  data:select(2,2):add(-mean_u)
  data:select(2,2):div(std_u)
  -- normalize v globally:
  mean_v = data:select(2,3):mean()
  std_v = data:select(2,3):std()
  data:select(2,3):add(-mean_v)
  data:select(2,3):div(std_v)
return data
end

function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end
MaxPooling = nn.SpatialMaxPooling

function evalu(data,model,n_cha,n_pix)
	res = torch.Tensor((#data)[1],n_cha,n_pix,n_pix)
	for i = 1,(#data)[1] do
		xlua.progress(i,(#data)[1])
		res[{{i},{},{},{}}] = model:forward(data[{{i},{},{},{}}])
	end
	return res
end


--data = patchify(parseData(torch.load('stl-10/extra.t7b').data[1],100000,3,96,96),3,10000,true)

print('loading extra')
data = parseData(torch.load('stl-10/extra.t7b').data[1],10000,3,96,96)

--convo layer 1
print('patching layer1')
patch = patchify(data,3,20000,1,false)
--model  =dofile("models/sample.lua")
layer = kmeans_data(patch,64,200)
torch.save('l1',layer)
vgg = nn.Sequential()
ConvBNReLU(3,64)
vgg:add(MaxPooling(4,4,4,4):ceil())
vgg:get(1).weight = layer
print('forwarding')
data = evalu(data,vgg,64,24)

--
--convo layer  2

print('patching layer2')
patch = patchify(data,3,20000,0.5,false)
layer = kmeans_data(patch,64,200)
torch.save('l2',layer)
vgg = nn.Sequential()
ConvBNReLU(64,64)
vgg:get(1).weight = layer
print('forwarding')
data = evalu(data,vgg,64,24)

--covo layer 3

print('patching layer3')
patch = patchify(data,3,20000,0.25,false)
layer = kmeans_data(patch,128,200)
torch.save('l3',layer)
vgg = nn.Sequential()
ConvBNReLU(64,128)
vgg:add(MaxPooling(3,3,3,3):ceil())
vgg:get(1).weight = layer
print('forwarding')
--data = evalu(data,vgg,128,8)
----
----convo layer 4
--
--print('patching layer4)
--patch = patchify(data,3,20000,false)
--layer = kmeans_data(patch,256,100)
--torch.save('l4',layer)
--vgg = nn.Sequential()
--ConvBNReLU(64,64)
--vgg:get(1).weight = layer
--print('forwarding')
--data = evalu(data,vgg,256,8)

----convo layer 5
--
--patch = patchify(data,3,20000,false)
--layer = kmeans_data(patch,256,100)
--torch.save('l5',layer)
--vgg = nn.Sequential()
--ConvBNReLU(64,64)
--vgg:get(1).weight = layer
--vgg:add(MaxPooling(2,2,2,2):ceil())
--data = evalu(data,vgg,256,4)
--
----covo layer 6
--
--patch = patchify(data,3,20000,false)
--layer = kmeans_data(patch,256,100)
--torch.save('l6',layer)
--vgg = nn.Sequential()
--ConvBNReLU(64,64)
--vgg:get(1).weight = layer
--data = evalu(data,vgg,256,4)
--
--
----covo layer 7
--
--patch = patchify(data,3,20000,false)
--layer = kmeans_data(patch,256,100)
--torch.save('l7',layer)
--vgg = nn.Sequential()
--ConvBNReLU(64,64)
--vgg:get(1).weight = layer
--data = evalu(data,vgg,256,4)
--
----covo layer 8
--
--patch = patchify(data,3,20000,false)
--layer = kmeans_data(patch,256,100)
--torch.save('l8',layer)
--vgg = nn.Sequential()
--ConvBNReLU(64,64)
--vgg:get(1).weight = layer
--vgg:add(MaxPooling(2,2,2,2):ceil())







