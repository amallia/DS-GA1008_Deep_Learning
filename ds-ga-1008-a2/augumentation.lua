require 'nn'
require 'image'
require 'xlua'
--require 'iterm'
dofile './provider.lua'
train = torch.load('provider.t7')
--provider = torch.load 'provider.t7'

--temp = provider.trainData.data 

function aug(data_in,n)
	origin_data = data_in.data:clone()
	origin_labels = data_in.labels:clone()
   indice = torch.randperm(data_in.data:size(1)):long():split(n)

   train_rotate = torch.Tensor(4000,3,96,96)
   for t, v in ipairs(indice) do
      rotate_para = torch.uniform(-0.2,0.2)
      scale_para = torch.uniform(100,120)
      xlua.progress(t,data_in.data:size(1)/(n))
      for i= 1,v:size(1) do
         data_in.data[v[i]] = image.crop( image.scale( image.rotate( data_in.data[v[i]], rotate_para), scale_para),'c' ,96,96) 
         data_in.labels[v[i]] = data_in.labels[v[i]] 

      end
   end
   
   data_in.data = torch.cat(data_in.data,origin_data,1)
   data_in.labels = torch.cat(data_in.labels,origin_labels,1)
   
   return data_in
end


a2 = aug(train.trainData,200)
a2 = aug(train.trainData,200)
a3 = aug(train.valData,100)
a4 = aug(train.valData,100)
train.trainData.data =  torch.cat(train.trainData.data,train.valData.data,1)
train.trainData.labels =  torch.cat(train.trainData.labels,train.valData.labels,1)

torch.save('provider_3.t7',train)




--provider.trainData.data = torch.Tensor(8000,3,96,96)
--provider.trainData.data = torch.cat(temp,train_rotate, 1)

--label = torch.cat(provider.trainData.labels, provider.trainData.labels,1)
--provider.trainData.labels = label
--provider.extraData.data = 0
--torch.save('provider_rotate.t7', provider)
