stringx = require('pl.stringx')
require 'nngraph'
require 'torch'
require 'io'
require 'base.lua'
ptb = require('data')

local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=0, 
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate
                max_max_epoch=13, -- final epoch
                max_grad_norm=5 -- clip when gradients exceed this norm value
                gpu  = true 
               }

if gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end


function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

state_train = {data=transfer_data(ptb.traindataset(20))}


vocab = ptb.vocab_map
vocab_hash = ptb.inverse_map
model = torch.load('model.net')


function readline()
  local line = io.read("*line")
  if string.len(line) == 0 or tonumber(stringx.split(line)[1]) == nil then
    return false, line
  else
    return true, line
  end
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function predict()
  reset_state(state_input)
  g_disable_dropout(model.rnns)

  local input_len = state_input.data:size(1)
  local predictions = transfer_data(
                          torch.zeros(predict_len + input_len)
                          )
  g_replace_table(model.s[0], model.start_s)
  
  for i = 1,input_len do
    local x = state_input.data[i]
    local y = x
    perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    -- Process prediction
    g_replace_table(model.s[0], model.s[1])
	io.write(data[i+1])
	io.write(' ')
  end

  last_predicted = torch.multinomial(torch.exp(pred[1]:float()), 1)[1]
  io.write(vocab_hash[last_predicted])
  io.write(' ')
  x = torch.zeros(20):cuda()
  for i=1, data[1]-1 do
	 x[1] = last_predicted
	 perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x,x,model.s[0]}))
	 g_replace_table(model.s[0], model.s[1])
	 last_predicted = torch.multinomial(torch.exp(pred[1]:float()), 1)[1]
	 io.write(vocab_hash[last_predicted])
	 io.write(' ')
  end
  io.write('\n')
end




function query_sentences()
  code = false
  while not code do
	  print("Query: len word1 word2 etc.")
	  code, line = readline()
  end
  data = stringx.split(line)
  data_input = torch.zeros(#data-1)
  predict_len = tonumber(data[1])
  for i=1,data_input:size(1) do
    if ptb.vocab_map[data[i+1]] == nil then
        data[i+1] = '<unk>'
    end
    data_input[i] = ptb.vocab_map[data[i+1]]
  end
  data_input = data_input:
             resize(data_input:size(1), 1):
             expand(data_input:size(1), params.batch_size)
  
  -- Create global state
  state_input = {}
  state_input.data = transfer_data(data_input)

  -- Run generator
  predictions = predict()

end

while true do
 query_sentences()
end



