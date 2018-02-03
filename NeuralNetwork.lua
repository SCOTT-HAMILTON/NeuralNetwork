function createNeuralNetwork(nb_inputs_node, hidden_layers_nodes, node_output_node)
  local _neural = {}
  _neural.nb_inputs_node = 2
  _neural.hidden_layers_nodes = {}
  for i = 1, #hidden_layers_nodes do
    _neural.hidden_layers_nodes[i] = hidden_layers_nodes[i]
  end
  _neural.nb_hidden_layers = #_neural.hidden_layers_nodes
  _neural.node_output_node = 1

  _neural.inputs = {}
  _neural.hiden = {}

  _neural.learning_rate = 1

  math.randomseed(os.time())

  for i = 1, _neural.nb_inputs_node do
    _neural.inputs[i] = {}
    _neural.inputs[i].name = "input"
    _neural.inputs[i].val = math.random()
  end

  for l = 1, _neural.nb_hidden_layers do
    _neural.hiden[l] = {}
    for n = 1, _neural.hidden_layers_nodes[l] do
      _neural.hiden[l][n] = {}
      _neural.hiden[l][n].name = "hidden"
      _neural.hiden[l][n].val = 0
      _neural.hiden[l][n].bias = math.random()
      _neural.hiden[l][n].layer = l
    end
  end

  _neural.outputs = {}
  for i = 1, _neural.node_output_node do
    _neural.outputs[i] = {}
    _neural.outputs[i].val = 0
    _neural.outputs[i].bias = math.random()
    _neural.outputs[i].name = "output"
  end

  _neural.weights = {}

  for l = 1, _neural.nb_hidden_layers+1 do
    _neural.weights[l] = {}
    _neural.weights[l].layer_weights = {}
    
    local node_end_loop = {}
    if (l>_neural.nb_hidden_layers) then
      node_end_loop = _neural.outputs
    else
      node_end_loop = _neural.hiden[l]
    end
    for w = 1, #node_end_loop do
      local node_start_loop = {}
      if (l == 1) then
        node_start_loop = _neural.inputs
      else
        node_start_loop = _neural.hiden[l-1]
      end
      for s = 1, #node_start_loop do
        
        _neural.weights[l].layer_weights[#_neural.weights[l].layer_weights+1] = {}
        _neural.weights[l].layer_weights[#_neural.weights[l].layer_weights].start_node = node_start_loop[s]
        _neural.weights[l].layer_weights[#_neural.weights[l].layer_weights].end_node = node_end_loop[w]
        _neural.weights[l].layer_weights[#_neural.weights[l].layer_weights].val = math.random()
        if (node_start_loop[s].weights_start == nil) then
          node_start_loop[s].weights_start = {}
        end
        node_start_loop[s].weights_start[#node_start_loop[s].weights_start+1] = _neural.weights[l].layer_weights[#_neural.weights[l].layer_weights]
        
        if (node_end_loop[w].weights_end == nil) then
          node_end_loop[w].weights_end = {}
        end
        node_end_loop[w].weights_end[#node_end_loop[w].weights_end+1] = _neural.weights[l].layer_weights[#_neural.weights[l].layer_weights]
      end
      
    end
  end

  _neural.actFct = function(val)
    return val/(1+math.abs(val))
  end

  _neural.guess = function(inputs)
    if inputs ~= nil then
      for i = 1, _neural.nb_inputs_node do
        _neural.inputs[i].val = inputs[i]
      end
    end
    for l = 1, #_neural.hiden do
      for n = 1, #_neural.hiden[l] do
        _neural.hiden[l][n].val = 0
      end
    end
    for i = 1, #_neural.outputs do
      _neural.outputs[i].val = 0
    end
    for l = 1, #_neural.weights do
      for w = 1, #_neural.weights[l].layer_weights do
        local val = _neural.weights[l].layer_weights[w].start_node.val*_neural.weights[l].layer_weights[w].val
        _neural.weights[l].layer_weights[w].end_node.val = _neural.weights[l].layer_weights[w].end_node.val+val
    
      end
    end
    
    for l = 1, #_neural.hiden do
      for n = 1, #_neural.hiden[l] do
        _neural.hiden[l][n].val = _neural.actFct(_neural.hiden[l][n].val+_neural.hiden[l][n].bias)
      end
    end
    for i = 1, #_neural.outputs do
      _neural.outputs[i].val = _neural.actFct(_neural.outputs[i].val+_neural.outputs[i].bias)
    end
  end

  _neural.propagate = function(node, weight)
    --print("prop !!")
    if node.name == "hidden" then
      local w_sum = 0
      for w = 1, #weight.end_node.weights_end do
        w_sum = w_sum+weight.end_node.weights_end[w].val
      end
      node.err = (weight.val/w_sum)*weight.end_node.err
      weight.val = weight.val+(node.err*node.val*_neural.learning_rate)
    end
    if (node.weights_end == nil) then
      return false
    end
    for i = 1, #node.weights_end do
      _neural.propagate(node.weights_end[i].start_node, node.weights_end[i])
    end
  end

  _neural.train = function(answers)
    for i = 1, #_neural.outputs do
      _neural.outputs[i].err = answers[i]-_neural.outputs[i].val
      _neural.propagate(_neural.outputs[i], nil)
    end
  end
  
  return _neural
end