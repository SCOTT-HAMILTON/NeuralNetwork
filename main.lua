require("NeuralNetwork")

local answers = {0.5} --réponses attendues
local continue = true

neural = createNeuralNetwork(2, {2}, 1) 
--[[
  Ceci va creer un reseau de neurone avec 2 neurones d'entrées(inputs), 
  un layer de neurone caché (hidden) contenant 2 neurones, et 1 neurone 
  de sortie
]]


while (continue) do
    neural.train(answers)
    neural.guess({1, 2}) -- Donne une réponse avec 1 et 2 en entrée (input inchangé si rien n'est donné en paramètre de la fonction)
    print("result : "..neural.outputs[1].val)--affiche le resultat trouvé pour le premier neurone de sortie
    continue = false --calcul s'il est necessaire de continuer l'entrainement (le critère d'arret est d'avoir une réponse juste avec une precision de 1/100 000 000
    for i = 1, #neural.outputs do
      if (math.abs(neural.outputs[i].val - answers[i])>math.abs(neural.outputs[i].val)*0.00000001) then continue = true end
    end
  end
