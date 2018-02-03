require("NeuralNetwork")

local answers = {0.5} --reponses attendu
local continue = true

neural = createNeuralNetwork(2, {2}, 1) 
--[[
  Ceci va creer un reseau de neurone avec 2 neurone d'entree(inputs), 
  un layer de neurone cache (hidden) conetant 2 neurones et 1 neurone 
  de sorties
]]


while (continue) do
    neural.train(answers)
    neural.guess({1, 2}) -- Donne une reponse avec 1 et 2 en input (input inchange si rien n'est donne en parametre de la fonction)
    print("result : "..neural.outputs[1].val)--affiche le resultat trouve pour le premier neurone de sortie
    continue = false --calcul s'il est necessaire de continuer l'entrainement (le critere d'arret est d'avoir une reponse juste avec une precision 1/100 000 000
    for i = 1, #neural.outputs do
      if (math.abs(neural.outputs[i].val - answers[i])>math.abs(neural.outputs[i].val)*0.00000001) then continue = true end
    end
  end
