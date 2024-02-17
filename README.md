Para las redes se tienen los siguientes comentarios:

network_keras.py -- Red original traducida a keras, una eficiencia de 0.9258000254631042

network_RMS_tanh.py -- Red con optimizador RMS y función de activación tanh, una eficiencia de 0.9782000184059143, mejoro la eficiencia

network_adagrad_relu.py -- Red con optimizador adagrad y función de activación relu, una eficiencia de 0.9002000093460083, empeoro la eficancia

network_ADAM_sigmoid.py -- Red con optimizador ADAM y función de activación sigmoid, una eficancia de 0.9804999828338623, mejoro de forma significativa la eficiencia

network_ADAM_sigmoid_L1.py -- Red igual a la anterior, con regularizador l1, eficancia de 0.11349999904632568, empeoro totalmente la red

network_ADAM_sigmoid_L2.py -- Red igual a la anterior, con regularizador l2, eficancia de  0.09740000218153, destruyo totalmente la red

network_ADAM_sigmoid_L1_L2.py -- Red igual a la anterior, con regularizadorres l1 y l2, eficancia de   0.1134999990463256, empeoro totalmente la red

network_ADAM_sigmoid_L1_L2_Dropout.py -- Red igual a la anterior, con regularizadores l1, l2 y dropout, eficancia de 0.11349999904632568, empeoro totalmente la red

network_ADAM_sigmoid_Dropout.py -- Red igual a la anterior, con regularizador dropout, eficancia de 0.9775000214576721, bajo un poco la eficiencia de la red
