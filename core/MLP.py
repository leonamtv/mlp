import numpy as np

class MLP :

    def __init__ ( self, qtd_in, qtd_h, qtd_out, ni=0.001 ) :
        self.qtd_in  = qtd_in
        self.qtd_h   = qtd_h
        self.qtd_out = qtd_out

        self.ni = ni

        self.wh = np.random.random (( self.qtd_in + 1, self.qtd_h ))
        self.wo = np.random.random (( self.qtd_h + 1, self.qtd_out ))

    def feed ( self, x ) :
        """
        Calcula o feed forward da rede.
        """

        input_x = x.copy()
        input_x.append(1)

        def sigmoid ( x ) :
            return 1. / ( 1 + np.exp ( -x ))

        H = sigmoid ( np.dot ( np.array ( input_x ), self.wh ))

        O = sigmoid ( np.dot ( H, self.wo ))

        return O

    def treinar ( self, x, y, threshold=0.5 ) :
        """
        Recebe uma entrada x e uma saída y e faz o treinamento.

        O parâmetro treshold define o limite de comparação para
        definirmos o erro de classificação.

        O método retorna o erro de amostragem e o erro de classi-
        ficação da amostra.
        """
       
        input_x = np.append( np.array(x.copy()), [1])
        input_x.shape = ( 1, len(input_x) )

        output_y = np.transpose ( np.array ( y ))
        output_y.shape = ( self.qtd_out, 1 )

        def sigmoid ( x ) :
            return 1. / ( 1 + np.exp ( -x ))

        H = sigmoid ( np.dot ( input_x, self.wh ))

        H = np.append ( H, [1] )

        H.shape = ( self.qtd_h + 1, 1 )
        
        O = sigmoid ( np.dot ( np.transpose( H ), self.wo ))

        O.shape = ( self.qtd_out, 1 )

        erro = np.subtract( output_y, O )

        classif = [ 0 if output <= threshold else 1 for output in O ]

        erro_classif = np.sum ( np.subtract( output_y, np.array ( classif )))

        DO = O * ( 1 - O ) * erro

        DH = H * ( 1 - H ) * np.dot ( self.wo, DO )

        for i in range ( self.qtd_in + 1 ) :
            for h in range ( self.qtd_h ) :
                self.wh[i][h] += self.ni * DH[h] * np.transpose(input_x)[i]

        for i in range ( self.qtd_h + 1 ) :
            for h in range (  self.qtd_out ) :
                self.wo[i][h] += self.ni * DO[h] * H[i]     

        return np.sum ( np.abs ( erro )), 0 if erro_classif == 0 else 1