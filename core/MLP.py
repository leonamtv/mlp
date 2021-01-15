import numpy as np

class MLP :

    def __init__ ( self, qtd_in, qtd_h, qtd_out ) :
        self.qtd_in  = qtd_in
        self.qtd_h   = qtd_h
        self.qtd_out = qtd_out

        self.wh = np.random.random ( self.qtd_in + 1, self.qtd_h )
        self.wo = np.random.random ( self.qtd_h + 1, self.qtd_out )

    def treinar ( self, x, y, threshold=0.5 ) :
        
        input_x = x.copy()
        input_x.append(1)

        def sigmoid ( x ) :
            return 1. / ( 1 + np.exp ( -x ))

        H = sigmoid ( np.dot ( np.array ( input_x ), self.wh ))

        O = sigmoid ( np.dot ( H, self.wo ))

        print ( O )