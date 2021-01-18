import numpy as np

class MLP :

    def __init__ ( self, qtd_in, qtd_h, qtd_out, ni=0.001 ) :
        self.qtd_in  = qtd_in
        self.qtd_h   = qtd_h
        self.qtd_out = qtd_out

        self.ni = ni

        self.wh = np.random.random (( self.qtd_in + 1, self.qtd_h ))
        self.wo = np.random.random (( self.qtd_h + 1, self.qtd_out ))

        print(self.wh.shape)
        print(self.wo.shape)

    def feed ( self, x ) :

        input_x = x.copy()
        input_x.append(1)

        def sigmoid ( x ) :
            return 1. / ( 1 + np.exp ( -x ))

        H = sigmoid ( np.dot ( np.array ( input_x ), self.wh ))

        O = sigmoid ( np.dot ( H, self.wo ))

        return O

    def treinar ( self, x, y, threshold=0.5 ) :
        
        input_x = x.copy()
        input_x.append(1)

        output_y = np.array ( y )

        def sigmoid ( x ) :
            return 1. / ( 1 + np.exp ( -x ))

        H = sigmoid ( np.dot ( np.array ( input_x ), self.wh ))

        H = np.append ( H, [1] )

        O = sigmoid ( np.dot ( H, self.wo ))

        erro = np.subtract( output_y, O )

        classif = [ 0 if output <= threshold else 1 for output in O ]

        erros_classif = np.subtract( output_y, np.array ( classif ))
        erro_classif = np.sum ( erros_classif )

        DO = O * ( np.ones ( len ( O )) - O ) * ( output_y - O )

        DH = np.dot( H * ( np.ones ( len ( H )) - H ), np.dot ( DO, np.transpose ( self.wo )))

        sh = self.ni * np.dot ( DH, np.array(input_x))   

        print(sh.shape)

        # self.wh = self.wh 
        # self.wo = self.wo + self.ni * np.dot ( DO, H )   

        return np.sum ( np.abs ( erro )), 0 if erro_classif == 0 else 1