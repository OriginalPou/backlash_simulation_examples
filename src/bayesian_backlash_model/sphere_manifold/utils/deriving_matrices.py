from sympy import sin, cos, Pow, symbols, MatrixSymbol, Matrix, pprint, simplify
import numpy as np

sym_c_angles_ = MatrixSymbol('c_angles', 2, 1)
sym_alpha = sym_c_angles_[0,0]
sym_beta  = sym_c_angles_[1,0]

L_t  = symbols('L_t')
L_f  = symbols('L_f')
D = symbols('D')

sym_t_base_to_tip_ = Matrix([
            [(L_t * sin(sym_beta) + L_f/sym_beta * (1 - cos(sym_beta)) ) * cos(sym_alpha)],
            [(L_t * sin(sym_beta) + L_f/sym_beta * (1 - cos(sym_beta)) ) * sin(sym_alpha)],
            [L_t * cos(sym_beta) + L_f/sym_beta * sin(sym_beta)]
        ])

sym_R_base_to_tip_ = Matrix([
            [Pow(sin(sym_alpha),2) + cos(sym_beta)*Pow(cos(sym_alpha),2),   -sin(sym_alpha)*cos(sym_alpha)*(1-cos(sym_beta)),               cos(sym_alpha)*sin(sym_beta)],
            [-sin(sym_alpha)*cos(sym_alpha)*(1-cos(sym_beta)),              Pow(cos(sym_alpha),2) + cos(sym_beta)*Pow(sin(sym_alpha),2),    sin(sym_alpha)*sin(sym_beta)],
            [-cos(sym_alpha)*sin(sym_beta),                                 -sin(sym_alpha)*sin(sym_beta),                                  cos(sym_beta)]
        ])

sym_jacobian_t_base_to_tip_ = sym_t_base_to_tip_.jacobian(sym_c_angles_)
angular_angular_wrt_alpha = simplify( sym_R_base_to_tip_.diff(sym_alpha) @ sym_R_base_to_tip_)
pprint(angular_angular_wrt_alpha)
#pprint(np.array([angular_angular_wrt_alpha[2, 1], angular_angular_wrt_alpha[0, 2], angular_angular_wrt_alpha[1, 0]]))