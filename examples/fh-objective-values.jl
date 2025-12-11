using DifferentialEquations, RegularizedProblems, ADNLPModels, NLPModels
using LinearAlgebra
data, simulate, resid, misfit, x0 = RegularizedProblems.FH_smooth_term()
model = ADNLPModel(misfit, 1 / 10 * ones(5); matrix_free=true)

truth = [0.0, 0.20, 1.0, 0.0, 0.0]
x_ir2n = [0.0, 0.20, 0.991, -0.009, 0.0]
x_r2n = [0.0, 0.28, 0.78, 0.00, 0.0]
x_ir2 = [0.0, 0.20, 0.98, -0.01, 0.0]
x_r2 = [0.0, 0.26, 0.84, 0.00, 0.0]

for x in [truth, x_ir2n, x_r2n, x_ir2, x_r2]
  fk = obj(model, x)
  println("fk for $x = $fk")
end

1 / 2 * norm(resid(x_ir2n))^2