using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver
using Printf
using SpecialFunctions

include("model_functions_with_xi.jl")

print("started\n")

# run to get new checkpoints with the correct int_G_weight function
# model_setup(checkpoint_name="evan_test_flow", k=0.0005)

# change to pickup from the new checkpoint path
checkpoint_loc = "evan_flow_iteration1585.jld2"
model_pickup("evan_filtered", checkpoint_file_path=checkpoint_loc, k=0.0005)