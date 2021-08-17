##
using Revise
using GLMakie
using Pkg
Pkg.activate(".")
using ProteinStructureReconstruction
##
a = plot(rand(10))
display(a)