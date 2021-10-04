##
using Revise
using GLMakie
using Pkg
Pkg.activate(".")
using ProteinStructureReconstruction
using HDF5
using Images
import Statistics: median
import GeometryBasics: Point2f0
##
filepath = "./Data/Fed_X63_Z3_SIM.h5"
h5file = h5open(filepath, "r")
##
# TODO: don't load everything in memory at once -  just draw overlay first, then 
##
x_range, y_range = 2000:2300, 1600:1900;
img_data = h5file["Data"][x_range, y_range, :, :] |> x -> reinterpret(N0f16, x);