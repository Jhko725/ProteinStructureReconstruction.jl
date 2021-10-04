##
using Revise
using GLMakie
using Pkg
Pkg.activate(".")
using ProteinStructureReconstruction
using Images
using HDF5


##
module ZStackIO
	
	import Images
    using RecursiveArrayTools: VectorOfArray
	export find_zstack_folder_by_number, read_STORM_folder, BoundingBox, process_STORM_data
	unravel_array(vec_arr) = convert(Array, VectorOfArray(vec_arr))

	function find_zstack_folder_by_number(dir::AbstractString = "./"; nb::Integer)
		name_regex = Regex("^Stack $(nb)") 
		for path in readdir(dir) 
			if occursin(name_regex, path) && isdir(dir)
				return joinpath(dir, path)
			end
		end
		error("The z stack with the given number does not exist!")
	end

	
	function read_STORM_folder(dir::AbstractString = "./")
		root, folders, files = first(walkdir(dir))
        print(folders)
		names = map(target_name_from_foldername, folders)
		image_stacks = map(x -> read_subfolder(root, x), folders)

		if "overlay.tif" in files
			overlay = Images.load(joinpath(root, "overlay.tif"))
		else
			error("overlay image does not exist!")
		end

		return overlay, names, image_stacks
	end

	function read_subfolder(root, subdir::AbstractString)
		files = readdir(joinpath(root, subdir), join = true)
		sort!(files, by = slice_nb_from_filename)
		images = map(Images.load, files)
		return images
	end
	
	function slice_nb_from_filename(filename::AbstractString)
		m = match(r"z(?<slice_nb>\d+) \d+.tif$", filename)
		return parse(Int, m[:slice_nb])
	end
	
	function target_name_from_foldername(foldername::AbstractString)
		re_test = match(r"^(?<exp_name>[A-Z, a-z]*)_[\w, \s]*", foldername)
		return re_test[:exp_name]
	end


	struct BoundingBox{T<:Integer}
		lower_left::Tuple{T, T}
		upper_right::Tuple{T, T}
	end
	BoundingBox(diagonal_vertices::AbstractVector) = BoundingBox(diagonal_vertices...)
	
	Base.size(bb::BoundingBox) = bb.upper_right .- bb.lower_left .+ 1
	Base.size(bb::BoundingBox, dim::Integer) = Base.size(bb)[dim]
	
	function Base.to_indices(A::AbstractMatrix, inds, I::Tuple{BoundingBox})
		x_min, y_min = I[1].lower_left
		x_max, y_max = I[1].upper_right
		return Base.to_indices(A, inds, (x_min:x_max, y_min:y_max))
	end
	

	function find_scale_bar_length(overlay_grayed::AbstractMatrix)
		iswhite = isequal.(overlay_grayed, 1.0)
		white_components = Images.label_components(iswhite)
		bboxes = BoundingBox.(Images.component_boxes(white_components))

		# first bbox is for the null label (label = 0)
		bbox_sizes = size.(bboxes[2:end]) 

		# scale bar has smallest x (vertical) size
		scale_bar_index = argmin(first.(bbox_sizes)) 

		scale_bar_length = bbox_sizes[scale_bar_index][2]
		return scale_bar_length
	end
	
	function find_image_bbox(overlay_grayed::AbstractMatrix)
	# know apriori that the non image features (scale, colorbar) can be cropped out with this choice of indexing
	overlay_cropped = overlay_grayed[:, 1:1500]
	
	# then find the bounding box for the relevant image region
	isnonzero = Int.(overlay_cropped .> 0.0)
	bboxes = BoundingBox(Images.component_boxes(isnonzero)[end])
	end


	function process_STORM_data(overlay::AbstractMatrix, names, image_stacks) 
		overlay_grayed = Images.Gray.(overlay)
		
		scale_bar_len = find_scale_bar_length(overlay_grayed)
		bbox = find_image_bbox(overlay_grayed)
		
		cropped_stacks = unravel_array(map(x -> crop_image_stack(x, bbox), image_stacks))
        return cropped_stacks, scale_bar_len, names
	end
	
	function crop_image_stack(image_stack::AbstractVector{T}, bbox::BoundingBox) where {T <: AbstractMatrix}
		cropped_stack = map(img -> getindex(img, bbox), image_stack)
		return unravel_array(cropped_stack)
	end
end
##
stack, scale_bar_len, names = ZStackIO.find_zstack_folder_by_number("./Data/STORM/", nb = 2) |> ZStackIO.read_STORM_folder |> args -> ZStackIO.process_STORM_data(args...);

##
scale = Float32.([5.0*10^-6/scale_bar_len, 5.0*10^-6/scale_bar_len, 2.0*10^-7])
##
rawview.(stack)
##
filename = "./Data/STORM_stack2.h5"
fid = h5open(filename, "w")