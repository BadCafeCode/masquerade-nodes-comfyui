# Masquerade Nodes

This is a node pack for ComfyUI, primarily dealing with masks. Some example workflows this pack enables are:

(Note that all examples use the default 1.5 and 1.5-inpainting models. Results are generally better with fine-tuned models.)

1. Fine control over composition via automatic photobashing (see `examples/composition-by-photobashing.json`)
![CompositionExample](https://user-images.githubusercontent.com/3157454/233810990-4f873506-90c4-4594-ad7f-def400c54885.png)
2. Inpaint all faces at a higher resolution (see `examples/inpaint-faces.json`)
![FaceInpaintingExample](https://user-images.githubusercontent.com/3157454/233810991-dc56d0cb-7477-4e7e-aa7c-b822612c42e5.png)
3. Inpaint all buildings with a particular LORA (see `examples/inpaint-with-lora.json`)
![LoRAInpaintingExample](https://user-images.githubusercontent.com/3157454/233810996-a4e2f30b-a9a6-414f-9e6f-054da37d02ff.png)
4. Filtering out images/change save location of images that contain certain objects/concepts without the side-effects caused by placing those concepts in a negative prompt (see `examples/filter-by-season.json`)
![FilterImagesExample](https://user-images.githubusercontent.com/3157454/233810992-849858e5-64d4-46d4-9d7f-afa20dde622f.png)

The lynchpin of these workflows is the [Mask by Text](#mask-by-text) node. This node makes use of ClipSeg to dynamically create masks from images via text prompts.

# Node Descriptions

## Mask By Text
#### Inputs
* `image` - The image to use to calculate the mask for.
* `prompt` - One or more prompts to select, separated by the `|` symbol.
* `negative_prompt` - Zero or more prompts to select, separated by the `|` symbol.
* `precision` - How sure to be about the mask created.
* `normalize` - Whether results should be normalized across the image.
	* Note - This can be useful if you know for certain that your prompt exists in the image, but the model inconsistently recognizes it. I would generally recommend trying with normalize turned off first.
#### Outputs
1. Thresholded Mask - The calculated mask after taking precision and normalization into account.
2. Raw ClipSeg Results - The raw results from ClipSeg. Values are between 0.0 and 1.0.

#### Tips
* Using multiple prompts/negative prompts within one node is an ease-of-use and runtime optimization feature. The same result can be achieved with multiple Mask by Text nodes by using `Combine Masks -> Union` and `Combine Masks -> Difference`.
* You can use the raw ClipSeg results for more complex operations (like comparing the relative likelihood of two concepts).

## Mask Morphology
#### Inputs
* `image` - The mask to perform the operation on.
* `distance` - The distance across which to perform the operation.
* `op` - The operation to perform.
	* `dilate` - Expand the size of the mask.
	* `erode` - Contract the size of the mask.
	* `open` - Erode and then dilate. Useful to remove white specs in the mask.
	* `close` - Dilate and then erode. Useful to remove black specs in the mask.

#### Outputs
1. The resultant mask.

#### Tips
* Open is useful to remove white specs in a mask. You should generally use it before a Separate Mask Components node.
* Close is useful to remove black specs in a mask.

## Combine Masks
#### Inputs
* `image1` - The first mask to use.
* `image2` - The second mask to use.
* `op` - The operation to perform.
	* `union (max)` - The maximum value between the two masks.
	* `intersection (min)` - The minimum, value between the two masks.
	* `difference` - The pixels that are white in the first mask but black in the second.
	* `multiply` - The result of multiplying the two masks together. Same as intersection for any thresholded masks.
	* `multiply_alpha` - An alpha channel is added to the first image if required and then the second image (treated as a mask) is multiplied on top.
	* `add` - The result of adding the two masks together. Same as union for any thresholded masks (if clamped).
	* `greater_or_equal` - image1 >= image1.
	* `greater` - image1 > image1. Same as difference for any thresholded masks.
* `clamp_result` - Whether results should be clamped between 0 and 1. Usually you want this.
* `round_result` - Whether results should be rounded to 0 or 1.

#### Outputs
1. The resultant mask or image.
 
#### Tips
* This node can also be used with RGB/RGBA images. This can be useful to alpha out part of an image based on a mask (via multiplication).

## Unary Mask Op
#### Inputs
* `image` - The mask to operate on.
* `op` - The operation to perform.
	* `invert` - Inverts the mask.
	* `average` - Sets the value of all pixels in the mask to be equal to the value of the average pixel.
	* `round` - Rounds all pixels to the nearest integer (generally 0 or 1).
	* `clamp` - Clamps all values between 0 and 1.
	* `abs` - Takes the absolute value of all pixels.
 
#### Outputs
1. The resultant mask.

#### Tips
* Average is deceptively useful. For example, you can use two Mask to Text nodes with different concepts (like 'New York' and 'Chicago') and average the raw ClipSeg result to see which city the image is more likely to depict.

## Unary Image Op
This node is the same as Unary Mask Op, but will operate across all channels of an image individually. This can be particularly useful after converting to HSV colorspace.
#### Inputs
* `image` - The image to operate on.
* `op` - The operation to perform.
	* `invert` - Inverts all channels of the image.
	* `average` - Sets the value of all pixels in the image to be equal to the value of the average pixel in that channel.
	* `round` - Rounds all pixels to the nearest integer (generally 0 or 1).
	* `clamp` - Clamps all values between 0 and 1.
	* `abs` - Takes the absolute value of all pixels.
 
#### Outputs
1. The resultant image.

## Blur
#### Inputs
* `image` - The image to blur.
* `radius` - How much to blur the image.
* `sigma_factor` - Control the falloff speed of the blur.

#### Outputs
1. The resultant mask or image.

#### Tips
* This node can be used on both masks and images.
* You can usually leave the sigma factor at 1.

## Image To Mask
#### Inputs
* `image` - The image to convert.
* `method` - The method to use for conversion.
	* `intensity` - Uses the grayscale color of the image.
	* `alpha` - Uses the alpha mask of the image.

#### Outputs
1. The resultant mask in a form usable by inpainting.

## Mix Images By Mask
#### Inputs
* `image1` - The first image.
* `image2` - The second image.
* `mask` - The mask to use to interpolate between the two images.

#### Outputs
1. The resultant image.

## Mix Color By Mask
#### Inputs
* `image` - The first image.
* `r, g, b` - The red, green and blue components to use for the color.
* `mask` - The mask to use to interpolate between the image and the color.

#### Outputs
1. The resultant image.

## Mask To Region
#### Inputs
* `mask` - The mask to use to calculate the region for.
* `padding` - How much padding to add around the mask.
* `constraints` - Once we determine a minimum size, the mask will be grown until it meets the specified constraint.
	* `keep_ratio` - The region's ratio will be the same as the ratio between `constraint_x` and `constraint_y`.
	* `keep_ratio_divisible` - The region's ratio will be the same as the ratio between `constraint_x` and `constraint_y`. Additionally, the region's width will divide evenly into `constraint_x` and the region's height will divide evenly into `constraint_y`.
	* `multiple_of` - The region's width will be a multiple of `constraint_x` and the region's height will be a multiple of `constraint_y`.
	* `ignore` - No special constraints (`constraint_x` and `constraint_y` are meaningless).
* `constraint_x` - See constraints.
* `constraint_y` - See constraints.
* `min_width` - The minimum width of the region.
* `min_height` - The minimum height of the region.
* `batch_behavior` - How to handle batches of multiple images/masks.
	* `match_ratio` - The region will have the same ratio for all masks. Additionally, all masks will be scaled to the size of the largest mask with bicubic filtering applied.
	* `match_size` - All regions will grow to the size of the largest mask.

#### Outputs
1. A mask of the resultant region.

#### Tips
* This node is particularly useful for an inpainting at full resolution. You can use it to select a reasonable area around the masked area you want to inpaint.

## Cut By Mask
#### Inputs
* `image` - The image or mask to cut the masked area out of.
* `mask` - The mask specifying the area to cut.
* `force_resize_width` - If non-zero, the image will be resized to this width.
* `force_resize_height` - If non-zero, the image will be resized to this height.
* `mask_mapping_optional` - If there are a variable number of masks for each image (due to use of [Separate Mask Components](#separate-mask-components)), use the mask mapping output of that node to cut the masks out of the correct image. Leave this unused otherwise.

#### Outputs
1. The resultant image or mask.

#### Notes
* If there are multiple masks/images and `force_resize_width` and `force_resize_height` are zero, all images will be resized to the size of the largest.
* If the mask is not rectangular, alpha will be applied to areas outside of the mask. If you paste the image back with [Paste By Mask](#paste-by-mask), areas with 0 alpha will not be pasted.

#### Tips
* You will frequently want to use [Mask To Region](#mask-to-region) to generate the mask to cut out. You can then cut out both the image and the original mask to get your inpainting target.
* If you are using a batch size greater than one, you should use `keep_ratio` when creating the region to ensure that masks aren't distorted due to resizing.

## Paste By Mask
#### Inputs
* `image_base` - The image to paste the mask into.
* `image_to_paste` - The image to paste.
* `mask` - The mask (over the base image) specifying where to paste the image.
* `resize_behavior` - How to handle pasted images that don't match the size of the area to paste.
	* `resize` - Resize the image to match the size of the area to paste.
	* `keep_ratio_fill` - Resize the image to match the size of the region to paste while preserving aspect ratio. The resize will extent outside the masked area.
	* `keep_ratio_fit` - Resize the image to match the size of the region to paste while preserving aspect ratio. The resize will be entirely contained within the masked area.
	* `source_size` - Use the size of the image to paste as the size of the region to paste. It will be centered on the masked area and areas outside of the mask will not be pasted.
	* `source_size_unmasked` - Use the size of the image to paste as the size of the region to paste. It will be centered on the masked area and may extend outside the masked area.
* `mask_mapping_optional` - If there are a variable number of masks for each image (due to use of [Separate Mask Components](#separate-mask-components)), use the mask mapping output of that node to paste the masks into the correct image. Leave this unused otherwise.
#### Outputs
1. The resultant image.

#### Tips
* Because modifying the paste mask to be different than the mask used to cut may alter the size/position at which it is pasted, you can add alpha to the `image_to_paste` (e.g. via [Combine Masks](#combine-masks) -> `multiply_alpha`) to paste only some parts.

## Get Image Size
#### Inputs
* `image` - The image to get the size of.

#### Outputs
1. The width of the image.
2. The height of the image.

#### Notes
* This utility function can be useful to propagate sizes through a graph so they only have to be updated at the original latent image location.

## Change Channel Count
#### Inputs
* `image` - The image to change the channel count of.
* `kind`
	* `mask` - Convert the image to one channel. If there is variable alpha, alpha will be used. Otherwise, image intensity will be used.
	* `RGB` - Convert the image to three channels. Any alpha is discarded.
	* `RGBA` - Convert the image to four channels. If there are already 3 channels, the image is purely opaque. If there is only 1 channel, all 4 channels will be equal to that value (so both intensity and alpha conversions back to a mask will result in the same value).

#### Outputs
1. The resultant image.

#### Notes
* All nodes in this pack will perform automatic conversions to appropriate channel counts. You only need to use this node when interfacing with an external node that assumes a specific count. (For example, inpainting requires `RGB` images.)

## Constant Mask
#### Inputs
* `value` - The value to use for the mask.
* `explicit_height` - The explicit height of the mask. Will only be used if `copy_image_size` is empty.
* `explicit_width` - The explicit width of the mask. Will only be used if `copy_image_size` is empty.
* `copy_image_size` - If specified, the mask will have the same size as the given image.

#### Outputs
1. The resultant mask.

#### Tips
* This is generally used with [Combine Masks](#combine-masks).

## Prune By Mask
#### Inputs
* `image` - The batch of images to prune.
* `mask` - Masks determining which images to keep.

#### Outputs
1. The pruned batch of images. Only images with an associated mask with an average value greater than 0.5 will be included.

#### Tips
* This node is only useful when using batch sizes > 1.

## Separate Mask Components
#### Inputs
* `mask` - The mask to separate into contiguous components.

#### Outputs
1. A batch of masks. Each contiguous area of the mask will be separated into its own mask. Any values greater than 0 are considered part of a contiguous area.
2. A mapping from the original mask to the new masks. This is useful when using [Cut By Mask](#cut-by-mask) and [Paste By Mask](#paste-by-mask) when a batch size larger than one was fed into this node.

## Create Rect Mask
#### Inputs
* `mode`
	* `percent` - Input values (`x`, `y`, `width`, and `height`) are treated as a percentage of the overall image (for example, 50.0 is half the image).
	* `pixels` - Input values (`x`, `y`, `width`, and `height`) are treated as pixel values.
* `origin` - The origin to use for `x` and `y` (`topleft`, `bottomleft`, `topright`, `bottomright`).
* `x` - The x offset from the selected `origin` of the rect.
* `y` - The y offset from the selected `origin` of the rect.
* `width` - The width of the rect.
* `height` - The height of the rect.
* `image_width` - The width of the overall image to use. Will only be used if `copy_image_size` is empty.
* `image_height` - The height of the overall image to use. Will only be used if `copy_image_size` is empty.
* `copy_image_size` - If specified, the mask will have the same size as the given image.

## Create QR Code
#### Inputs
* `text` - The content to embed in the QR Code
* `size` - The size of the QR Code (across height and width) in pixels.
* `qr_version` - The version of QR Code to use. Higher versions can encode more data, but are larger.
* `error_correction` - The level of error correction to use.
* `box_size` - The size of each box in the QR Code in pixels.
* `border` - The size of the border around the QR Code in pixels.

## Convert Color Space
#### Inputs
* `image` - The image to convert.
* `in_space` - The color space of the input image -- valid values are `RGB`, `HSV`, and `HSL`.
* `out_space` - The color space of the output image -- valid values are `RGB`, `HSV`, and `HSL`.

## Incrementer
#### Inputs
* `seed` - The current value.
* `control_after_generate` - Set to "Increment" to actually increment.
* `max_value` - The value to perform modulo against
