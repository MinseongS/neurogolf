# Semantic Mechanism Classes

This taxonomy is for human review.  Tasks may belong to many classes at
once; classes are hypotheses, not exclusive labels.  A class becomes an
optimization target only after at least one task verifies the mechanism.

## Color Classes

- `color_fixed` — **Fixed output colour**: Output colour can be hardcoded or selected from a fixed small set.
- `color_preserve_input` — **Preserve input colour**: Output keeps arbitrary input colours on copied/moved cells.
- `color_marker_copy` — **Copy marker/object colour**: Colour is selected from a marker, hint, or source object.
- `color_recolor_rule` — **Recolour by rule**: Input colours are mapped to different output colours.
- `color_mode_count` — **Colour by count/mode/rank**: Colour comes from count, mode, rank, argmax, or palette statistics.
- `color_palette_lut` — **Palette/LUT remap**: Colour choice is naturally a small lookup table or channel remap.

## Shape Classes

- `shape_fixed_template` — **Fixed template**: Shape is fixed or from a tiny constant template.
- `shape_copy_object` — **Copy object shape**: Shape is copied from an input object/component.
- `shape_clone_duplicate` — **Clone/duplicate shape**: A shape is repeated one or more times.
- `shape_extend_line_ray` — **Extend line/ray**: Line, ray, or span is extended from observed marks.
- `shape_bbox_rect` — **Bounding box/rectangle**: Shape is a bbox, solid rectangle, frame, or rectangular interior.
- `shape_enclosed_fill` — **Enclosed fill**: Output fills holes/enclosed background regions.
- `shape_local_stencil` — **Local stencil**: Cell state depends mainly on a local neighbourhood.
- `shape_component` — **Connected component**: Rule depends on connected components or object grouping.
- `shape_row_col_profile` — **Row/column profile**: Rule can be read from row/column counts, bands, or separators.
- `shape_template_match` — **Template match**: Small template/sprite is matched, rotated, or stamped.

## Direction Classes

- `direction_none_or_fixed` — **No/fixed direction**: No dynamic direction, or direction is a fixed constant.
- `direction_axis_aligned` — **Axis-aligned direction**: Uses horizontal/vertical rows, columns, bars, or spans.
- `direction_diagonal` — **Diagonal direction**: Uses diagonal or slanted relationships.
- `direction_marker_relative` — **Marker-relative direction**: Direction or target is inferred from marker/hint placement.
- `direction_rotation_reflection` — **Rotation/reflection candidate**: Needs orientation, flip, rotation, or dihedral candidates.
- `direction_gravity` — **Gravity/drop direction**: Objects move/drop/fall toward an edge or obstacle.

## Action Classes

- `action_copy` — **Copy**: Output copies cells/objects from input.
- `action_move_translate` — **Move/translate**: Object is shifted to a different location.
- `action_clone_repeat` — **Clone/repeat**: Object/template is repeated or tiled.
- `action_extend` — **Extend**: Existing marks are extended into lines/spans/rays.
- `action_fill` — **Fill**: Region is filled or completed.
- `action_crop_resize` — **Crop/resize**: Output crops, pads, resizes, upscales, or downscales.
- `action_erase_filter` — **Erase/filter/select**: Some input content is removed or a subset is selected.
- `action_reorder_pack` — **Reorder/pack/sort**: Objects are sorted, packed, ranked, or rearranged.

## Placement Classes

- `placement_same` — **Same position**: Output changes colour/value at same positions.
- `placement_fixed_offset` — **Fixed offset**: Placement is a constant translation/offset.
- `placement_marker_target` — **Marker target**: Placement is controlled by markers or target slots.
- `placement_grid_repeat` — **Grid repeat/tile**: Placement follows a repeated grid/lattice/tile.
- `placement_canonical` — **Canonical crop/top-left**: Output is canonicalized to a crop, bbox, or top-left origin.

## Compiler Classes

- `compiler_direct_output_algebra` — **Direct output algebra**: Emit thresholded final output without full intermediate carriers.
- `compiler_direct_onehot_gather` — **Direct one-hot gather**: Route input one-hot channels directly to output.
- `compiler_final_equal_overlay` — **Final Equal/overlay**: Carry scalar labels or masks until final Equal/Where output.
- `compiler_single_conv_qlinear` — **Single Conv/QLinearConv**: Collapse local predicates/counts into one Conv/QLinearConv family.
- `compiler_tiny_lut_gather` — **Tiny LUT/Gather**: Use small lookup tables, Gather, or channel remap.
- `compiler_einsum_symbolic` — **Einsum symbolic**: Use algebraic contraction or selector factorization.
- `compiler_roi_pool_crop` — **Roi/crop/pool primitive**: Use RoiAlign, MaxRoiPool, Resize, GridSample, or crop primitives.
- `compiler_sparse_scatter` — **Sparse scatter/edit stream**: Use sparse coordinate updates instead of full masks where safe.
- `compiler_bounded_scan` — **Bounded scan/flood-fill**: Use MaxPool/CumSum/scan, ideally cropped or compressed.
- `compiler_qlinear_uint8` — **QLinear/uint8 compression**: Replace fp32/fp16 routing with uint8/QLinear exact forms.

## Cost Classes

- `cost_mem0_param_game` — **Mem0 param game**: Memory is zero/tiny; improvements must reduce params.
- `cost_full_label_plane_floor` — **Full label plane floor**: A 30x30 scalar label/mask carrier likely dominates.
- `cost_full_onehot_floor` — **Full one-hot floor**: A 10-channel full-canvas carrier likely dominates.
- `cost_connectivity_wall` — **Connectivity wall**: Flood-fill/component connectivity is the hard cost driver.
- `cost_assignment_wall` — **Assignment wall**: Matching/correspondence/ambiguous assignment is the hard cost driver.
- `cost_exact_preserve_rewrite` — **Exact-preserve rewrite target**: Current source is exact-preserve or low-semantics and should be challenged.

