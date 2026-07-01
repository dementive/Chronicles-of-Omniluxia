# gui template parsing

ERROR:
error(structure): expected value, found block
  --> [MOD] gui\technology_view.gui
79 |     {
   |     ^

CODE:
template DashedArrowLine
{
	gfxtype = linegfx
	width = 4

	animation_speed = { 0 0 }
	uv_scale = { 1.0 1.0 }
	texture = "gfx/interface/missions/arrow_texture.dds"
	mask_uv_scale = { 1.0 1.0 }
	mask = "gfx/interface/missions/arrow_alpha.dds"
}

# 