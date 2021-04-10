# Thesis

Required Packages: <br />

plotly:         pip install plotly <br />
NURBS-Python:   pip install --user geomdl <br />

Constants:  <br />
(All distances must carry same units)<br />

wp_spacing        --    distance between way points<br />
radius            --    radius of sanding pad<br />
ecc               --    eccentricity of sanding pad (distance pressed into surface)<br />
overlap           --    desired overlap between passes<br />
direction         --    Direction of initial seed curve<br />
increasing        --    True if parametric direction inceases with euclidean direction; False otherwise<br />
equal_spacing     --    Distance between equally-spaced passes<br />
first_last_offset --    Offset distance of approach points <br />
force_pt_offset   --    Offset distance for beginning of force control<br />
num_rows          --    Number of passes for uniform trajectory<br />
num_rows_equal    --    Number of passes for equally-spaced trajectory<br />
seed_min          --    Min parametric coordinate of seed curve (0.0-1.0)<br />
seed_max          --    Max parametric coordinate of seed curve (0.0-1.0)<br />
u_start           --    Starting parametric coordinate of seed curve (offset from edge)<br />
uniform		  --	Compute uniform trajectory if true; compute standard trajectory if false <br />

First plot shows ellipses used to approximate local contact area.  Second plot shows just the trajectory on the surface <br />




		
