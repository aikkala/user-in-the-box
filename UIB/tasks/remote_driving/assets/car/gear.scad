include <gears.scad> 
/* 
From:  https://github.com/chrisspen/gears/blob/d8779513ced77eecefb5490613ee0ab49e321055/gears.scad
License: 
OpenSCAD / Gears Library for OpenSCAD (https://www.thingiverse.com/thing:1604369) by janssen86 is licensed under the Creative Commons - Attribution - Non-Commercial - Share Alike license. http://creativecommons.org/licenses/by-nc-sa/3.0/ 
*/

scale([0.001, 0.001, 0.001])
rotate([90, 0, 0])
spur_gear(2, 20, 10, 5, pressure_angle=20, helix_angle=0, optimized=true);
