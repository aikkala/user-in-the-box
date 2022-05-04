include <gears.scad> 
/* 
From:  https://github.com/chrisspen/gears/blob/d8779513ced77eecefb5490613ee0ab49e321055/gears.scad
License: 
OpenSCAD / Gears Library for OpenSCAD (https://www.thingiverse.com/thing:1604369) by janssen86 is licensed under the Creative Commons - Attribution - Non-Commercial - Share Alike license. http://creativecommons.org/licenses/by-nc-sa/3.0/ 
*/

scale([0.001, 0.001, 0.001])
union() {
    rotate([90, 0, 0])
    rack(2, 100, 10, 10, pressure_angle=20, helix_angle=0);
    
    translate([48, -10, -10])
    cube([35, 10, 10]);
    
    translate([-83, -10, -10])
    cube([35, 10, 10]);
    
    translate([-93, -10, -10])
    get_joint();
    
    translate([93, 0, -10])
    rotate([0, 0, 180])
    get_joint();
}

module get_joint() {
    union() {
        get_cover();
    
        translate([0, 0, 8])
        get_cover();
        
        translate([2, 5, 0])
        cylinder(10, 1.5, 1.5, $fn=30);
    } 
}

module get_cover() {
    union() {
        radius = 2;
        angles = [45, 135];
        points = [
            for(a = [angles[0]:1:angles[1]]) [radius * cos(a),      radius * sin(a)]
        ];
        translate([2, 5, 0])
        rotate([0, 0, 90])
        linear_extrude(height = 2, center = false, convexity = 10)
        polygon(concat([[0, 0]], points));
            
        translate([2, 5, 0])
        rotate([0, 0, 90])
        linear_extrude(height = 2, center = false, convexity = 10)
        polygon([
            [5, -8], 
            [radius * cos(angles[0]), radius * sin(angles[0])], 
            [radius * cos(angles[1]), radius * sin(angles[1])],
            [-5, -8]
        ]); 
    }
}
