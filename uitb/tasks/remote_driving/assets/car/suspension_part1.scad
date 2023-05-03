scale([0.001, 0.001, 0.001])
union() {    
    cube([5, 10, 10]);
    
    translate([15, 10, 0])
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