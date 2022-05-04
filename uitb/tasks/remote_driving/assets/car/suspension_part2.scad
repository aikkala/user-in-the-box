scale([0.001, 0.001, 0.001])
rotate([0, 0, 180]) // Is unpleasant to do in mujoco.
union() {
    translate([-20, 5, 5])
    rotate([0, 90, 0])
    cylinder(20, 3, 3, $fn=30);
    
    cube([15, 10, 10]);
    
    translate([25, 10, 0])
    rotate([0, 0, 180])
    get_joint();
    
    steering_arm();
}

module steering_arm() {
    difference() {
        union() {
            translate([0, 0, 3])
            get_connector();
            
            translate([0, 0, 7])
            get_connector();
        }
        translate([25, -25, 0])
        cylinder(15, 1.5, 1.5, $fn=30);
    } 
}

module get_connector() {
    minkowski() {
        rotate([180, 0, 0])
    linear_extrude(height = 1, center = false, convexity = 10)
        polygon([
            [5, 0],
            [24, 25],
            [25, 25],
            [25, 24],
            [10, 0]
        ]);
        linear_extrude(height = 1, center = false, convexity = 10)
        circle(3, $fn=30);
    }
     
}

module get_joint() {
    difference() {
        union() {
            translate([0, 0, 2])
            get_cover();
        
            translate([0, 0, 6])
            get_cover();
        } 
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