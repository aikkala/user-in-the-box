scale([0.001, 0.001, 0.001])
rotate([0, 0, -90])
difference() {
    minkowski() {
        chassis_base();
        sphere(50, $fn=80);
    }
    translate([0, 0, -60])
    cube([1000, 1000, 80], center = true);
    wheel_cutouts();
}

module chassis_base() {
    union() {
        rotate([90, 0, 0])
        linear_extrude(130, center=true)
        polygon([
            [0, 0],
            [90, 40],
            [90, 0]
        ]);
        translate([90, -65, 0])
        cube([250, 130, 40]);
        //wheel_cutouts();
    }
}

module wheel_cutouts() {
    radius = 40;
    wheelbase = 110;
    inset = 22;
    
    x_front = 60;
    x_back = 318;
    
    translate([0, 0, -18])
    union() {
        // Cutouts for the front wheels.
        translate([x_front, wheelbase, 0])
        wheel_cutout(inset, radius);
        translate([x_front, -wheelbase, 0])
        wheel_cutout(inset, radius);
        
        // Cutouts for the back wheels.
        translate([x_back, wheelbase, 0])
        wheel_cutout(inset, radius);
        translate([x_back, -wheelbase, 0])
        wheel_cutout(inset, radius);
    }
}

module wheel_cutout(inset, radius) {
    rotate([90, 0, 0])
    cylinder(inset, radius, radius, center = true);
}