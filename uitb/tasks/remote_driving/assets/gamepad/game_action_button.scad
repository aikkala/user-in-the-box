scale([0.00125, 0.00125, 0.00125])
scale([0.85, 0.85, 1])
difference() {
    minkowski() {
        cylinder(3, 3, 3, $fn=40, true);
        sphere(2, $fn=20);
    };
    translate([0, 0, -3])
    cube([10, 10, 5], true);
}



