scale([0.00125, 0.00125, 0.00125])
difference() {

    // The cross.
    union() {
        cube([8, 30, 5], true);
        rotate([0, 0, 90])
        cube([8, 30, 5], true);
    }
    
    // The indentations on the cross.
    translate([0, 0, 3.5])
    
    // Change the uncommented line to use a different
    // resolution cut-out for the center sphere.
    sphere(3.5, $fa=5, $fs=0.5);
    //sphere(3.5, $fa=5, $fs=0.1);
    //sphere(3.5);

    translate([13, 0, 2])
    rotate([0, 0, 90])
    scale([3,3,1])
    linear_extrude(height = 1, center = true, twist = 0)
    polygon(points=[[0,0],[-1,sqrt(3)],[1,sqrt(3)]]);
    
    translate([-13, 0, 2])
    rotate([0, 0, 270])
    scale([3,3,1])
    linear_extrude(height = 1, center = true, twist = 0)
    polygon(points=[[0,0],[-1,sqrt(3)],[1,sqrt(3)]]);
    
    translate([0, 13, 2])
    rotate([0, 0, 180])
    scale([3,3,1])
    linear_extrude(height = 1, center = true, twist = 0)
    polygon(points=[[0,0],[-1,sqrt(3)],[1,sqrt(3)]]);
    
    translate([0, -13, 2])
    rotate([0, 0, 0])
    scale([3,3,1])
    linear_extrude(height = 1, center = true, twist = 0)
    polygon(points=[[0,0],[-1,sqrt(3)],[1,sqrt(3)]]);
}

