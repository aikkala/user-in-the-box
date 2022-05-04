scale([0.00125, 0.00125, 0.00125])
difference() {
    union() {
        difference() {
            body_outline();
            translate([-3, 5, 33/2]) 
            d_pad_indent();
            scale([1.1, 1.1, 1])
            translate([-3, 5, 33/2])
            d_pad();
            translate([144-32, 0, 33/2])
            button_indent();
        }
        translate([22, -18, 0])
        thumb_stick_holder();
        translate([90, -18, 0])
        thumb_stick_holder();
        
        // Block representing usb-c port
        translate([(144-32)/2, 32-5, 10])
        cube([10, 10, 5], true);
    }
    body_inner();

    // Add a usb-c port for better detail
    translate([(144-32)/2, 32-2, 10])
    rotate([90, 0, 0])
    usb_c_port();
    
    // Clear left-over material inside thumb stick holders.
    translate([22, -18, 0])
    cylinder(33, 8, 8, true);
    translate([90, -18, 0])
    cylinder(33, 8, 8, true);
}

module body_outline () {
    linear_extrude(height = 33, center = true, twist = 0)
    union() {
        circle(32);
        translate([144-32, 0, 0])
        circle(32);
        translate([(144-32) / 2, (64-55) / 2, 0])
        square([144-32, 55], true);
    };
}

module usb_c_port () {
    linear_extrude(height = 7, center = true, twist = 0)
    difference() {
        hull() {
            circle(1.5, $fn=50);
            translate([5, 0, 0])
            circle(1.5, $fn=50);
        }
        hull() {
            circle(1, $fn=50);
            translate([5, 0, 0])
            circle(1, $fn=50);
        }
    }
}

module thumb_stick_holder() {
    difference() {
        minkowski() {
            cylinder(33, 8, 8, true);
            sphere(4);
        }
        translate([0, 0, 5])
        cylinder(33, 8, 8, true);
    };
}

module body_inner () {
    linear_extrude(height = 29, center = true, twist = 0)
    union() {
        circle(28);
        translate([144-32, 0, 0])
        circle(28);
        translate([(144-28) / 2, (64-55) / 2, 0])
        square([144-28, 47], true);
    };
}

module d_pad_indent () {
    linear_extrude(height = 1, center = true, twist = 0)
    circle(20);
}

module button_indent () {
    union() {
        linear_extrude(height = 1, center = true, twist = 0)
        difference() {
            circle(28);
            translate([-10, 8, 0])
            rotate([0, 0, 40])
            pill(5.5, 15);
            translate([3, -2, 0])
            rotate([0, 0, 40])
            pill(5.5, 15);
        }
        linear_extrude(height = 5, center = true, twist = 0)
        union() {
            translate([3, -2, 0])
            circle(4.5);
            translate([3 + 15 * cos(40), -2 + 15 * sin(40), 0])
            circle(4.5);
            translate([-10, 8, 0])
            circle(4.5);
            translate([-10 + 15 * cos(40), 8 + 15 * sin(40), 0])
            circle(4.5);
        }
    }
}

module pill (radius, width) {
    hull() {
        circle(radius);
        translate([width, 0, 0])
        circle(radius);
    }
}

module d_pad() {
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

}