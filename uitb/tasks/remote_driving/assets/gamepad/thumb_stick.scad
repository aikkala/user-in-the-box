// The whole thumb stick.
shaft_length = 30;
scale([0.00125, 0.00125, 0.00125])
scale([0.5, 0.5, 0.5])
union() {
    // The little dots for improved grip.
    translate([13, 0, shaft_length + 0.6])
    sphere(1);
    
    translate([-13, 0, shaft_length + 0.6])
    sphere(1);
    
    translate([0, 13, shaft_length + 0.6])
    sphere(1);
    
    translate([0, -13, shaft_length + 0.6])
    sphere(1);
    
    // The head of the thumb stick.
    translate([0, 0, shaft_length])
    difference() {
        linear_extrude(height = 3, center = true, convexity = 10, twist = 0)
        circle(18);
        translate([0, 0, 89.6])
        sphere(90);
    };
    
    // The shaft of the thumb stick.
    translate([0, 0, 5])
    cylinder(shaft_length - 5, 8, 8);
    
    // The lower sphere used to mount the thumb stick
    // to the game controller body.
    difference() {
        translate([0, 0, -20])
        sphere(30);
        translate([0, 0, -100])
        cube([200, 200, 200], true);
    };
}
