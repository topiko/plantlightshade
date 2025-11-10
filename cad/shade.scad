include <../mirrors/surface.txt>;
$fa=1; $fs = .02;
t = 0.3;

module shade2d() {

    function get_surf(t) = concat(surf, [for (i=[len(surf)-1:-1:0]) [surf[i][0], surf[i][1] + t]]);

    surf_ = get_surf(0.001);
    difference(){
        // This offset introduces some error to the computed curve...
        offset(t / 2) polygon(points=surf_);
        translate([-100,-20]) square(100);
        translate([0, t/2]) polygon(points=get_surf(10));
        }
}

module mirror3d() {
    rotate_extrude(angle=360) shade2d();
}

difference() {
    mirror3d();
    cylinder(h=200, d=4, center =true);
}


// shade2d();
